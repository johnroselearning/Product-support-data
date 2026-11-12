"""Optional train-only feature engineering for classical support-ticket baselines.

This exporter never edits canonical or SFT records. It fits all transformations on
the training split, then applies them unchanged to validation and test.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, PolynomialFeatures, RobustScaler,
    StandardScaler, TargetEncoder
)

SPLITS=("train","validation","test")
TEXT_NUMERIC_FIELDS=(
    "text_chars","text_words","text_lines","question_marks","exclamation_marks",
    "text_missing","has_http_status","has_stack_trace","has_error_keyword",
    "has_log_level","has_http_method","has_url"
)
PATTERNS={
    "has_http_status":re.compile(r"\b(?:4\d\d|5\d\d)\b"),
    "has_stack_trace":re.compile(r"(?i)\b(?:traceback|stack trace|exception|caused by:)\b"),
    "has_error_keyword":re.compile(r"(?i)\b(?:error|failed|failure|timeout|timed out|denied)\b"),
    "has_log_level":re.compile(r"(?i)\b(?:debug|info|warn|warning|error|fatal|critical)\b"),
    "has_http_method":re.compile(r"(?i)\b(?:GET|POST|PUT|PATCH|DELETE|OPTIONS)\s+/"),
    "has_url":re.compile(r"(?i)\bhttps?://[^\s<>{}]+")
}

def _get_path(record,path):
    value=record
    for part in path.split("."):
        if not isinstance(value,dict):
            return None
        value=value.get(part)
    return value

def _datetime_features(value):
    parsed=pd.to_datetime(value,errors="coerce",utc=True)
    if pd.isna(parsed):
        return {
            "datetime_missing":1.0,"hour_sin":float("nan"),"hour_cos":float("nan"),
            "weekday_sin":float("nan"),"weekday_cos":float("nan"),
            "month_sin":float("nan"),"month_cos":float("nan"),"is_weekend":float("nan")
        }
    hour=parsed.hour+parsed.minute/60.0+parsed.second/3600.0
    weekday=parsed.dayofweek
    month=parsed.month-1
    return {
        "datetime_missing":0.0,
        "hour_sin":math.sin(2*math.pi*hour/24.0),
        "hour_cos":math.cos(2*math.pi*hour/24.0),
        "weekday_sin":math.sin(2*math.pi*weekday/7.0),
        "weekday_cos":math.cos(2*math.pi*weekday/7.0),
        "month_sin":math.sin(2*math.pi*month/12.0),
        "month_cos":math.cos(2*math.pi*month/12.0),
        "is_weekend":float(weekday>=5)
    }

def _feature_row(record,categorical_fields,datetime_field):
    ticket=record.get("ticket") or {}
    raw_text=ticket.get("description")
    text=raw_text if isinstance(raw_text,str) else ""
    row={
        "description":text,
        "text_chars":len(text),
        "text_words":len(text.split()),
        "text_lines":len(text.splitlines()),
        "question_marks":text.count("?"),
        "exclamation_marks":text.count("!"),
        "text_missing":float(not bool(text.strip()))
    }
    for name,pattern in PATTERNS.items():
        row[name]=float(bool(pattern.search(text)))
    for field in categorical_fields:
        value=_get_path(record,field)
        row[field]=value if value is not None and str(value).strip() else float("nan")
    if datetime_field:
        row.update(_datetime_features(_get_path(record,datetime_field)))
    return row

def _load_split(input_dir,split,label_key,categorical_fields,datetime_field):
    path=input_dir/"master"/f"{split}.jsonl"
    if not path.exists():
        return pd.DataFrame(),[],[],{"missing_target":0,"privacy_flagged":0}
    rows=[]
    labels=[]
    case_ids=[]
    excluded={"missing_target":0,"privacy_flagged":0}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record=json.loads(line)
            if record.get("quality",{}).get("privacy_flags"):
                excluded["privacy_flagged"]+=1
                continue
            label=(record.get("labels") or {}).get(label_key)
            if label is None or not str(label).strip():
                excluded["missing_target"]+=1
                continue
            rows.append(_feature_row(record,categorical_fields,datetime_field))
            labels.append(str(label).strip())
            case_ids.append(str(record.get("case_id","")))
    return pd.DataFrame(rows),labels,case_ids,excluded

def _build_preprocessor(args,n_train):
    if args.scaling=="standard":
        scaler=StandardScaler()
    elif args.scaling=="minmax":
        scaler=MinMaxScaler()
    elif args.scaling=="robust":
        scaler=RobustScaler()
    else:
        scaler="passthrough"
    if args.imputation=="knn":
        imputer=KNNImputer(n_neighbors=max(1,min(5,n_train)))
    else:
        imputer=SimpleImputer(strategy=args.imputation,keep_empty_features=True)
    numeric_pipeline=FeatureUnion([
        ("values",Pipeline([
            ("imputer",imputer),
            ("polynomial",PolynomialFeatures(degree=args.polynomial_degree,include_bias=False)),
            ("scaler",scaler)
        ])),
        ("missing_indicators",MissingIndicator(features="all",error_on_new=False))
    ])
    categorical_encoder=OneHotEncoder(handle_unknown="ignore")
    if args.categorical_encoding=="target":
        categorical_encoder=TargetEncoder(cv=max(2,min(5,n_train)),shuffle=True,random_state=42)
    categorical_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="constant",fill_value="__MISSING__",keep_empty_features=True)),
        ("encoder",categorical_encoder)
    ])
    numeric_fields=list(TEXT_NUMERIC_FIELDS)
    if args.datetime_field:
        numeric_fields.extend([
            "datetime_missing","hour_sin","hour_cos","weekday_sin","weekday_cos",
            "month_sin","month_cos","is_weekend"
        ])
    return ColumnTransformer(
        transformers=[
            ("text_tfidf",TfidfVectorizer(
                ngram_range=(1,2),min_df=args.min_df,max_features=args.max_features,
                sublinear_tf=True
            ),"description"),
            ("numeric",numeric_pipeline,numeric_fields),
            ("categorical",categorical_pipeline,list(args.categorical_fields))
        ],
        remainder="drop",sparse_threshold=1.0,verbose_feature_names_out=True
    )

def _save_split(output_dir,split,matrix,labels,case_ids):
    if not sparse.issparse(matrix):
        matrix=sparse.csr_matrix(matrix)
    matrix=sparse.csr_matrix(matrix)
    sparse.save_npz(output_dir/f"X_{split}.npz",matrix)
    (output_dir/f"y_{split}.json").write_text(json.dumps(labels,ensure_ascii=False,indent=2)+"\n")
    (output_dir/f"case_ids_{split}.json").write_text(json.dumps(case_ids,ensure_ascii=False,indent=2)+"\n")
    return {"records":len(labels),"matrix_shape":[int(matrix.shape[0]),int(matrix.shape[1])],
            "label_counts":dict(Counter(labels))}

def run(args):
    if args.polynomial_degree not in (1,2,3):
        raise ValueError("--polynomial-degree must be 1, 2 or 3.")
    if args.min_df<1 or args.max_features<1:
        raise ValueError("--min-df and --max-features must be positive.")
    if not args.categorical_fields:
        raise ValueError("At least one categorical field is required.")
    if any(field=="labels" or field.startswith("labels.") for field in args.categorical_fields):
        raise ValueError("Target label fields cannot be used as categorical input features.")
    loaded={
        split:_load_split(args.input,split,args.label_key,args.categorical_fields,args.datetime_field)
        for split in SPLITS
    }
    train_frame,train_labels,_,train_excluded=loaded["train"]
    if train_frame.empty:
        raise ValueError("No labeled, non-privacy-flagged training records were found.")
    if args.categorical_encoding=="target" and len(set(train_labels))<2:
        raise ValueError("Target encoding needs at least two target classes in training.")
    if args.datetime_field and not train_frame["datetime_missing"].eq(0).any():
        raise ValueError(
            f"No parseable training timestamps found at '{args.datetime_field}'. "
            "The current canonical schema has no datetime field; map one in an adapter first."
        )
    preprocessor=_build_preprocessor(args,len(train_labels))
    args.output.mkdir(parents=True,exist_ok=True)
    split_summary={}
    for split,(frame,labels,case_ids,excluded) in loaded.items():
        if frame.empty:
            split_summary[split]={"records":0,"excluded":excluded,"matrix_shape":None}
            continue
        if split=="train":
            matrix=preprocessor.fit_transform(frame,labels)
        else:
            matrix=preprocessor.transform(frame)
        info=_save_split(args.output,split,matrix,labels,case_ids)
        info["excluded"]=excluded
        split_summary[split]=info
    joblib.dump(preprocessor,args.output/"preprocessor.joblib")
    manifest={
        "format_version":"1.0",
        "label_key":args.label_key,
        "categorical_fields":list(args.categorical_fields),
        "categorical_encoding":args.categorical_encoding,
        "numeric_imputation":args.imputation,
        "numeric_scaling":args.scaling,
        "polynomial_degree":args.polynomial_degree,
        "datetime_field":args.datetime_field,
        "text_features":["word unigram/bigram TF-IDF","character/word/line counts",
                         "technical pattern presence flags"],
        "fit_policy":"All vectorizers, encoders, imputers, polynomial terms and scalers are fitted using train only; validation/test are transformed with that fitted state.",
        "splits":split_summary,
        "privacy_note":"Use cleaned, privacy-reviewed input. Heuristic PII screening is incomplete. TF-IDF vocabulary and the saved preprocessor can retain distinctive ticket terms; generated files are gitignored and should not be published without review."
    }
    (args.output/"feature_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    )
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path(__file__).resolve().parent/"cleaned")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"features")
    parser.add_argument("--label-key",required=True,help="Target key in labels, e.g. category or priority_source.")
    parser.add_argument("--categorical-fields",default="source,task",
                        help="Comma-separated record fields used as nominal predictors; labels.* is forbidden.")
    parser.add_argument("--categorical-encoding",choices=("onehot","target"),default="onehot")
    parser.add_argument("--imputation",choices=("mean","median","most_frequent","knn"),default="median")
    parser.add_argument("--scaling",choices=("standard","minmax","robust","none"),default="robust")
    parser.add_argument("--polynomial-degree",type=int,choices=(1,2,3),default=1)
    parser.add_argument("--datetime-field",default=None,
                        help="Optional dotted record path, e.g. ticket.created_at; not present in today's canonical schema.")
    parser.add_argument("--min-df",type=int,default=1)
    parser.add_argument("--max-features",type=int,default=20000)
    args=parser.parse_args()
    args.categorical_fields=[field.strip() for field in args.categorical_fields.split(",") if field.strip()]
    result=run(args)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__=="__main__":
    main()
