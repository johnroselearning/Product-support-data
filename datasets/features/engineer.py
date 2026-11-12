"""Fit a reusable, leakage-aware numerical feature pipeline on master JSONL data.

The label is carried as a target and is never included in the input matrix.
All learned transforms (TF-IDF vocabulary, imputers, encoders and scalers) are
fit on the training split only. Target encoding uses leave-one-out values for
training rows and a smoothed mapping learned from all training rows elsewhere.
"""
import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from quality.privacy import detect_flags

SPLITS = ("train", "validation", "test")
BASE_NUMERIC_FEATURES = (
    "text_chars", "text_words", "text_lines", "mean_token_chars",
    "characters_per_word", "digits_per_word", "lines_per_word",
    "digit_char_ratio", "uppercase_char_ratio", "punctuation_char_ratio",
    "question_mark_count", "http_error_code_count", "exception_term_count",
    "stack_trace_line_count", "version_pattern_count", "database_term_count",
    "network_term_count", "authentication_term_count",
)
DATE_COMPONENTS = (
    "year", "month", "day", "hour", "weekday", "is_weekend",
    "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "hour_sin", "hour_cos",
)
DATE_NAME_RE = re.compile(r"(?:date|time|created|updated|opened|closed|resolved)", re.I)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?$")
SENSITIVE_PARTS = {
    "id", "key", "name", "email", "phone", "address", "user", "customer",
    "reporter", "account", "ip", "token", "password", "secret", "label",
    "target", "outcome", "resolution", "answer", "cause", "severity",
    "priority", "category", "class", "description", "text", "message", "body",
}

PATTERNS = {
    "http_error_code_count": re.compile(r"\b(?:HTTP\s*)?[45]\d{2}\b", re.I),
    "exception_term_count": re.compile(r"\b(?:exception|traceback|caused by|segmentation fault)\b", re.I),
    "stack_trace_line_count": re.compile(r"(?m)^\s*(?:at\s+\S+|File \"[^\"]+\", line \d+|Traceback \(most recent call last\))"),
    "version_pattern_count": re.compile(r"\bv?\d+\.\d+(?:\.\d+){0,2}\b", re.I),
    "database_term_count": re.compile(r"\b(?:sql|database|mysql|postgres(?:ql)?|mongodb|redis|query)\b", re.I),
    "network_term_count": re.compile(r"\b(?:dns|tcp|udp|http|https|tls|timeout|connection refused|proxy)\b", re.I),
    "authentication_term_count": re.compile(r"\b(?:login|authentication|authorization|oauth|sso|mfa|permission|forbidden)\b", re.I),
}


def _attribute_parts(key):
    return set(re.findall(r"[a-z0-9]+", str(key).casefold()))


def _is_safe_attribute(key):
    parts = _attribute_parts(key)
    return bool(parts) and not parts.intersection(SENSITIVE_PARTS)


def _attribute_dict(record):
    attributes = record.get("attributes", {})
    if not isinstance(attributes, dict):
        return {}
    # Defend the transform boundary too, because callers may supply older or
    # hand-authored master records that did not pass through the source adapter.
    safe = {}
    for key, value in attributes.items():
        if not _is_safe_attribute(key):
            continue
        if value is None or isinstance(value, (bool, int, float, str, datetime)):
            if not isinstance(value, str) or len(value) <= 256:
                safe[str(key)] = value
    return safe


def _is_missing_category(value):
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _text(record):
    ticket = record.get("ticket", {})
    value = ticket.get("description", "") if isinstance(ticket, dict) else ""
    if not isinstance(value, str):
        return ""
    return value


def _target(record):
    labels = record.get("labels", {})
    if not isinstance(labels, dict) or not labels:
        return None
    normalized = {str(k): str(v) for k, v in labels.items() if v is not None}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False) if normalized else None


def _to_number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else np.nan
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not NUMERIC_RE.fullmatch(text):
            return np.nan
        percent = text.endswith("%")
        try:
            number = float(text[:-1] if percent else text)
        except ValueError:
            return np.nan
        return number / 100.0 if percent else number
    return np.nan


def _parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _datetime_features(value):
    moment = _parse_datetime(value)
    if moment is None:
        return {name: np.nan for name in DATE_COMPONENTS}
    month_angle = 2.0 * math.pi * (moment.month - 1) / 12.0
    weekday_angle = 2.0 * math.pi * moment.weekday() / 7.0
    hour_angle = 2.0 * math.pi * moment.hour / 24.0
    return {
        "year": moment.year,
        "month": moment.month,
        "day": moment.day,
        "hour": moment.hour,
        "weekday": moment.weekday(),
        "is_weekend": int(moment.weekday() >= 5),
        "month_sin": math.sin(month_angle), "month_cos": math.cos(month_angle),
        "weekday_sin": math.sin(weekday_angle), "weekday_cos": math.cos(weekday_angle),
        "hour_sin": math.sin(hour_angle), "hour_cos": math.cos(hour_angle),
    }


def _text_features(text):
    chars = len(text)
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    letters = [ch for ch in text if ch.isalpha()]
    digits = sum(ch.isdigit() for ch in text)
    punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    result = {
        "text_chars": chars,
        "text_words": len(tokens),
        "text_lines": len(text.splitlines()) or (1 if text else 0),
        "mean_token_chars": (sum(map(len, tokens)) / len(tokens)) if tokens else 0.0,
        "characters_per_word": chars / len(tokens) if tokens else 0.0,
        "digits_per_word": digits / len(tokens) if tokens else 0.0,
        "lines_per_word": (len(text.splitlines()) or (1 if text else 0)) / len(tokens) if tokens else 0.0,
        "digit_char_ratio": digits / chars if chars else 0.0,
        "uppercase_char_ratio": sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0,
        "punctuation_char_ratio": punctuation / chars if chars else 0.0,
        "question_mark_count": text.count("?"),
    }
    result.update({name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()})
    return result


def _infer_attribute_types(records):
    values_by_key = {}
    for record in records:
        for key, value in _attribute_dict(record).items():
            values_by_key.setdefault(key, []).append(value)
    numeric, categorical, dates = [], [], []
    for key in sorted(values_by_key, key=str.casefold):
        observed = [value for value in values_by_key[key] if value is not None and str(value).strip()]
        if not observed:
            continue
        if DATE_NAME_RE.search(key) and any(_parse_datetime(value) for value in observed):
            dates.append(key)
            continue
        numbers = [_to_number(value) for value in observed]
        parse_rate = sum(np.isfinite(number) for number in numbers) / len(observed)
        if parse_rate >= 0.9:
            numeric.append(key)
        else:
            categorical.append(key)
    return numeric, categorical, dates


class SupportFeatureEngineer:
    """Reusable preprocessing and vectorization for canonical support records."""

    def __init__(self, categorical_encoding="onehot", scaling="standard",
                 polynomial_degree=2, tfidf_max_features=50000,
                 embedding_model=None, target_smoothing=10.0,
                 numeric_imputation="median", categorical_imputation="constant"):
        if categorical_encoding not in {"onehot", "ordinal", "target"}:
            raise ValueError("categorical_encoding must be onehot, ordinal, or target")
        if scaling not in {"standard", "minmax", "robust", "none"}:
            raise ValueError("scaling must be standard, minmax, robust, or none")
        if polynomial_degree not in {0, 1, 2, 3}:
            raise ValueError("polynomial_degree must be between 0 and 3")
        if tfidf_max_features < 1:
            raise ValueError("tfidf_max_features must be positive")
        if target_smoothing <= 0:
            raise ValueError("target_smoothing must be positive")
        if numeric_imputation not in {"mean", "median", "most_frequent", "knn"}:
            raise ValueError("numeric_imputation must be mean, median, most_frequent, or knn")
        if categorical_imputation not in {"constant", "most_frequent"}:
            raise ValueError("categorical_imputation must be constant or most_frequent")
        self.categorical_encoding = categorical_encoding
        self.scaling = scaling
        self.polynomial_degree = polynomial_degree
        self.tfidf_max_features = tfidf_max_features
        self.embedding_model = embedding_model
        self.target_smoothing = target_smoothing
        self.numeric_imputation = numeric_imputation
        self.categorical_imputation = categorical_imputation
        self._sentence_encoder = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sentence_encoder"] = None
        return state

    def _ensure_embedding_model(self):
        if not self.embedding_model:
            return None
        if self._sentence_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence embeddings require the optional dependency. Install with: "
                    "pip install sentence-transformers"
                ) from exc
            self._sentence_encoder = SentenceTransformer(self.embedding_model)
        return self._sentence_encoder

    def _embed(self, texts):
        if not self.embedding_model:
            return None
        if not texts:
            dimension = getattr(self, "embedding_dimension_", 0)
            return np.empty((0, dimension), dtype=np.float32)
        matrix = self._ensure_embedding_model().encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        matrix = np.asarray(matrix, dtype=np.float32)
        if getattr(self, "embedding_dimension_", None) is None:
            self.embedding_dimension_ = int(matrix.shape[1])
        elif matrix.shape[1] != self.embedding_dimension_:
            raise ValueError("Embedding model returned a different vector dimension")
        return matrix

    def _row_values(self, record):
        text = _text(record)
        numeric = _text_features(text)
        attributes = _attribute_dict(record)
        for key in self.numeric_attributes_:
            numeric[f"attribute__{key}"] = _to_number(attributes.get(key))
        for key in self.datetime_attributes_:
            values = _datetime_features(attributes.get(key))
            numeric.update({f"attribute__{key}__{name}": value for name, value in values.items()})
        categorical = {
            "source": str(record.get("source", "__missing__")),
            "task": str(record.get("task", "__missing__")),
        }
        for key in self.categorical_attributes_:
            value = attributes.get(key)
            categorical[f"attribute__{key}"] = str(value).strip() if value is not None and str(value).strip() else np.nan
        return text, numeric, categorical

    def _matrix_values(self, records):
        texts, numeric_rows, categorical_rows = [], [], []
        for record in records:
            text, numeric, categorical = self._row_values(record)
            texts.append(text)
            numeric_rows.append([numeric.get(key, np.nan) for key in self.numeric_input_names_])
            categorical_rows.append([categorical.get(key, "__missing__") for key in self.categorical_names_])
        numeric = np.asarray(numeric_rows, dtype=np.float64).reshape(len(records), len(self.numeric_input_names_))
        categorical = np.asarray(categorical_rows, dtype=object).reshape(len(records), len(self.categorical_names_))
        return texts, numeric, categorical

    def _target_strings(self, records):
        return [_target(record) for record in records]

    def _fit_target_encoders(self, categorical, target_ids):
        self.target_encoders_ = []
        n_classes = len(self.target_classes_)
        y = np.zeros((len(target_ids), n_classes), dtype=np.float64)
        for row, target_id in enumerate(target_ids):
            y[row, target_id] = 1.0
        self.target_prior_ = y.mean(axis=0) if len(y) else np.full(n_classes, 1 / max(n_classes, 1))
        for column in range(categorical.shape[1]):
            sums, counts = {}, {}
            for row, value in enumerate(categorical[:, column]):
                sums.setdefault(value, np.zeros(n_classes, dtype=np.float64))
                sums[value] += y[row]
                counts[value] = counts.get(value, 0) + 1
            self.target_encoders_.append({"sums": sums, "counts": counts})

    def _target_encoded_matrix(self, categorical, target_ids=None, leave_one_out=False):
        n_rows = categorical.shape[0]
        n_classes = len(self.target_classes_)
        if categorical.shape[1] == 0 or n_classes == 0:
            return np.empty((n_rows, 0), dtype=np.float64)
        y = None
        if leave_one_out:
            y = np.zeros((len(target_ids), n_classes), dtype=np.float64)
            for row, target_id in enumerate(target_ids):
                y[row, target_id] = 1.0
            total_targets = y.sum(axis=0)
        result = np.zeros((n_rows, categorical.shape[1] * n_classes), dtype=np.float64)
        for col in range(categorical.shape[1]):
            encoder = self.target_encoders_[col]
            for row, value in enumerate(categorical[:, col]):
                prior = self.target_prior_
                sums = encoder["sums"].get(value, np.zeros(n_classes, dtype=np.float64))
                count = encoder["counts"].get(value, 0)
                if leave_one_out:
                    own = y[row]
                    prior = ((total_targets - own) / (n_rows - 1)) if n_rows > 1 else np.full(n_classes, 1 / n_classes)
                    sums = sums - own
                    count -= 1
                result[row, col * n_classes:(col + 1) * n_classes] = (
                    sums + self.target_smoothing * prior
                ) / (count + self.target_smoothing)
        return result

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("The training split is empty; cannot fit feature transforms")
        self.numeric_attributes_, self.categorical_attributes_, self.datetime_attributes_ = _infer_attribute_types(records)
        self.numeric_input_names_ = list(BASE_NUMERIC_FEATURES)
        for key in self.numeric_attributes_:
            self.numeric_input_names_.append(f"attribute__{key}")
        for key in self.datetime_attributes_:
            self.numeric_input_names_.extend(f"attribute__{key}__{name}" for name in DATE_COMPONENTS)
        self.categorical_names_ = ["source", "task"] + [f"attribute__{key}" for key in self.categorical_attributes_]

        targets = self._target_strings(records)
        known_targets = sorted({target for target in targets if target is not None})
        self.target_classes_ = known_targets
        self.target_to_id_ = {target: index for index, target in enumerate(known_targets)}
        self.target_ids_ = [self.target_to_id_.get(target, -1) for target in targets]
        if any(target_id < 0 for target_id in self.target_ids_):
            raise ValueError("Every training record must have a non-empty label target")

        texts, numeric, categorical = self._matrix_values(records)
        base_count = len(self.numeric_input_names_)
        if self.numeric_imputation == "knn":
            # Put heterogeneous numeric columns on a common scale for distance
            # calculations, then restore their original ranges before expansion.
            self.knn_distance_scaler_ = MinMaxScaler()
            scaled_numeric = self.knn_distance_scaler_.fit_transform(numeric)
            self.numeric_imputer_ = KNNImputer(
                n_neighbors=min(5, len(records)), add_indicator=True,
                keep_empty_features=True,
            )
            knn_imputed = self.numeric_imputer_.fit_transform(scaled_numeric)
            restored = self.knn_distance_scaler_.inverse_transform(knn_imputed[:, :base_count])
            imputed = np.hstack([restored, knn_imputed[:, base_count:]])
        else:
            self.knn_distance_scaler_ = None
            self.numeric_imputer_ = SimpleImputer(
                strategy=self.numeric_imputation, add_indicator=True,
                keep_empty_features=True,
            )
            imputed = self.numeric_imputer_.fit_transform(numeric)
        base_numeric = imputed[:, :base_count]
        indicators = imputed[:, base_count:]
        if self.polynomial_degree >= 2:
            self.polynomial_ = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            polynomial = self.polynomial_.fit_transform(base_numeric)
            poly_names = list(self.polynomial_.get_feature_names_out(self.numeric_input_names_))
        else:
            self.polynomial_ = None
            polynomial = base_numeric
            poly_names = list(self.numeric_input_names_)
        indicator_names = [
            f"missing__{self.numeric_input_names_[index]}"
            for index in getattr(self.numeric_imputer_.indicator_, "features_", [])
        ]
        self.numeric_feature_names_ = poly_names + indicator_names
        numeric_with_indicators = np.hstack([polynomial, indicators]) if indicators.shape[1] else polynomial
        self.scaler_ = {
            "standard": StandardScaler(), "minmax": MinMaxScaler(),
            "robust": RobustScaler(), "none": None,
        }[self.scaling]
        if self.scaler_:
            self.scaler_.fit(numeric_with_indicators)

        self.categorical_missing_indices_ = list(range(categorical.shape[1])) if self.categorical_imputation == "most_frequent" else []
        self.categorical_imputer_ = SimpleImputer(
            strategy="constant" if self.categorical_imputation == "constant" else "most_frequent",
            fill_value="__missing__" if self.categorical_imputation == "constant" else None,
            keep_empty_features=True,
        )
        categorical = self.categorical_imputer_.fit_transform(categorical)

        if self.categorical_encoding == "onehot":
            self.categorical_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
            self.categorical_encoder_.fit(categorical)
            encoded_names = list(self.categorical_encoder_.get_feature_names_out(self.categorical_names_))
        elif self.categorical_encoding == "ordinal":
            # Source/task are nominal metadata, so keep them one-hot even when
            # the caller opts into ordinal encoding for structured attributes.
            self.categorical_encoder_ = None
            self.meta_categorical_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
            self.meta_categorical_encoder_.fit(categorical[:, :2])
            self.attribute_categorical_encoder_ = None
            if self.categorical_attributes_:
                self.attribute_categorical_encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32,
                )
                self.attribute_categorical_encoder_.fit(categorical[:, 2:])
            encoded_names = list(self.meta_categorical_encoder_.get_feature_names_out(self.categorical_names_[:2]))
            encoded_names.extend(self.categorical_names_[2:])
        else:
            self.categorical_encoder_ = None
            self._fit_target_encoders(categorical, self.target_ids_)
            encoded_names = [
                f"{name}__target_class_{class_id}"
                for name in self.categorical_names_
                for class_id in range(len(self.target_classes_))
            ]
        self.categorical_missing_names_ = [
            f"missing__{self.categorical_names_[index]}" for index in self.categorical_missing_indices_
        ]
        self.categorical_feature_names_ = encoded_names + self.categorical_missing_names_

        self.word_vectorizer_ = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=self.tfidf_max_features,
            token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, dtype=np.float32,
        )
        self.char_vectorizer_ = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 5), max_features=self.tfidf_max_features,
            sublinear_tf=True, dtype=np.float32,
        )
        # Fit each vocabulary only on train, even when documents are very short.
        try:
            self.word_vectorizer_.fit(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).casefold():
                raise
            self.word_vectorizer_ = None
        try:
            self.char_vectorizer_.fit(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).casefold():
                raise
            self.char_vectorizer_ = None
        self.embedding_dimension_ = None
        if self.embedding_model:
            embedding = self._embed(texts[:1])
            self.embedding_dimension_ = int(embedding.shape[1])

        self.feature_names_ = (
            self.numeric_feature_names_
            + self.categorical_feature_names_
            + [f"tfidf_word_{index:05d}" for index in range(len(self.word_vectorizer_.vocabulary_) if self.word_vectorizer_ else 0)]
            + [f"tfidf_char_{index:05d}" for index in range(len(self.char_vectorizer_.vocabulary_) if self.char_vectorizer_ else 0)]
            + ([f"embedding_{index:04d}" for index in range(self.embedding_dimension_)] if self.embedding_model else [])
        )
        self.is_fitted_ = True
        return self

    def transform_records(self, records, training_rows=False):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit(training_records) before transform_records")
        records = list(records)
        if not records:
            return sparse.csr_matrix((0, len(self.feature_names_)), dtype=np.float32)
        texts, numeric, categorical = self._matrix_values(records)
        base_count = len(self.numeric_input_names_)
        if self.numeric_imputation == "knn":
            scaled_numeric = self.knn_distance_scaler_.transform(numeric)
            knn_imputed = self.numeric_imputer_.transform(scaled_numeric)
            restored = self.knn_distance_scaler_.inverse_transform(knn_imputed[:, :base_count])
            imputed = np.hstack([restored, knn_imputed[:, base_count:]])
        else:
            imputed = self.numeric_imputer_.transform(numeric)
        base_numeric, indicators = imputed[:, :base_count], imputed[:, base_count:]
        polynomial = self.polynomial_.transform(base_numeric) if self.polynomial_ else base_numeric
        numeric_values = np.hstack([polynomial, indicators]) if indicators.shape[1] else polynomial
        if self.scaler_:
            numeric_values = self.scaler_.transform(numeric_values)
        numeric_matrix = sparse.csr_matrix(numeric_values.astype(np.float32))

        missing_mask = np.asarray([
            [_is_missing_category(value) for value in row]
            for row in categorical
        ], dtype=bool)
        categorical = self.categorical_imputer_.transform(categorical)
        if self.categorical_encoding == "onehot":
            categorical_matrix = self.categorical_encoder_.transform(categorical).tocsr()
        elif self.categorical_encoding == "ordinal":
            meta_matrix = self.meta_categorical_encoder_.transform(categorical[:, :2]).tocsr()
            if self.attribute_categorical_encoder_ is not None:
                attribute_matrix = sparse.csr_matrix(
                    self.attribute_categorical_encoder_.transform(categorical[:, 2:]).astype(np.float32)
                )
                categorical_matrix = sparse.hstack([meta_matrix, attribute_matrix], format="csr")
            else:
                categorical_matrix = meta_matrix
        else:
            target_ids = [self.target_to_id_.get(value, -1) for value in self._target_strings(records)]
            if training_rows:
                if any(value < 0 for value in target_ids):
                    raise ValueError("Training rows for target encoding must have labels from the fitted training set")
                categorical_values = self._target_encoded_matrix(categorical, target_ids, leave_one_out=True)
            else:
                categorical_values = self._target_encoded_matrix(categorical)
            categorical_matrix = sparse.csr_matrix(categorical_values.astype(np.float32))

        word_matrix = self.word_vectorizer_.transform(texts).tocsr() if self.word_vectorizer_ else sparse.csr_matrix((len(texts), 0), dtype=np.float32)
        char_matrix = self.char_vectorizer_.transform(texts).tocsr() if self.char_vectorizer_ else sparse.csr_matrix((len(texts), 0), dtype=np.float32)
        if self.categorical_missing_indices_:
            missing_matrix = sparse.csr_matrix(missing_mask[:, self.categorical_missing_indices_].astype(np.float32))
            categorical_matrix = sparse.hstack([categorical_matrix, missing_matrix], format="csr")
        blocks = [numeric_matrix, categorical_matrix, word_matrix, char_matrix]
        if self.embedding_model:
            blocks.append(sparse.csr_matrix(self._embed(texts)))
        matrix = sparse.hstack(blocks, format="csr", dtype=np.float32)
        if matrix.shape[1] != len(self.feature_names_):
            raise RuntimeError(f"Feature schema mismatch: matrix has {matrix.shape[1]} columns, schema has {len(self.feature_names_)}")
        return matrix

    def target_metadata(self, records):
        metadata = []
        for record in records:
            target = _target(record)
            metadata.append({
                "case_id": record.get("case_id"),
                "target": json.loads(target) if target else None,
                "target_id": self.target_to_id_.get(target),
            })
        return metadata

    def schema(self):
        return {
            "format_version": "1.0",
            "target_is_excluded_from_features": True,
            "fit_split": "train",
            "feature_count": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "feature_blocks": {
                "numeric_and_polynomial": len(self.numeric_feature_names_),
                "categorical": len(self.categorical_feature_names_),
                "word_tfidf": len(self.word_vectorizer_.vocabulary_) if self.word_vectorizer_ else 0,
                "character_tfidf": len(self.char_vectorizer_.vocabulary_) if self.char_vectorizer_ else 0,
                "sentence_embedding": self.embedding_dimension_ or 0,
            },
            "numeric_attributes": self.numeric_attributes_,
            "categorical_attributes": self.categorical_attributes_,
            "datetime_attributes": self.datetime_attributes_,
            "target_classes": [{"target_id": i, "labels": json.loads(target)} for i, target in enumerate(self.target_classes_)],
            "configuration": {
                "categorical_encoding": self.categorical_encoding,
                "scaling": self.scaling,
                "polynomial_degree": self.polynomial_degree,
                "tfidf_max_features_per_analyzer": self.tfidf_max_features,
                "embedding_model": self.embedding_model,
                "target_smoothing": self.target_smoothing,
                "numeric_imputation": self.numeric_imputation,
                "categorical_imputation": self.categorical_imputation,
            },
            "notes": [
                "Imputers, scalers, categorical vocabularies and TF-IDF vocabularies are fit on train only.",
                "Training target-encoded values use leave-one-out statistics; validation/test use train mappings.",
                "Unknown validation/test classes have a null target_id and are retained in metadata.",
                "Date features are emitted only when an allowlisted datetime attribute exists in train.",
                "The current normalized sources provide mostly text and labels; structured features depend on adapter inputs.",
            ],
        }


def read_records(input_dir):
    splits = {}
    for split in SPLITS:
        path = input_dir / "master" / f"{split}.jsonl"
        records = []
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    attributes = _attribute_dict(record)
                    safe_attribute_text = " ".join(
                        str(value) for key, value in attributes.items()
                        if isinstance(value, str)
                        and not DATE_NAME_RE.search(key)
                        and not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?", value)
                    )
                    # privacy_text = "\n".join([
                    #     _text(record),
                    #     " ".join(str(value) for value in record.get("labels", {}).values()),
                    #     safe_attribute_text,
                    # ])
                    privacy_fields = [
                        _text(record),
                        *(str(value) for value in record.get("labels", {}).values()),
                        safe_attribute_text,
                    ]
                    if any(detect_flags(value) for value in privacy_fields):
                        raise ValueError(f"Refusing record with a privacy pattern at {path}:{line_number}")   
                    # if detect_flags(privacy_text):
                    #     raise ValueError(f"Refusing record with a privacy pattern at {path}:{line_number}")
                    records.append(record)
        splits[split] = records
    return splits


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_features(input_dir, output_dir, **options):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    splits = read_records(input_dir)
    engineer = SupportFeatureEngineer(**options).fit(splits["train"])
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, records in splits.items():
        matrix = engineer.transform_records(records, training_rows=(split == "train" and engineer.categorical_encoding == "target"))
        sparse.save_npz(output_dir / f"{split}.npz", matrix, compressed=True)
        _write_jsonl(output_dir / f"{split}_metadata.jsonl", engineer.target_metadata(records))
        counts[split] = len(records)
    (output_dir / "feature_schema.json").write_text(json.dumps(engineer.schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    joblib.dump(engineer, output_dir / "feature_pipeline.joblib")
    report = {
        "input": str(input_dir), "output": str(output_dir), "records_by_split": counts,
        "feature_count": len(engineer.feature_names_),
        "target_classes": len(engineer.target_classes_),
        "feature_pipeline": str(output_dir / "feature_pipeline.joblib"),
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Build train-fitted numerical features from normalized support master records.")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "generated")
    parser.add_argument("--output", type=Path, help="Defaults to <input>/features")
    parser.add_argument("--categorical-encoding", choices=("onehot", "ordinal", "target"), default="onehot")
    parser.add_argument("--scaling", choices=("standard", "minmax", "robust", "none"), default="standard")
    parser.add_argument("--polynomial-degree", type=int, choices=(0, 1, 2, 3), default=2, help="0 or 1 disables interactions and squares")
    parser.add_argument("--tfidf-max-features", type=int, default=50000, help="Vocabulary limit for each word/character analyzer")
    parser.add_argument("--embedding-model", help="Optional sentence-transformers model name/path; model is loaded only when requested")
    parser.add_argument("--numeric-imputation", choices=("mean", "median", "most_frequent", "knn"), default="median")
    parser.add_argument("--categorical-imputation", choices=("constant", "most_frequent"), default="constant")
    parser.add_argument("--target-smoothing", type=float, default=10.0)
    args = parser.parse_args()
    output = args.output or args.input / "features"
    result = generate_features(
        args.input, output,
        categorical_encoding=args.categorical_encoding,
        scaling=args.scaling,
        polynomial_degree=args.polynomial_degree,
        tfidf_max_features=args.tfidf_max_features,
        embedding_model=args.embedding_model,
        numeric_imputation=args.numeric_imputation,
        categorical_imputation=args.categorical_imputation,
        target_smoothing=args.target_smoothing,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
