# Statistical EDA and optional feature engineering

The canonical master and SFT records remain the source of truth. EDA and classical feature generation write separate outputs and never rewrite those records.

## Is each technique necessary?

For SFT or transformer fine-tuning, one-hot encoding, scaling, polynomial expansion, and TF-IDF are usually not required: the model's tokenizer and training code handle text. For a classical ticket-classification baseline, TF-IDF, text measurements, selected categorical encodings, and numeric scaling are useful comparison features. This repository supports both paths without mixing them.

## Statistical EDA

Run:

    python datasets/eda.py

The JSON summary reports mean, median, mode, sample variance, sample standard deviation, IQR, quartiles, p95, skewness, and excess kurtosis for character and word counts, overall and by split. It also reports Pearson, Spearman, and Kendall correlations between those two numeric measurements.

The existing label/source/split counts and text-length plots remain. These statistics describe the data; they do not transform training records. The correlations are only between numeric text measurements. Category labels are not assigned arbitrary numbers for correlation analysis.

## Classical feature matrix

Install dependencies and run against cleaned data:

    pip install -r datasets/requirements.txt
    python datasets/feature_engineering.py --label-key category
    python datasets/feature_engineering.py --label-key priority_source

Outputs go under datasets/features by default: sparse X matrices, matching y and case-ID lists, a feature manifest, and a saved preprocessor. The generated directory is gitignored. Use cleaned inputs because PII screening is heuristic, and review outputs before sharing; a TF-IDF vocabulary or serialized preprocessor can retain distinctive terms.

The feature exporter fits the vectorizer, categorical encoder, imputer, polynomial expansion, and scaler on train only. Validation and test are transformed with that fitted state. It excludes rows with missing targets and rows carrying privacy flags; it never imputes a target label.

Supported options:

- **Categorical inputs:** One-hot encoding by default, or train-fold target encoding. Only configured record fields are encoded; target fields under labels are rejected. Ordinal encoding is intentionally not the default because current predictor categories such as source and task are nominal and have no meaningful order.
- **Numeric inputs:** Character, word, and line counts; punctuation counts; and boolean technical-text regex signals such as 4xx/5xx status presence, stack-trace/error wording, log level, HTTP method, and URL presence.
- **Scaling and imputation:** Standard, min-max, or robust scaling; mean, median, most-frequent, or KNN imputation; and an explicit missingness indicator for each numeric input. Current generated text measures are generally present, so imputation is available for optional future fields rather than an assumption that current data has gaps.
- **Datetime:** If an adapter later provides a timestamp, pass its dotted record path with --datetime-field. The exporter creates hour, weekday, month, weekend, and sine/cosine cycle features. The current canonical schema has no timestamp, so this is opt-in and errors if the requested training field has no parseable values.
- **Interactions:** Polynomial features of degree 2 or 3 can be enabled for the small numeric feature set. Degree 1 is the default to avoid adding unsupported complexity.
- **Text:** Word unigram/bigram TF-IDF and compact text measurements are implemented. Embeddings are not generated because the repository has no selected embedding model or runtime.

No preprocessing choice is universally best. Use training-only fit, grouped splits already produced by the data pipeline, and validation results to choose a baseline. Keep the publisher test split for final evaluation.
