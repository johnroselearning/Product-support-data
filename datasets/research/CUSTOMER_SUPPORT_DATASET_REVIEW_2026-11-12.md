# Customer-support dataset source review

Review date: 2026-11-12
Repository: Product-support-data

## Scope and evidence level

This is a public-page and dataset-preview inspection, not a full local-file audit. Kaggle's interactive preview exposed the data card, declared license, file name/size, row/column summaries, and sample rows. The full Kaggle CSV archives were not downloaded into this repository in this review. Therefore, row-level distributions, exact byte hashes, full PII scans, and complete semantic quality measurements remain pending.

No raw data was committed. Candidate sources remain disabled from ingestion.

## 1. Suraj520 — Customer Support Ticket Dataset

Source: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

Observed on Kaggle:
- Declared license: CC0 1.0.
- File: `customer_support_tickets.csv`.
- Preview reports 8,469 rows, 17 columns, approximately 3.95 MB.
- Fields include Ticket ID, Customer Name, Customer Email, Customer Age, Customer Gender, Product Purchased, Date of Purchase, Ticket Type, Ticket Subject, Ticket Description, Ticket Status, Resolution, Ticket Priority, Ticket Channel, First Response Time, Time to Resolution, and Customer Satisfaction Rating.
- The data card says the email domain `example.com` is intentional for privacy.
- Previewed descriptions include the literal `{product_purchased}` placeholder and repeated boilerplate such as "I'm having an issue with the {product_purchased}. Please assist." Other samples append unrelated free text, including phone-number-like strings and miscellaneous notes.

Assessment:
- Potential use: ticket metadata analysis, queue/category/priority experiments, response-time and satisfaction analytics, and possibly resolution-text analysis after checking target quality.
- Not suitable to assume that description-only classification or technical troubleshooting is reliable. The displayed examples show generic templates and possible label/text mismatch.
- Names, emails, phone-like values and other customer fields require privacy screening even though the publisher says example.com is intentional. CC0 does not waive independent privacy/personality rights.
- Decision: register as `review_required`; do not enable ingestion until full CSV inspection, privacy review, field-level missingness, label consistency, template rate, and source provenance are measured.

## 2. Waseem Alastal — Customer Support Ticket Dataset

Source: https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset

Observed on Kaggle:
- File: `customer_support_tickets.csv`, approximately 3.95 MB.
- Preview reports 8,469 rows and 17 columns.
- Previewed initial records and field layout match the Suraj520 dataset preview (same first displayed customer/ticket examples and field names).
- Kaggle page displays Database Contents License (DbCL) 1.0.

Assessment:
- Strong evidence that this is a re-upload or mirror of the Suraj520 dataset, but no full-file checksum or complete row-by-row comparison has been performed. Do not call it byte-identical yet.
- DbCL 1.0 requires compliance with ODbL and is concerned with rights in database contents; the dataset/database rights and any separate rights in personal/customer content must be checked. The Kaggle page preview did not expose a clear companion ODbL notice. See Open Data Commons DbCL and FAQ references in the repository report's source links.
- Decision: register as `duplicate_candidate`, not as an independent ingestion source. Compare downloaded file hashes and row-level fingerprints before deciding whether it is the same distribution. Prefer the original Suraj source if identity is confirmed and its CC0 declaration is applicable.

## 3. Tobias Bueck — Kaggle multilingual distribution

Source: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

Observed on Kaggle:
- Kaggle data card displays CC BY 4.0.
- Dataset author describes the data as synthetically generated.
- The card describes English, German, French, Spanish and Portuguese and fields for queue, priority, language, subject, body, agent answer, ticket type, business type and tags.
- The current Kaggle card's actual file list, exact row count, file checksums, and observed language distribution were not captured in this review.

Assessment:
- Potential use: multilingual ticket routing, priority/type classification, tag prediction, and customer-response style experiments.
- It is synthetic and does not establish real-world technical troubleshooting correctness, root cause, tool execution, or verified fixes.
- Kaggle's CC BY 4.0 declaration differs from the Hugging Face distribution registered as `tobi`, whose Hugging Face dataset card declares CC BY-NC 4.0 and shows 61,765 rows, English/German, and three CSV files.
- Do not assume one distribution's license automatically changes the other distribution's license. Capture source-specific version, file hashes, license text and attribution. Confirm that the Kaggle uploader controls the rights to the synthetic content and that the exact downloaded files are covered by the displayed CC BY 4.0 license.
- Decision: register the Kaggle distribution separately as `review_required`; keep the Hugging Face distribution research/non-commercial only unless separately licensed.

## Licensing and PSI project context

The PSI Community repository uses the MIT License. That permits commercial use of PSI software under its terms; it does not grant rights to third-party datasets. Dataset permissions must be assessed independently.

A non-commercial dataset can still be useful in a segregated research/evaluation workspace where the actual activity is permitted. It must not silently flow into a commercial training corpus or a distributed artifact. The exact meaning of commercial use and whether a particular model-training use is permitted can be fact-specific; obtain legal review for production use.

## Next inspection required before ingestion

Download the exact Kaggle files locally (do not commit raw files) and produce:
1. SHA-256 file hash, source revision/version, file names and file sizes.
2. Row/column count, dtypes, null rates, unique counts and duplicate IDs.
3. Exact and normalized duplicate fingerprints within and across datasets.
4. PII scan for names, emails, phone numbers, addresses, account/order identifiers and secrets.
5. Placeholder/template prevalence (e.g. `{product_purchased}`), boilerplate frequency and text-length distribution.
6. Cross-tabs and mutual information/association checks between text, subject, type, priority, resolution and satisfaction labels.
7. Manual sample review of at least 100 stratified rows, including low-frequency labels and empty/templated descriptions.
8. License snapshots and attribution metadata for the exact downloaded version.

Keep raw data outside Git. Store only scripts, metadata, reports and permitted derived outputs in this repository.

## Primary references

- Suraj520 Kaggle dataset: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
- Waseem Alastal Kaggle dataset: https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset
- Tobias Bueck Kaggle dataset: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
- Tobias Bueck Hugging Face distribution: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
- Open Data Commons DbCL 1.0: https://opendatacommons.org/licenses/dbcl/1-0/
- Open Data Commons license FAQ: https://opendatacommons.org/faq/licenses/
- Creative Commons CC0 1.0 deed: https://creativecommons.org/publicdomain/zero/1.0/deed.en
