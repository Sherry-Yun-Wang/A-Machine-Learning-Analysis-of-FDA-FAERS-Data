# GLP-1 RA Pharmacovigilance: Calibrated Risk Prediction from FAERS

This repository contains the complete analytical pipeline for the manuscript
**"From Reporting Frequency to Risk Intelligence: A Calibrated AI Approach to
Pharmacovigilance Using FAERS"** (submitted to *npj Digital Medicine*).

## Canonical notebook

**`analysis.ipynb`** is the single canonical notebook for this project. It
contains the entire pipeline end to end — cohort construction, all five
predictive models, calibration, SHAP explainability, ablation experiments,
and supplementary analyses — and is the only notebook that should be used to
reproduce the results reported in the manuscript.

A previous version of this repository also included a second notebook,
`analysis_LR_RF___1_.ipynb`, containing only the logistic regression and
random forest pipeline. That notebook was a subset of `analysis.ipynb` with
identical numerical results and has been **removed** to avoid ambiguity
about which notebook is authoritative, per reviewer feedback.

## Execution order

The notebook is organized into the following sections, intended to be run
top to bottom in a single session:

1. **Cohort construction** (cells 1–40) — Loads FAERS data from the attached
   SQLite database via DuckDB, builds the GLP-1 receptor agonist drug
   dictionary, applies role-code and brand/generic windowing logic to
   identify the initial 250,093-case cohort, and computes the case-level
   serious-outcome flag.

2. **Drug normalization and exposure classification** (cells 41–94) —
   Builds the molecule-to-brand mapping catalog, applies brand/generic/mixed
   exposure classification, and computes pre-launch mention counts for
   quality-control purposes.

3. **Cohort finalization and descriptive statistics** (cells 95–113) —
   Performs the INNER JOIN that reduces the cohort to the 242,312-row
   analytic dataset (excluding 7,781 cases that failed drug-name mapping),
   and generates the demographic/clinical summary statistics reported in
   Table 1 / Supplementary Table 1.

4. **Logistic Regression and Random Forest** (cells 114–137) — Group-aware
   train/validation/test splitting (60/15/25%), Platt and isotonic
   calibration, threshold tuning on the validation set, final test-set
   evaluation, the no-cat_* ablation for both models, and SHAP analysis
   (TreeExplainer for RF, LinearExplainer for LR).

5. **TabularTransformer (TTF)** (cells 138–145) — Model training using the
   `tabular_transformer` library, the corresponding no-cat_* ablation, and
   KernelSHAP-based feature importance.

6. **FTTransformer (FTT)** (cells 146–151) — Model training using
   `torch_frame`, the corresponding no-cat_* ablation, and KernelSHAP-based
   feature importance.

7. **LLM / BERT classifier** (cells 152–154) — Row-to-text serialization,
   fine-tuning of `bert-base-uncased` on the serialized FAERS reports, the
   corresponding no-cat_* ablation, and KernelSHAP-based feature importance
   on the resulting text classifier.

Each modeling section (4–7) follows an identical pattern: group-aware
splitting by `primaryid`, oversampling of the training set only,
threshold selection on the validation set to maximize F1, and final
evaluation on the held-out test set. This consistency is intentional and
allows direct comparison of ROC-AUC, PR-AUC, and Brier score across all
five model families.

## Supplementary files

- **`pt_names_12_CATEGORIES_COMPLETE.xlsx`** — The complete MedDRA Preferred
  Term (PT) to adverse-event-category mapping used to construct the 12
  binary AE-category indicator features (`cat_dose_admin`, `cat_gi`, etc.).
  This file is provided in full for reviewer and reader verification,
  including the PT-to-category leakage audit described in the manuscript's
  response to reviewers.

## Reproducibility notes

- All random seeds are fixed (`SEED = 42`) for data splitting, oversampling,
  and model initialization where supported by the underlying library.
  Note that GPU-based training (TTF, FTT, LLM) may show minor run-to-run
  variation (typically <0.002 AUROC) due to non-deterministic CUDA/cuDNN
  operations even with a fixed seed.
- The LLM classifier requires a CUDA-capable GPU for practical training
  times; CPU training is supported but substantially slower.
- See the ablation cells (one per model family) for the no-AE-category
  feature configuration used to assess the contribution of the 12 binary
  AE-category indicators to overall model discrimination.

## Data availability

FAERS data are publicly available from the FDA
(https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers).
This repository does not redistribute raw FAERS data; the notebook expects
a locally prepared SQLite database (`sqldb`) following the standard FAERS
quarterly file schema.
