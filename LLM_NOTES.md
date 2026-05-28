## Tools used

- Codex (GPT-5 coding assistant)

## Where used

- Flaw identification
- Code refactoring
- Verification runs
- Memo drafting

## Representative prompts

- Review the assessment script and identify leakage, validation, encoding, and artifact risks.
- Rewrite the training script into a safer, audit-conscious baseline with time-based validation.
- Add saved artifacts and a simple top-3 reason-code mechanism.
- Draft submission notes that explain what was fixed and what still remains risky.

## What I accepted

- Using a chronological split instead of a random split.
- Saving a bundled pipeline plus metadata, schema, and sample reason-code artifacts.

## What I rejected or corrected

- I rejected the idea of preserving the original high-R2 setup because it relied on post-outcome leakage and target tampering.
- I corrected an intermediate artifact-schema implementation so the saved schema reflects the actual dataframe columns and dtypes rather than an internal object description.

## What I personally verified

- Leakage check: `defaulted_in_next_12_months` was removed from model features.
- OOT split: training uses `2022`, validation uses `2023`, and holdout uses `2024`.
- Encoder boundary: preprocessing is frozen inside the saved pipeline and unknown categories are handled with `ignore`.
- Saved artifacts: the bundle, metadata, schema, manifest, reason-code examples, and slice metrics are written under `artifacts/`.
- Reason-code logic: the script emits top-3 local reason codes using XGBoost contribution values aggregated to raw features.
- Run output: the script runs successfully and reports validation and holdout metrics.
