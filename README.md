# Arbix-ai Assessment Submission | Round 1

| Item       | Details                    |
|------------|----------------------------|
| Start Time | 28-May-2025, 07:02 PM     |
| End Time | 28-May-2025, 08:33 PM     |

## Summary

This submission audits and rewrites the provided baseline into a safer, more auditable training pipeline for an entitlement-score model. The focus is leakage prevention, temporal validation, artifact reproducibility, and basic explainability rather than maximizing headline `R2`.

## Files

- `fixed_yogyank_training.py`: corrected training script
- `audit_memo.md`: plain-English audit summary
- `LLM_NOTES.md`: disclosed AI-tool usage
- `artifacts/`: saved model pipeline, schema, metadata, metrics, and sample reason codes

## Setup

Install dependencies:

```powershell
python -m pip install pandas scikit-learn numpy scipy joblib
```

## Run

```powershell
python .\fixed_yogyank_training.py
```

The script expects `farmer_scoring_sample_yogyank_round1_final.csv` in the same folder.

## Validation Approach

I used a temporal holdout:

- Train on applications before `2024`
- Validate on applications from `2024`

This is safer than a random shuffle split because it better approximates future scoring conditions and reduces the chance of hidden temporal leakage.

## What I Fixed

- Removed direct target manipulation based on `pm_kisan_status`
- Dropped `defaulted_in_next_12_months` because it is future information at scoring time
- Replaced ad hoc label encoding with frozen one-hot encoding that handles unseen categories
- Switched from random shuffled validation to temporal validation
- Saved a full pipeline plus schema, version metadata, metrics, and sample reason codes
- Added simple per-row top-3 reason code generation

## Assumptions

- `application_year` is available only for splitting and monitoring, not used as a predictive input
- The supplied CSV is a synthetic assessment sample and not a final production training source
- The versioned policy layer sits outside the model and is responsible for any downstream eligibility or cutoff logic

## What I Skipped Due To Time

- Formal fairness hypothesis tests and threshold governance
- Hyperparameter tuning and comparison across multiple model families
- A dedicated inference CLI or service wrapper
- Unit tests and CI automation


## Latest Local Run

The latest verified local run produced approximately:

- Validation `R2`: `0.3156`
- Validation `MAE`: `81.52`

These numbers are intentionally much more modest than the original script because the leakage and target contamination were removed.
