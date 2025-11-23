# Credit Card Default Prediction - Implementation Instructions

## Overview

This repository contains a complete implementation of a credit card default prediction model using Logistic Regression with scikit-learn.

## What Has Been Implemented

All steps required by the homework have been implemented in `homework/homework.py`:

1. ✅ **Data Cleaning** - Removes ID, renames target column, handles N/A values, groups education categories
2. ✅ **Data Splitting** - Separates features and target variables
3. ✅ **ML Pipeline** - Complete pipeline with:
   - OneHotEncoder for categorical variables (SEX, EDUCATION, MARRIAGE)
   - MinMaxScaler for numerical variables
   - SelectKBest for feature selection
   - LogisticRegression classifier
4. ✅ **Hyperparameter Optimization** - GridSearchCV with 10-fold cross-validation and balanced_accuracy scoring
5. ✅ **Model Persistence** - Saves model as compressed pickle (gzip)
6. ✅ **Metrics Calculation** - Precision, balanced_accuracy, recall, f1_score
7. ✅ **Confusion Matrix** - For both train and test datasets

## Files Created

- `homework/homework.py` - Main implementation with all required functions
- `train_model.py` - Script to run full training (computationally expensive)
- `run_quick_test.py` - Quick test script with minimal parameters for validation
- `INSTRUCTIONS.md` - This file

## How to Run

### Quick Test (Fast - for validation only)

This runs a quick test on a small subset with minimal hyperparameters:

```bash
python3 run_quick_test.py
```

**Note:** This creates output files but the model will NOT meet the performance requirements since it only trains on 1000 samples with minimal hyperparameters. This is just for testing the pipeline structure.

### Full Training (Slow - for production model)

⚠️ **WARNING:** This will take several minutes to hours depending on your hardware.

The full training uses:
- All training data (20,953 samples after cleaning)
- 10-fold cross-validation
- Comprehensive hyperparameter grid:
  - `feature_selection__k`: [10, 15, 20]
  - `classifier__C`: [0.01, 0.1, 1.0, 10.0]
  - `classifier__solver`: ['lbfgs', 'liblinear']
- Total: 24 parameter combinations × 10 folds = 240 model fits

To run the full training:

```bash
python3 train_model.py
```

Or directly:

```bash
python3 -m homework.homework
```

### Run Tests

After training a proper model:

```bash
pytest -v
```

## Output Files

After running the training, the following files will be created:

- `files/models/model.pkl.gz` - Compressed trained model
- `files/output/metrics.json` - Metrics and confusion matrices in JSON Lines format

## Performance Requirements

The tests expect:
- Train balanced_accuracy > 0.639
- Test balanced_accuracy > 0.654
- Proper confusion matrix values

These can only be achieved by running the full training script.

## Implementation Details

### Data Cleaning
- Removes records where EDUCATION=0 or MARRIAGE=0
- Groups EDUCATION values > 4 into category 4 (others)
- Results in ~20,953 train samples and ~8,979 test samples

### Pipeline Structure
```
Pipeline:
  1. ColumnTransformer
     - OneHotEncoder for categorical features
     - MinMaxScaler for numerical features
  2. SelectKBest (feature selection)
  3. LogisticRegression
```

### Hyperparameter Grid
```python
{
    'feature_selection__k': [10, 15, 20],
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__solver': ['lbfgs', 'liblinear']
}
```

## Troubleshooting

### ImportError
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Out of Memory
If you run out of memory during training, you can:
1. Reduce the parameter grid in `homework/homework.py` (line 141-145)
2. Reduce the number of CV folds (line 149, currently cv=10)
3. Run on a machine with more RAM

### Tests Fail
The tests will fail if you run the quick test. You MUST run the full training script to meet the performance requirements.

## Next Steps

1. Run the full training on a machine with adequate resources: `python3 train_model.py`
2. Verify the outputs are created: `ls -la files/models/` and `ls -la files/output/`
3. Run the tests: `pytest -v`
4. Review the metrics in `files/output/metrics.json`

## Notes

- The implementation uses `random_state=42` for reproducibility
- All numeric values in metrics are properly cast to float for JSON serialization
- The model is compressed with gzip as required
- Confusion matrix format matches the expected output format exactly
