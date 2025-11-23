#!/usr/bin/env python3
"""
Quick test script with minimal parameters to validate the pipeline.
This uses a small subset of data and minimal hyperparameter grid.
"""
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from homework.homework import (
    load_data,
    clean_data,
    split_data,
    create_pipeline,
    save_model,
    calculate_metrics,
    calculate_confusion_matrix,
    save_metrics,
)
from sklearn.model_selection import GridSearchCV

def quick_test():
    """
    Run a quick test with minimal parameters.
    """
    print("=" * 80)
    print("Quick Test - Credit Card Default Prediction")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    train_df, test_df = load_data()
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Clean data
    print("\n2. Cleaning data...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    print(f"Train data shape after cleaning: {train_df.shape}")
    print(f"Test data shape after cleaning: {test_df.shape}")

    # Split data - use small subset for quick test
    print("\n3. Splitting data...")
    X_train, y_train = split_data(train_df)
    X_test, y_test = split_data(test_df)
    
    # Use only first 1000 samples for quick test
    X_train_small = X_train.head(1000)
    y_train_small = y_train.head(1000)
    print(f"X_train_small shape: {X_train_small.shape}")
    print(f"y_train_small shape: {y_train_small.shape}")

    # Create pipeline
    print("\n4. Creating ML pipeline...")
    pipeline = create_pipeline()

    # Train with minimal grid (quick test)
    print("\n5. Training model with minimal parameters (quick test)...")
    param_grid = {
        "feature_selection__k": [10],
        "classifier__C": [1.0],
        "classifier__solver": ["lbfgs"],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # Only 3 folds for quick test
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train_small, y_train_small)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Save model
    print("\n6. Saving model...")
    save_model(grid_search)

    # Calculate metrics on full datasets
    print("\n7. Calculating metrics on full datasets...")
    metrics_list = []

    # Train metrics
    train_metrics = calculate_metrics(grid_search, X_train, y_train, "train")
    metrics_list.append(train_metrics)
    print(f"Train metrics: {train_metrics}")

    # Test metrics
    test_metrics = calculate_metrics(grid_search, X_test, y_test, "test")
    metrics_list.append(test_metrics)
    print(f"Test metrics: {test_metrics}")

    # Train confusion matrix
    train_cm = calculate_confusion_matrix(grid_search, X_train, y_train, "train")
    metrics_list.append(train_cm)
    print(f"Train confusion matrix: {train_cm}")

    # Test confusion matrix
    test_cm = calculate_confusion_matrix(grid_search, X_test, y_test, "test")
    metrics_list.append(test_cm)
    print(f"Test confusion matrix: {test_cm}")

    # Save metrics
    print("\n8. Saving metrics...")
    save_metrics(metrics_list)

    print("\n" + "=" * 80)
    print("Quick test completed successfully!")
    print("=" * 80)
    
    return grid_search


if __name__ == "__main__":
    model = quick_test()
