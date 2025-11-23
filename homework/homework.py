# flake8: noqa: E501
"""
Credit Card Default Prediction using Logistic Regression
"""
import gzip
import json
import os
import pickle
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data():
    """Load train and test data from zip files."""
    # Load train data
    with zipfile.ZipFile("files/input/train_data.csv.zip", "r") as z:
        with z.open("train_default_of_credit_card_clients.csv") as f:
            train_df = pd.read_csv(f)

    # Load test data
    with zipfile.ZipFile("files/input/test_data.csv.zip", "r") as z:
        with z.open("test_default_of_credit_card_clients.csv") as f:
            test_df = pd.read_csv(f)

    return train_df, test_df


def clean_data(df):
    """
    Step 1: Clean the dataset.
    - Rename 'default payment next month' to 'default'
    - Remove 'ID' column
    - Remove records with N/A information (EDUCATION=0, MARRIAGE=0)
    - Group EDUCATION values > 4 into category 4 (others)
    """
    # Rename column
    df = df.rename(columns={"default payment next month": "default"})

    # Remove ID column
    df = df.drop(columns=["ID"])

    # Remove records with N/A information
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    # Group EDUCATION > 4 into category 4
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


def split_data(df):
    """
    Step 2: Split dataset into features and target.
    """
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


def create_pipeline():
    """
    Step 3: Create ML pipeline with:
    - OneHotEncoder for categorical variables (SEX, EDUCATION, MARRIAGE)
    - MinMaxScaler for numerical variables
    - SelectKBest for feature selection
    - LogisticRegression model
    """
    # Define categorical and numerical columns
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
            ("num", MinMaxScaler(), numerical_features),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    return pipeline


def train_model(pipeline, X_train, y_train):
    """
    Step 4: Optimize hyperparameters using GridSearchCV.
    Use 10-fold cross-validation and balanced_accuracy scoring.
    """
    # Define parameter grid
    param_grid = {
        "feature_selection__k": [10, 15, 20],
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    # Create GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    # Fit the model
    print("Training model... This may take a while.")
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search


def save_model(model, filename="files/models/model.pkl.gz"):
    """
    Step 5: Save model as compressed pickle.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def calculate_metrics(model, X, y, dataset_name):
    """
    Step 6: Calculate metrics for a dataset.
    """
    y_pred = model.predict(X)

    metrics = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred)),
    }

    return metrics


def calculate_confusion_matrix(model, X, y, dataset_name):
    """
    Step 7: Calculate confusion matrix for a dataset.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }

    return cm_dict


def save_metrics(metrics_list, filename="files/output/metrics.json"):
    """
    Save metrics to JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")
    print(f"Metrics saved to {filename}")


def main():
    """
    Main function to run the complete pipeline.
    """
    print("=" * 80)
    print("Credit Card Default Prediction - Logistic Regression")
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

    # Split data
    print("\n3. Splitting data into features and target...")
    X_train, y_train = split_data(train_df)
    X_test, y_test = split_data(test_df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Create pipeline
    print("\n4. Creating ML pipeline...")
    pipeline = create_pipeline()
    print("Pipeline created successfully")

    # Train model
    print("\n5. Training model with hyperparameter optimization...")
    model = train_model(pipeline, X_train, y_train)

    # Save model
    print("\n6. Saving model...")
    save_model(model)

    # Calculate metrics
    print("\n7. Calculating metrics...")
    metrics_list = []

    # Train metrics
    train_metrics = calculate_metrics(model, X_train, y_train, "train")
    metrics_list.append(train_metrics)
    print(f"Train metrics: {train_metrics}")

    # Test metrics
    test_metrics = calculate_metrics(model, X_test, y_test, "test")
    metrics_list.append(test_metrics)
    print(f"Test metrics: {test_metrics}")

    # Train confusion matrix
    train_cm = calculate_confusion_matrix(model, X_train, y_train, "train")
    metrics_list.append(train_cm)
    print(f"Train confusion matrix: {train_cm}")

    # Test confusion matrix
    test_cm = calculate_confusion_matrix(model, X_test, y_test, "test")
    metrics_list.append(test_cm)
    print(f"Test confusion matrix: {test_cm}")

    # Save metrics
    print("\n8. Saving metrics...")
    save_metrics(metrics_list)

    print("\n" + "=" * 80)
    print("Process completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
