"""
Model Training with Experimentation Tracking:
- Load train/test from HF
- Define model and hyperparameters
- Tune with GridSearchCV
- Log parameters and metrics with MLflow
- Register best model on HF Model Hub
"""
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from huggingface_hub import HfApi, hf_hub_download


def load_data_from_hf():
    """Load train and test data from Hugging Face Hub."""
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = "Matheshrangasamy/tourism-dataset"

    train_path = hf_hub_download(
        repo_id=repo_id, filename="train.csv",
        repo_type="dataset", token=hf_token,
    )
    test_path = hf_hub_download(
        repo_id=repo_id, filename="test.csv",
        repo_type="dataset", token=hf_token,
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def train_and_tune_model(X_train, y_train):
    """
    Define model and hyperparameters, perform GridSearchCV tuning.
    Using Gradient Boosting Classifier.
    """
    # Define the model
    model = GradientBoostingClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid,
        cv=3, scoring="f1", n_jobs=-1, verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    print("\n--- Model Evaluation ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def save_and_register_model(model, feature_names):
    """Save model locally and register on Hugging Face Model Hub."""
    os.makedirs("tourism_project/model_building", exist_ok=True)

    # Save model with joblib
    model_path = "tourism_project/model_building/best_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved locally at: {model_path}")

    # Save feature names for inference
    feature_path = "tourism_project/model_building/feature_names.joblib"
    joblib.dump(feature_names, feature_path)

    # Upload to Hugging Face Model Hub
    api = HfApi()
    hf_token = os.environ.get("HF_TOKEN")
    model_repo_id = "Matheshrangasamy/tourism-model"

    # Create model repo if not exists
    api.create_repo(
        repo_id=model_repo_id, repo_type="model",
        exist_ok=True, token=hf_token,
    )

    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.joblib",
        repo_id=model_repo_id, repo_type="model", token=hf_token,
    )

    # Upload feature names
    api.upload_file(
        path_or_fileobj=feature_path,
        path_in_repo="feature_names.joblib",
        repo_id=model_repo_id, repo_type="model", token=hf_token,
    )

    print(f"Model registered on HF Hub: {model_repo_id}")


def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Tourism_Package_Prediction")

    # Load data
    train_df, test_df = load_data_from_hf()

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]

    feature_names = X_train.columns.tolist()

    with mlflow.start_run(run_name="GradientBoosting_Tourism"):
        # Train and tune
        best_model, best_params = train_and_tune_model(X_train, y_train)

        # Log parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test)

        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log model
        mlflow.sklearn.log_model(best_model, "gradient_boosting_model")

        # Save and register on HF
        save_and_register_model(best_model, feature_names)

    print("\nModel training and registration completed!")


if __name__ == "__main__":
    main()
