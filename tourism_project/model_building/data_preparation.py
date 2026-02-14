"""
Data Preparation:
- Load dataset from Hugging Face
- Clean data and remove unnecessary columns
- Split into train/test
- Upload train/test back to Hugging Face
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download


def load_data_from_hf():
    """Load the tourism dataset from Hugging Face Hub."""
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = "Matheshwaran/tourism-dataset"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename="tourism.csv",
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df):
    """
    Perform data cleaning:
    - Drop unnecessary columns (unnamed index, CustomerID)
    - Fix Gender inconsistencies (Fe Male -> Female)
    - Handle missing values
    - Encode categorical variables
    """
    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop CustomerID as it's just an identifier
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Fix Gender inconsistency: 'Fe Male' should be 'Female'
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

    # Handle missing values
    # Numerical columns: fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables
    df["TypeofContact"] = df["TypeofContact"].map(
        {"Self Enquiry": 0, "Company Invited": 1}
    )

    occupation_map = {
        "Salaried": 0,
        "Small Business": 1,
        "Large Business": 2,
        "Free Lancer": 3,
    }
    df["Occupation"] = df["Occupation"].map(occupation_map)

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    product_map = {
        "Basic": 0,
        "Standard": 1,
        "Deluxe": 2,
        "Super Deluxe": 3,
        "King": 4,
    }
    df["ProductPitched"] = df["ProductPitched"].map(product_map)

    marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Unmarried": 3}
    df["MaritalStatus"] = df["MaritalStatus"].map(marital_map)

    designation_map = {
        "Executive": 0,
        "Manager": 1,
        "Senior Manager": 2,
        "AVP": 3,
        "VP": 4,
    }
    df["Designation"] = df["Designation"].map(designation_map)

    # Handle any remaining NaN from mapping failures
    df = df.dropna()

    print(f"Data cleaned successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def split_and_save(df):
    """Split data into train/test and save locally."""
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Combine X and y back for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save locally
    os.makedirs("tourism_project/data", exist_ok=True)
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)

    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    return train_df, test_df


def upload_splits_to_hf():
    """Upload train and test CSV files to Hugging Face Hub."""
    api = HfApi()
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = "Matheshwaran/tourism-dataset"

    for filename in ["train.csv", "test.csv"]:
        api.upload_file(
            path_or_fileobj=f"tourism_project/data/{filename}",
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        print(f"Uploaded {filename} to {repo_id}")


if __name__ == "__main__":
    # Step 1: Load from HF
    df = load_data_from_hf()

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: Split and save locally
    split_and_save(df)

    # Step 4: Upload splits to HF
    upload_splits_to_hf()

    print("\nData preparation completed successfully!")
