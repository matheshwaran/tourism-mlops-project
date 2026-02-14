"""
Register the tourism dataset on Hugging Face Hub.
"""
import os
from huggingface_hub import HfApi


def register_dataset():
    """Upload tourism.csv to Hugging Face Dataset space."""
    api = HfApi()
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = "Matheshwaran/tourism-dataset"

    # Create the dataset repo if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=hf_token,
    )

    # Upload the CSV file
    api.upload_file(
        path_or_fileobj="tourism_project/data/tourism.csv",
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )
    print(f"Dataset registered successfully at: {repo_id}")


if __name__ == "__main__":
    register_dataset()
