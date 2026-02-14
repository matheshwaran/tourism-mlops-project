"""
Hosting Script:
Push all deployment files (Dockerfile, app.py, requirements.txt)
to the Hugging Face Space for hosting the Streamlit app.
"""
import os
from huggingface_hub import HfApi


def deploy_to_hf_space():
    """Push deployment files to Hugging Face Space."""
    api = HfApi()
    hf_token = os.environ.get("HF_TOKEN")
    space_repo_id = "Matheshrangasamy/tourism-app"

    # Create the space if it doesn't exist (Docker SDK)
    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=hf_token,
    )

    # Files to upload to the HF Space
    deployment_files = {
        "tourism_project/deployment/Dockerfile": "Dockerfile",
        "tourism_project/deployment/app.py": "app.py",
        "tourism_project/deployment/requirements.txt": "requirements.txt",
    }

    for local_path, repo_path in deployment_files.items():
        if os.path.exists(local_path):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=space_repo_id,
                repo_type="space",
                token=hf_token,
            )
            print(f"Uploaded {local_path} -> {repo_path}")
        else:
            print(f"WARNING: {local_path} not found!")

    print(f"\nDeployment complete! Visit: https://huggingface.co/spaces/{space_repo_id}")


if __name__ == "__main__":
    deploy_to_hf_space()
