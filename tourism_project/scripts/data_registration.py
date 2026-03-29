"""Step 1: Register raw dataset on Hugging Face."""
import os, shutil
import pandas as pd
from huggingface_hub import login, HfApi

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
print(f"Raw dataset uploaded: https://huggingface.co/datasets/{DATASET_REPO}")
