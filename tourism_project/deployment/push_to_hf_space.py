"""Push deployment files to a Hugging Face Streamlit Space."""
import os
from huggingface_hub import HfApi

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME", "")
SPACE_REPO  = f"{HF_USERNAME}/tourism-predictor"

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=SPACE_REPO, repo_type="space",
                space_sdk="docker", exist_ok=True)
print(f"Space repo ready: https://huggingface.co/spaces/{SPACE_REPO}")

base = os.path.dirname(os.path.abspath(__file__))
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        api.upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                        repo_id=SPACE_REPO, repo_type="space")
        print(f"  Uploaded: {fname}")

print(f"\n✅ Space live at: https://huggingface.co/spaces/{SPACE_REPO}")
