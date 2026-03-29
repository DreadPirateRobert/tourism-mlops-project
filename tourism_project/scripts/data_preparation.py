"""Step 2: Clean, encode, split, and upload train/test datasets."""
import os, numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"

login(token=HF_TOKEN)
df = load_dataset(DATASET_REPO, data_files="tourism.csv", split="train").to_pandas()

df['Gender'] = df['Gender'].str.strip().str.replace('Fe Male', 'Female', regex=False)
df.drop(columns=['CustomerID'], inplace=True)

num_cols = df.select_dtypes(include=[np.number]).columns.difference(['ProdTaken']).tolist()
for c in num_cols: df[c].fillna(df[c].median(), inplace=True)

cat_cols = df.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)
    df[c] = LabelEncoder().fit_transform(df[c])

X, y = df.drop(columns=['ProdTaken']), df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = X_train.copy(); train_df['ProdTaken'] = y_train.values
test_df  = X_test.copy();  test_df['ProdTaken']  = y_test.values

os.makedirs("tourism_project/data", exist_ok=True)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv",   index=False)

api = HfApi()
for path, name in [("tourism_project/data/train.csv","train.csv"),
                   ("tourism_project/data/test.csv","test.csv")]:
    api.upload_file(path_or_fileobj=path, path_in_repo=name,
                    repo_id=DATASET_REPO, repo_type="dataset")
print("Train/Test datasets uploaded.")
