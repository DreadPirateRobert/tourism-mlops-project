"""Step 3: Train models, track with MLflow, register best on HF."""
import os, mlflow, mlflow.sklearn, joblib
import pandas as pd
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from huggingface_hub import login, HfApi

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO   = f"{HF_USERNAME}/tourism-best-model"

login(token=HF_TOKEN)
train_df = load_dataset(DATASET_REPO, data_files="train.csv", split="train").to_pandas()
test_df  = load_dataset(DATASET_REPO, data_files="test.csv",  split="train").to_pandas()
X_train, y_train = train_df.drop(columns=["ProdTaken"]), train_df["ProdTaken"]
X_test,  y_test  = test_df.drop(columns=["ProdTaken"]),  test_df["ProdTaken"]

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("tourism_package_prediction")

experiments = [
    ("DecisionTree",     DecisionTreeClassifier,     [{"max_depth":3},{"max_depth":5}]),
    ("RandomForest",     RandomForestClassifier,     [{"n_estimators":100},{"n_estimators":200}]),
    ("GradientBoosting", GradientBoostingClassifier, [{"n_estimators":100,"learning_rate":0.1}]),
    ("XGBoost",          XGBClassifier,              [{"n_estimators":100,"verbosity":0,
                                                       "eval_metric":"logloss"}]),
]

best = {"model":None,"f1":0,"name":None}
for name, Cls, params_list in experiments:
    for params in params_list:
        kw = {**params}
        if "random_state" in Cls().get_params(): kw["random_state"] = 42
        with mlflow.start_run(run_name=f"{name}"):
            m = Cls(**kw); m.fit(X_train, y_train)
            preds = m.predict(X_test)
            proba = m.predict_proba(X_test)[:,1]
            mlflow.log_param("model_type", name); mlflow.log_params(params)
            mlflow.log_metrics({"accuracy":accuracy_score(y_test,preds),
                                "f1_score":f1_score(y_test,preds),
                                "roc_auc":roc_auc_score(y_test,proba)})
            mlflow.sklearn.log_model(m, "model")
            if f1_score(y_test,preds) > best["f1"]:
                best.update(model=m, f1=f1_score(y_test,preds), name=name)

os.makedirs("tourism_project/model_building", exist_ok=True)
path = "tourism_project/model_building/best_model.pkl"
joblib.dump(best["model"], path)
api = HfApi()
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=path, path_in_repo="best_model.pkl",
                repo_id=MODEL_REPO, repo_type="model")
print(f"Best model ({best['name']}, F1={best['f1']:.4f}) registered: https://huggingface.co/{MODEL_REPO}")
