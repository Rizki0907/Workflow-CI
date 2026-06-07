import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
import warnings
warnings.filterwarnings('ignore')

# ── MLflow setup ──────────────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("Heart Disease Classification")

mlflow.set_experiment("Heart Disease Classification")

# ── Load dataset ────────────────────────────────────────────────────
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(os.path.join(base_dir, 'heart_preprocessed/heart_train.csv'))
    test  = pd.read_csv(os.path.join(base_dir, 'heart_preprocessed/heart_test.csv'))
    
    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test  = test.drop('target', axis=1)
    y_test  = test['target']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ── Plot helpers ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, run_id):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Sakit', 'Sakit'],
                yticklabels=['Tidak Sakit', 'Sakit'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = f'confusion_matrix_{run_id}.png'
    plt.savefig(path)
    plt.close()
    return path

def plot_feature_importance(model, feature_names, run_id):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices],
                y=[feature_names[i] for i in indices],
                palette='viridis')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    path = f'feature_importance_{run_id}.png'
    plt.savefig(path)
    plt.close()
    return path

# ── Main training ────────────────────────────────────────────────────
def train():
    X_train, X_test, y_train, y_test = load_data()
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest-Autolog") as run:
        run_id = run.info.run_id[:8]
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_pred)
        
        print("="*50)
        print("HASIL TRAINING")
        print("="*50)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"AUC-ROC  : {auc:.4f}")
        
        # Artefak tambahan
        cm_path = plot_confusion_matrix(y_test, y_pred, run_id)
        fi_path = plot_feature_importance(model, list(X_train.columns), run_id)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        
        # Cleanup
        os.remove(cm_path)
        os.remove(fi_path)
        
        print(f"\nRun ID : {run.info.run_id}")
        print("Artefak berhasil disimpan ke DagsHub!")

if __name__ == "__main__":
    train()