# sanity_check.py

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve
)

from utils import get_logger, DATA_DIR, MODELS_DIR

logger = get_logger(__name__)

# === Конфигурация ===
CLASS_TASKS = [
    ("clf_ic50_median", "IC50", True, None),
    ("clf_cc50_median", "CC50", True, None),
    ("clf_si_median", "SI", True, None),
    ("clf_si_gt8", "SI", False, 8),
]

REG_TASKS = [
    ("reg_ic50", "log1p_IC50_nM"),
    ("reg_cc50", "log1p_CC50_mM"),
    ("reg_si", "log1p_SI"),
]

def load_data():
    path = os.path.join(DATA_DIR, "data_prepared.csv")
    if not os.path.exists(path):
        logger.error(f"Файл не найден: {path}")
        return None
    df = pd.read_csv(path)
    for col in ["IC50", "CC50", "SI"]:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col])
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

def check_classification(task_name, target_col, use_median, fixed_threshold):
    logger.info(f"\n🔍 Проверка классификации: {task_name}")
    try:
        path = os.path.join(MODELS_DIR, "classification", task_name)
        model = joblib.load(os.path.join(path, f"{task_name}_model.joblib"))
        imputer = joblib.load(os.path.join(path, f"{task_name}_imputer.joblib"))
        scaler = joblib.load(os.path.join(path, f"{task_name}_scaler.joblib"))
        features = joblib.load(os.path.join(path, f"{task_name}_features.joblib"))
        threshold = fixed_threshold
        if threshold is None:
            threshold_path = os.path.join(path, f"{task_name}_threshold.joblib")
            if os.path.exists(threshold_path):
                threshold = joblib.load(threshold_path)

        df = load_data()
        if df is None:
            return
        df = df.dropna(subset=features + [target_col])
        X = df[features]
        y_raw = df[target_col]

        if use_median:
            threshold = y_raw.median()
        y = (y_raw > threshold).astype(int)

        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)

        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        logger.info(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
        logger.info(f"F1 Score:  {f1_score(y, y_pred):.4f}")
        logger.info(f"Precision: {precision_score(y, y_pred):.4f}")
        logger.info(f"Recall:    {recall_score(y, y_pred):.4f}")
        logger.info(f"ROC AUC:   {roc_auc_score(y, y_proba):.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y, y_proba):.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {task_name}")
        plt.legend()
        out_dir = os.path.join("plots", "sanity_checks")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"roc_{task_name}.png"))
        plt.close()

    except Exception as e:
        logger.error(f"❌ Модель или артефакты не найдены: {e}")

def check_regression(task_name, target_col):
    logger.info(f"\n📈 Проверка регрессии: {task_name}")
    try:
        path = os.path.join(MODELS_DIR, "regression", task_name)
        model = joblib.load(os.path.join(path, f"{task_name}_model.joblib"))
        imputer = joblib.load(os.path.join(path, f"{task_name}_imputer.joblib"))
        scaler = joblib.load(os.path.join(path, f"{task_name}_scaler.joblib"))
        features = joblib.load(os.path.join(path, f"{task_name}_features.joblib"))

        df = load_data()
        if df is None:
            return
        df = df.dropna(subset=features + [target_col])
        X = df[features]
        y = df[target_col]

        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        y_pred = model.predict(X_scaled)

        logger.info(f"R²:    {r2_score(y, y_pred):.4f}")
        logger.info(f"RMSE:  {mean_squared_error(y, y_pred, squared=False):.4f}")
        logger.info(f"MAE:   {mean_absolute_error(y, y_pred):.4f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, alpha=0.5, edgecolors='k')
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Scatter: {task_name}")
        plt.grid(True)
        out_dir = os.path.join("plots", "sanity_checks")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{task_name}_scatter.png"))
        plt.close()

    except Exception as e:
        logger.error(f"❌ Модель или артефакты не найдены: {e}")

def main():
    logger.info("=== 🔍 SANITY CHECK ВСЕХ МОДЕЛЕЙ ===")
    for task, target_col, use_median, threshold in CLASS_TASKS:
        check_classification(task, target_col, use_median, threshold)
    for task, target_col in REG_TASKS:
        check_regression(task, target_col)
    logger.info("=== ✅ Проверка завершена ===")

if __name__ == "__main__":
    main()
