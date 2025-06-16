# clf_si_gt8.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    confusion_matrix
)
from catboost import CatBoostClassifier

# === Логгирование ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Константы ===
DATA_PATH = "data/data_prepared.csv"
PLOTS_DIR = "plots/classification/clf_si_gt8"
MODELS_DIR = "models/classification"
TARGET_COLUMN = "SI"
TARGET_THRESHOLD = 8
TASK_NAME = "clf_si_gt8"
RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# === Функции ===
def load_data():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Данные успешно загружены из '{DATA_PATH}'. Размер: {df.shape}")
    return df

def preprocess_data(df):
    y = (df[TARGET_COLUMN] > TARGET_THRESHOLD).astype(int)
    logger.info(f"Задача: '{TARGET_COLUMN} > {TARGET_THRESHOLD}'. Баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    leakage_cols = [
        "IC50", "CC50", "IC50_mM", "CC50_mM", "SI", "SI_corrected", "SI_original",
        "log_IC50", "log_CC50", "log_SI", "log1p_IC50_nM", "log1p_SI_corrected",
        "SI_diff", "SI_diff_check", "SI_check", "ratio_IC50_CC50"

    ]
    features = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors='ignore')
    X = features.select_dtypes(include=np.number).copy()

    logger.info(f"Используем {X.shape[1]} числовых признаков.")
    return X, y, X.columns.tolist()

def train_and_evaluate(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        eval_metric="AUC",
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=0
    )

    metrics = []
    y_proba_oof = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Фолд {fold}/{N_SPLITS}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_prep = pipeline.fit_transform(X_train)
        X_val_prep = pipeline.transform(X_val)

        model.fit(X_train_prep, y_train, eval_set=(X_val_prep, y_val))
        y_pred = model.predict(X_val_prep)
        y_proba = model.predict_proba(X_val_prep)[:, 1]
        y_proba_oof[val_idx] = y_proba

        fold_metrics = {
            "Accuracy": accuracy_score(y_val, y_pred),
            "F1": f1_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "ROC AUC": roc_auc_score(y_val, y_proba)
        }
        metrics.append(fold_metrics)
        logger.info(f"Метрики (фолд {fold}): {fold_metrics}")

    # === Средние метрики ===
    logger.info(f"Средние метрики по {N_SPLITS}-CV:")
    for key in metrics[0].keys():
        avg = np.mean([m[key] for m in metrics])
        logger.info(f"{key}: {avg:.4f}")

    return model, pipeline, y_proba_oof

def find_optimal_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    logger.info(f" Оптимальный порог (Youden's J): {best_threshold:.4f} (J-стат: {j_scores[best_idx]:.4f})")
    return best_threshold

def plot_roc(y_true, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая (CatBoost, OOF)")
    plt.legend()
    plt.grid(True)
    path = f"{PLOTS_DIR}/roc_curve_catboost.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC-кривая сохранена: {path}")

def save_artifacts(model, pipeline, features, threshold):
    joblib.dump(model, f"{MODELS_DIR}/{TASK_NAME}_model.joblib")
    joblib.dump(pipeline, f"{MODELS_DIR}/{TASK_NAME}_preprocessor.joblib")
    joblib.dump(features, f"{MODELS_DIR}/{TASK_NAME}_features.joblib")
    joblib.dump(threshold, f"{MODELS_DIR}/{TASK_NAME}_optimal_threshold.joblib")
    logger.info(f"Артефакты сохранены: {MODELS_DIR}/{TASK_NAME}_*")

# === Main ===
def main():
    logger.info(f"=== Запуск задачи: {TASK_NAME} ===")

    df = load_data()
    X, y, features = preprocess_data(df)
    model, preprocessor, y_proba = train_and_evaluate(X, y)
    threshold = find_optimal_threshold(y, y_proba)

    # ROC и CM
    plot_roc(y, y_proba)
    y_pred_opt = (y_proba >= threshold).astype(int)
    logger.info(f"\n{classification_report(y, y_pred_opt, zero_division=0)}")

    cm = confusion_matrix(y, y_pred_opt)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Threshold={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path_cm = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(path_cm, bbox_inches='tight')
    plt.close()
    logger.info(f"Матрица ошибок сохранена: {path_cm}")

    # SHAP
    logger.info(" SHAP-анализ финальной модели...")
    try:
        X_proc = preprocessor.transform(X)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc)
        shap.summary_plot(shap_values, X, show=False, plot_type="bar", max_display=20)
        plt.title("SHAP Summary (bar)")
        plt.savefig(f"{PLOTS_DIR}/shap_bar.png", bbox_inches='tight')
        plt.close()

        shap.summary_plot(shap_values, X, show=False, plot_type="beeswarm", max_display=20)
        plt.title("SHAP Summary (beeswarm)")
        plt.savefig(f"{PLOTS_DIR}/shap_beeswarm.png", bbox_inches='tight')
        plt.close()
        logger.info("SHAP-графики сохранены.")
    except Exception as e:
        logger.warning(f"SHAP-анализ не выполнен: {e}")

    save_artifacts(model, preprocessor, features, threshold)
    logger.info(f" Задача {TASK_NAME} завершена.")

if __name__ == "__main__":
    main()
