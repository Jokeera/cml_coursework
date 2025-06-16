# plot.py (ФИНАЛЬНАЯ ВЕРСИЯ — ПРОВЕРЕННАЯ)

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_logger, PLOTS_DIR, MODELS_DIR, DATA_DIR, FINAL_DATA_REGRESS
from sklearn.metrics import roc_curve, auc

logger = get_logger(__name__)
sns.set(style="whitegrid")

# === Задачи ===
TASKS = {
    "regression": [
        "reg_log1p_IC50_nM",
        "reg_log1p_CC50_mM",
        "reg_si"
    ],
    "classification": [
        "clf_ic50nm_median_mi_top20_tuned_stack",
        "clf_cc50mm_median_mi_top20_tuned_stack",
        "clf_si_median_mi_top20_tuned_stack",
        "clf_si_gt8"
    ]
}

# === SHAP для моделей ===
def generate_shap_plots(task: str, task_type: str):
    logger.info(f"📊 SHAP: {task}")
    try:
        model = joblib.load(os.path.join(MODELS_DIR, task_type, task, f"{task}_model.joblib"))
        features = joblib.load(os.path.join(MODELS_DIR, task_type, task, f"{task}_features.joblib"))
        df = pd.read_csv(FINAL_DATA_REGRESS)
        X = df[features].copy()

        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        plot_dir = os.path.join(PLOTS_DIR, task_type, task)
        os.makedirs(plot_dir, exist_ok=True)

        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{task}_shap_bar.png"))
        plt.close()

        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{task}_shap_beeswarm.png"))
        plt.close()

        top_features = np.argsort(np.abs(shap_values.values).mean(0))[-3:]
        for idx in top_features:
            shap.plots.scatter(shap_values[:, idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{task}_shap_dependence_{features[idx]}.png"))
            plt.close()

        logger.info(f"✅ SHAP-графики сохранены: {plot_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка SHAP для {task}: {e}")

# === ROC-кривая ===
def generate_roc_curve(task: str):
    logger.info(f"🔍 ROC-кривая: {task}")
    try:
        y_true = joblib.load(os.path.join(MODELS_DIR, "classification", task, f"{task}_y_true.joblib"))
        y_proba = joblib.load(os.path.join(MODELS_DIR, "classification", task, f"{task}_y_pred_proba.joblib"))

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plot_dir = os.path.join(PLOTS_DIR, "classification", task)
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {task}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{task}_roc_curve.png"))
        plt.close()

        logger.info(f"✅ ROC-кривая сохранена: {plot_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка ROC-кривой: {e}")

# === Гистограммы ===
def generate_eda_histograms():
    logger.info("📊 Генерация гистограмм таргетов")
    df_path = FINAL_DATA_REGRESS
    if not os.path.exists(df_path):
        logger.warning("⚠️ Нет файла data_final_reg.csv")
        return

    df = pd.read_csv(df_path)
    columns = [
        "IC50_nM", "log1p_IC50_nM",
        "CC50_mM", "log1p_CC50_mM",
        "SI_corrected", "log1p_SI"
    ]
    for col in columns:
        if col not in df.columns:
            logger.warning(f"⚠️ Колонка {col} отсутствует в данных")
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Распределение: {col}")
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, "eda", f"hist_{col}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"✅ Гистограмма сохранена: {save_path}")

# === Main ===
def main():
    logger.info("=== Генерация всех графиков ===")
    for task in TASKS["regression"]:
        generate_shap_plots(task, "regression")

    for task in TASKS["classification"]:
        generate_shap_plots(task, "classification")
        generate_roc_curve(task)

    generate_eda_histograms()

if __name__ == "__main__":
    main()
