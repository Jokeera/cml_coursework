import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.impute import SimpleImputer
from utils import get_logger

# === Логгер и директории ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "sanity_checks")
os.makedirs(PLOTS_DIR, exist_ok=True)
logger = get_logger(__name__)

# === Конфигурация задач ===
FINAL_DATA_PATH = "data/eda_gen/data_final.csv"

TASKS = {
    # === Классификация ===
    "clf_ic50nm_median": {
        "type": "classification",
        "target_col": "IC50_gt_median",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/clf_ic50_median/model_clf_ic50nm_median_mi_top99_tuned_stack_catboost.joblib",
        "features_path": "features/selected_by_shap_clf_ic50nm_median_mi_top99_tuned_stack.txt",
    },
    "clf_cc50nm_median": {
        "type": "classification",
        "target_col": "CC50_gt_median",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/clf_cc50_median/model_clf_cc50nm_median_mi_top99_tuned_stack_xgboost.joblib",
        "features_path": "features/selected_by_shap_clf_cc50nm_median_mi_top99_tuned_stack.txt",
    },
        "clf_si_median": {
        "type": "classification",
        "target_col": "SI_gt_median",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/clf_si_median/model_clf_si_median_xgboost.joblib",
        # ИСПРАВЛЕНО: Указан правильный, короткий путь к файлу признаков
        "features_path": "features/selected_by_shap_clf_si_median.txt",
    },
    "clf_si_gt8": {
        "type": "classification",
        "target_col": "SI_gt_8",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/clf_si_gt8/model_clf_si_gt8.joblib",
        "features_path": "models/clf_si_gt8/features.joblib",
        # ИСПРАВЛЕНО: preprocessor_path удален, т.к. он уже внутри модели (пайплайна)
        # "preprocessor_path": "models/clf_si_gt8/preprocessor.joblib", 
        "threshold_path": "models/clf_si_gt8/optimal_threshold.joblib",
    },

    # === Регрессия ===
    "reg_log1p_IC50_nM": {
        "type": "regression",
        "target_col": "log1p_IC50_nM",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/regression/reg_log1p_IC50_nM/reg_log1p_IC50_nM_model.joblib",
        "features_path": "models/regression/reg_log1p_IC50_nM/reg_log1p_IC50_nM_features.joblib",
    },
    "reg_log1p_CC50_nM": {
        "type": "regression",
        "target_col": "log1p_CC50_nM",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/regression/reg_log1p_CC50_nM/reg_log1p_CC50_nM_model.joblib",
        "features_path": "models/regression/reg_log1p_CC50_nM/reg_log1p_CC50_nM_features.joblib",
    },
    "reg_si": {
        "type": "regression",
        "target_col": "log1p_SI",
        "data_path": FINAL_DATA_PATH,
        "model_path": "models/regression/reg_si/reg_si_model.joblib",
        "features_path": "models/regression/reg_si/reg_si_features.joblib",
    },
}

def load_artifact(path):
    """Безопасная загрузка артефакта."""
    if path is None:
        return None
    
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        logger.warning(f"⚠️  Артефакт не найден: {full_path}")
        return None
    try:
        if full_path.endswith(".txt"):
            features_df = pd.read_csv(full_path)
            return features_df.iloc[:, 0].tolist()
        return joblib.load(full_path)
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки артефакта {full_path}: {e}")
        return None

def check_classification(task_name, config):
    logger.info(f"\n🔍 Проверка классификации: {task_name}")
    try:
        model = load_artifact(config["model_path"])
        features = load_artifact(config["features_path"])
        preprocessor = load_artifact(config.get("preprocessor_path"))
        threshold = load_artifact(config.get("threshold_path")) or 0.5

        if model is None or features is None:
            logger.error(f"Не удалось загрузить модель и/или признаки для {task_name}. Проверка прервана.")
            return None

        df = pd.read_csv(os.path.join(BASE_DIR, config["data_path"]))
        
        required_cols = features + [config['target_col']]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"В данных отсутствуют необходимые колонки: {missing}")
        
        X = df[features]
        y = df[config['target_col']]

        valid_indices = y.notna()
        X = X[valid_indices].reset_index(drop=True)
        y = y[valid_indices].reset_index(drop=True)
        
        if y.empty:
            logger.error(f"Для задачи {task_name} не осталось данных после удаления пропусков в целевой колонке.")
            return None

        if preprocessor:
            logger.info("Применение отдельного препроцессора...")
            X_processed = preprocessor.transform(X)
            y_proba = model.predict_proba(X_processed)[:, 1]
        else:
            logger.info("Применение внутреннего пайплайна модели...")
            y_proba = model.predict_proba(X)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        logger.info(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
        logger.info(f"F1 Score:  {f1_score(y, y_pred, zero_division=0):.4f}")
        logger.info(f"Precision: {precision_score(y, y_pred, zero_division=0):.4f}")
        logger.info(f"Recall:    {recall_score(y, y_pred, zero_division=0):.4f}")
        logger.info(f"ROC AUC:   {roc_auc_score(y, y_proba):.4f}")
        logger.info(f"Confusion Matrix (T={threshold:.2f}):\n{confusion_matrix(y, y_pred)}")

        return {
            "task": task_name, "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, zero_division=0), 
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0), 
            "roc_auc": roc_auc_score(y, y_proba),
        }

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при проверке {task_name}: {e}", exc_info=False)
        return None

def check_regression(task_name, config):
    logger.info(f"\n📈 Проверка регрессии: {task_name}")
    try:
        model = load_artifact(config["model_path"])
        features = load_artifact(config["features_path"])
        
        if model is None or features is None:
            logger.error(f"Не удалось загрузить модель и признаки для {task_name}. Проверка прервана.")
            return None

        df = pd.read_csv(os.path.join(BASE_DIR, config["data_path"]))
        
        required_cols = features + [config['target_col']]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"В данных отсутствуют необходимые колонки: {missing}")
        
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        X = df[features]
        y = df[config['target_col']]

        logger.warning(f"Препроцессор для {task_name} не найден. Используется временная импутация.")
        X_imputed = SimpleImputer(strategy="median").fit_transform(X)
        y_pred = model.predict(X_imputed)

        logger.info(f"R²:    {r2_score(y, y_pred):.4f}")
        logger.info(f"RMSE:  {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
        logger.info(f"MAE:   {mean_absolute_error(y, y_pred):.4f}")

        return {
            "task": task_name, "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
        }

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при проверке {task_name}: {e}", exc_info=False)
        return None

def print_final_report(reg_results, clf_results):
    print("\n\n================= 📊 ИТОГОВЫЙ ОТЧЕТ ПО МОДЕЛЯМ =================\n")
    
    if reg_results:
        reg_df = pd.DataFrame(reg_results)
        reg_df.to_csv(os.path.join(PLOTS_DIR, "regression_metrics.csv"), index=False)
        print("📈 Регрессия:")
        for _, row in reg_df.iterrows():
            quality = "🔴 Плохо"
            if row["r2"] > 0.6: quality = "🟢 Отлично"
            elif row["r2"] > 0.4: quality = "🟡 Умеренно"
            print(f"- {row['task']:<45} R² = {row['r2']:.3f} | RMSE = {row['rmse']:.3f} | MAE = {row['mae']:.3f} → {quality}")
    else:
        print("📈 Регрессия: ❌ метрики не найдены")

    print("\n")

    if clf_results:
        clf_df = pd.DataFrame(clf_results)
        clf_df.to_csv(os.path.join(PLOTS_DIR, "classification_metrics.csv"), index=False)
        print("✅ Классификация:")
        for _, row in clf_df.iterrows():
            quality = "🔴 Плохо"
            if row["roc_auc"] >= 0.90: quality = "🟢 Отлично"
            elif row["roc_auc"] >= 0.75: quality = "🟡 Умеренно"
            print(f"- {row['task']:<45} ROC AUC = {row['roc_auc']:.3f} | F1 = {row['f1']:.3f} | Acc = {row['accuracy']:.3f} → {quality}")
    else:
        print("✅ Классификация: ❌ метрики не найдены")
        
    print("\n==============================================================\n")

def main():
    logger.info("=== 🚀 sanity_check.py запущен ===")
    
    for fname in ["regression_metrics.csv", "classification_metrics.csv"]:
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    
    classification_results, regression_results = [], []
    
    for task, config in TASKS.items():
        if config.get("type") == "classification":
            result = check_classification(task, config)
            if result:
                classification_results.append(result)
        elif config.get("type") == "regression":
            result = check_regression(task, config)
            if result:
                regression_results.append(result)
            
    logger.info("===========================")
    logger.info("== ✅ Все задачи проверены ==")
    logger.info("===========================")
    
    print_final_report(regression_results, classification_results)

if __name__ == "__main__":
    main()