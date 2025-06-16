# reg_si.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import logging
from sklearn.base import clone

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings

# === Логгирование ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Константы ===
TARGET = "log1p_SI"
TASK_NAME = "reg_si"
DATA_FILE = "data/eda_gen/data_final.csv"
PLOTS_DIR = f"plots/shap/{TASK_NAME}"
MODEL_DIR = os.path.join("models", "regression", TASK_NAME)
MODEL_OUT = os.path.join(MODEL_DIR, f"{TASK_NAME}_model.joblib")
FEATURES_OUT_JOBLIB = os.path.join(MODEL_DIR, f"{TASK_NAME}_features.joblib")
METRICS_OUT = f"data/metrics_{TASK_NAME}.csv"
PREDS_OUT = f"data/preds_{TASK_NAME}.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    np.random.seed(42)
    logger.info(f"--- SHAP-анализ и сравнение моделей для задачи регрессии SI ---")

    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        logger.error(f"❌ Колонка {TARGET} отсутстует в датасете")
        return

    y = df[TARGET]
    X_raw = df.drop(columns=[TARGET])
    
    # Заполнение пропусков
    X_imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_raw), columns=X_raw.columns)

    forbidden = [
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM", "CC50_gt_median",
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM", "IC50_gt_median",
        "SI", "SI_corrected", "log_SI", "log1p_SI", "log1p_SI_corrected",
        "SI_original", "SI_diff", "SI_diff_check", "SI_check", "SI_gt_median",  "SI_gt_8",
        "ratio_IC50_CC50", "Unnamed: 0"
    ]
    X = X_imputed.drop(columns=[col for col in forbidden if col in X_imputed.columns], errors="ignore")
    logger.info(f"Удалены потенциально утечные признаки.")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # GridSearchCV для CatBoost
    logger.info("🔧 GridSearchCV для CatBoost...")
    param_grid = {"iterations": [800], "depth": [6], "learning_rate": [0.03]}
    grid_model = GridSearchCV(
        estimator=CatBoostRegressor(loss_function="RMSE", eval_metric="R2", verbose=0, random_state=42),
        param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1
    )
    grid_model.fit(X_train, y_train)
    best_cat_model = grid_model.best_estimator_
    logger.info(f"🔍 CatBoost (тюнинг): R² = {r2_score(y_test, best_cat_model.predict(X_test)):.4f}")

    # Обучение других моделей
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=800, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        logger.info(f"🔍 {name}: R² = {r2_score(y_test, model.predict(X_test)):.4f}")

    # Сравнение и выбор лучшей модели (CatBoost vs остальные)
    all_models = {"CatBoost": best_cat_model, **models}
    best_model_name = max(all_models, key=lambda name: r2_score(y_test, all_models[name].predict(X_test)))
    best_model = all_models[best_model_name]
    logger.info(f"✅ Выбрана лучшая модель: {best_model_name} (R² на тесте = {r2_score(y_test, best_model.predict(X_test)):.4f})")

    # SHAP-анализ на лучшей модели
    logger.info(f"🔬 SHAP-анализ для модели {best_model_name}...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X) # Анализ на всех данных для глобальной важности
    
    # Визуализация SHAP
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/shap_bar_{TARGET}_top20.png"); plt.close()

    # Финальный отбор признаков на основе SHAP
    sfm = SelectFromModel(best_model, threshold="median", prefit=True)
    final_features = X.columns[sfm.get_support()].tolist()
    logger.info(f"✅ Отобрано признаков после SHAP/SelectFromModel: {len(final_features)}")
    
    # Переобучение лучшей модели на отобранных признаках
    logger.info(f"⚙️ Переобучение модели {best_model_name} на {len(final_features)} финальных признаках...")
    X_final = X[final_features]
    model_to_save = clone(best_model)
    model_to_save.fit(X_final, y)

    # Сохранение артефактов
    joblib.dump(model_to_save, MODEL_OUT)
    logger.info(f"💾 Модель сохранена: {MODEL_OUT}")

    joblib.dump(final_features, FEATURES_OUT_JOBLIB)
    logger.info(f"📋 Список признаков сохранён: {FEATURES_OUT_JOBLIB}")

    # Сохранение метрик и предсказаний для отчета
    metrics = {name: r2_score(y_test, model.predict(X_test)) for name, model in all_models.items()}
    pd.DataFrame(metrics.items(), columns=["Model", "R2_on_test"]).to_csv(METRICS_OUT, index=False)
    logger.info(f"📊 Метрики сохранены: {METRICS_OUT}")

    pd.DataFrame({"true": y_test, "pred_best_model": best_model.predict(X_test)}).to_csv(PREDS_OUT, index=False)
    logger.info(f"📂 Предсказания сохранены: {PREDS_OUT}")

if __name__ == "__main__":
    main()