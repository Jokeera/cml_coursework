# === reg_si.py ===

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.exceptions import ConvergenceWarning
import warnings

# === Логгирование ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Константы ===
TARGET = "SI_corrected"
TASK_NAME = "reg_log1p_SI"
DATA_FILE = "data/data_prepared.csv"
PLOTS_DIR = f"plots/shap/{TASK_NAME}"
FEATURES_CSV_OUT = f"data/features/selected_by_catboost_{TARGET}.csv"
MODEL_DIR = os.path.join("models", "regression", TASK_NAME)
MODEL_OUT = os.path.join(MODEL_DIR, f"{TASK_NAME}_model.joblib")
FEATURES_OUT_JOBLIB = os.path.join(MODEL_DIR, f"{TASK_NAME}_features.joblib")
METRICS_OUT = f"data/metrics_{TASK_NAME}.csv"
PREDS_OUT = f"data/preds_{TASK_NAME}.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("data/features", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")

def main():
    logger.info("--- SHAP-анализ и сравнение моделей для задачи регрессии SI ---")

    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        logger.error(f"❌ Колонка {TARGET} отсутстует в датасете")
        return

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # === Удаление утечек ===
    forbidden = [
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_mM", "log1p_IC50_nM",
        "SI", "SI_corrected", "log_SI", "log1p_SI", 
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_mM", "log1p_CC50_nM",
        "SI_diff_check", "SI_check", "Unnamed: 0"
    ]
    X = X.drop(columns=[col for col in forbidden if col in X.columns], errors="ignore")
    logger.info(f"Удалены потенциально утечные признаки: {[col for col in forbidden if col in df.columns]}")

    # === Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === GridSearchCV для CatBoost ===
    logger.info("🔧 GridSearchCV для CatBoost...")
    param_grid = {
        "iterations": [800],
        "depth": [6],
        "learning_rate": [0.03],
    }
    grid_model = GridSearchCV(
        estimator=CatBoostRegressor(loss_function="RMSE", eval_metric="R2", verbose=0, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )
    grid_model.fit(X_train, y_train)
    best_cat_model = grid_model.best_estimator_

    # === Предсказания и метрики ===
    y_pred_cat = best_cat_model.predict(X_test)
    logger.info(f"🔍 CatBoost (тюнинг): R² = {r2_score(y_test, y_pred_cat):.4f} | "
                f"RMSE = {mean_squared_error(y_test, y_pred_cat):.4f} | "
                f"MAE = {mean_absolute_error(y_test, y_pred_cat):.4f}")

    # === Другие модели
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=800, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Linear": LinearRegression()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        logger.info(f"🔍 {name}: R² = {r2:.4f}")

    # === Стекинг
    stack = StackingRegressor(
        estimators=[
            ("cat", best_cat_model),
            ("rf", models["RandomForest"]),
            ("gb", models["GradientBoosting"]),
            ("ridge", models["Ridge"]),
        ],
        final_estimator=Ridge(random_state=42),
        passthrough=False
    )
    stack.fit(X_train, y_train)
    r2_stack = r2_score(y_test, stack.predict(X_test))
    logger.info(f"🧱 StackingRegressor: R² = {r2_stack:.4f}")

    # === Финальная модель на всем датасете
    y_pred = best_cat_model.predict(X)
    logger.info(f"📈 R² (Train+Test): {r2_score(y, y_pred):.4f}")
    logger.info(f"📉 RMSE: {mean_squared_error(y, y_pred):.4f}")
    logger.info(f"📊 MAE:  {mean_absolute_error(y, y_pred):.4f}")

    # === SHAP-анализ
    explainer = shap.TreeExplainer(best_cat_model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_bar_{TARGET}_top20.png")
    plt.close()

    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_beeswarm_{TARGET}_top20.png")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(0)
    top3 = X.columns[np.argsort(-mean_abs_shap)[:3]]
    for feature in top3:
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_dependence_{TARGET}_{feature}.png")
        plt.close()

    # === SelectFromModel
    sfm = SelectFromModel(best_cat_model, threshold="median", prefit=True)
    selected = X.columns[sfm.get_support()]
    pd.DataFrame({"feature": selected}).to_csv(FEATURES_CSV_OUT, index=False)
    logger.info(f"✅ Отобрано признаков: {len(selected)}")
    logger.info(f"📂 Сохранено: {FEATURES_CSV_OUT}")

    # === Сохранение артефактов
    joblib.dump(best_cat_model, MODEL_OUT)
    logger.info(f"📂 Модель сохранена: {MODEL_OUT}")

    joblib.dump(list(X.columns), FEATURES_OUT_JOBLIB)
    logger.info(f"📋 Список признаков сохранён: {FEATURES_OUT_JOBLIB}")

    pd.DataFrame({
        "true": y_test,
        "pred_catboost": y_pred_cat,
        "pred_stack": stack.predict(X_test)
    }).to_csv(PREDS_OUT, index=False)
    logger.info(f"📂 Предсказания сохранены: {PREDS_OUT}")

    metrics = {
        "CatBoost": r2_score(y_test, y_pred_cat),
        "Stacking": r2_score(y_test, stack.predict(X_test)),
        **{name: r2_score(y_test, model.predict(X_test)) for name, model in models.items()}
    }
    pd.DataFrame(metrics.items(), columns=["Model", "R2"]).to_csv(METRICS_OUT, index=False)
    logger.info(f"📊 Метрики сохранены: {METRICS_OUT}")


if __name__ == "__main__":
    main()
