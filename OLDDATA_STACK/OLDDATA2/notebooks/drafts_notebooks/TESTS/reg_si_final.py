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
TARGET = "log1p_SI"
TASK_NAME = "reg_log1p_SI"
DATA_FILE = "data/data_final.csv"
PLOTS_DIR = f"plots/shap/{TASK_NAME}"
FEATURES_OUT = f"data/features/selected_by_catboost_{TASK_NAME}.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("data/features", exist_ok=True)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")

def main():
    logger.info("--- SHAP-анализ и сравнение моделей для задачи регрессии SI ---")

    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        logger.error(f"❌ Колонка {TARGET} отсутствует в датасете")
        return

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    forbidden = [
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_mM", "log1p_IC50_nM",
        "SI", "SI_corrected", "log_SI", "log1p_SI", 
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_mM", "log1p_CC50_nM",
        "SI_diff_check", "SI_check", "Unnamed: 0", "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI", "IC50_gt_median",	"CC50_gt_median", "SI_gt_median", "SI_gt_8"
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
        cv=3,
        scoring="r2",
        n_jobs=-1
    )
    grid_model.fit(X_train, y_train)
    best_cat_model = grid_model.best_estimator_

    y_pred_cat = best_cat_model.predict(X_test)
    logger.info(f"🔍 CatBoost (тюнинг): R² = {r2_score(y_test, y_pred_cat):.4f} | RMSE = {mean_squared_error(y_test, y_pred_cat):.4f} | MAE = {mean_absolute_error(y_test, y_pred_cat):.4f}")

    # === Обучение других моделей ===
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=800, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Linear": LinearRegression()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        logger.info(f"🔍 {name}: R² = {r2:.4f}")

    # === Стекинг ===
    stack = StackingRegressor(
        estimators=[
            ("cat", best_cat_model),
            ("rf", models["RandomForest"]),
            ("gb", models["GradientBoosting"]),
            ("ridge", models["Ridge"]),
        ],
        final_estimator=Ridge(),
        passthrough=False
    )
    stack.fit(X_train, y_train)
    r2_stack = r2_score(y_test, stack.predict(X_test))
    logger.info(f"🧱 StackingRegressor: R² = {r2_stack:.4f}")

    # === Финальная модель на всём датасете ===
    y_pred = best_cat_model.predict(X)
    logger.info(f"📈 R²:   {r2_score(y, y_pred):.4f}")
    logger.info(f"📉 RMSE: {mean_squared_error(y, y_pred):.4f}")
    logger.info(f"📊 MAE:  {mean_absolute_error(y, y_pred):.4f}")

    # === SHAP-анализ ===
    explainer = shap.TreeExplainer(best_cat_model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_bar_{TASK_NAME}_top20.png")
    plt.close()

    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_beeswarm_{TASK_NAME}_top20.png")
    plt.close()

    top3 = X.columns[np.argsort(np.abs(shap_values).mean(0))[-3:]]
    for feature in top3:
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_dependence_{TASK_NAME}_{feature}.png")
        plt.close()

    # === Отбор признаков SelectFromModel ===
    sfm = SelectFromModel(best_cat_model, threshold="median", prefit=True)
    selected = X.columns[sfm.get_support()]
    pd.DataFrame({"feature": selected}).to_csv(FEATURES_OUT, index=False)
    logger.info(f"✅ Отобрано признаков: {len(selected)}")
    logger.info(f"📂 Сохранено: {FEATURES_OUT}")

if __name__ == "__main__":
    main()