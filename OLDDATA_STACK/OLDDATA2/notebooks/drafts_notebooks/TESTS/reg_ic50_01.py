# === reg_ic50_01_edagen.py ===

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.exceptions import ConvergenceWarning
import warnings

# === Логгирование ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Константы ===
TARGET = "IC50_mM"
TASK_NAME = "reg_log1p_IC50_nM"
DATA_FILE = "data/data_prepared.csv"
PLOTS_DIR = f"plots/shap/{TASK_NAME}"
FEATURES_OUT = f"data/features/selected_by_catboost_{TASK_NAME}.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("data/features", exist_ok=True)

# === Отключение предупреждений sklearn (например, LinAlgWarning) ===
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")

def main():
    logger.info("--- SHAP-анализ и сравнение моделей для задачи регрессии ---")

    # === Загрузка данных ===
    try:
        df = pd.read_csv(DATA_FILE)
        if TARGET not in df.columns:
            raise ValueError(f"Колонка {TARGET} не найдена в {DATA_FILE}")
        y = np.log1p(df[TARGET])
        X = df.drop(columns=[TARGET])
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных: {e}")
        return

    forbidden = [
        "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM",
        "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM",
        "SI", "SI_corrected", "log_SI", "log1p_SI", "SI_diff_check"
    ]
    X = X.drop(columns=[col for col in forbidden if col in X.columns], errors="ignore")
    logger.info(f"Удалены потенциально утечные признаки: {[col for col in forbidden if col in df.columns]}")

    # === Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Обучение базовых моделей для стекинга ===
    models = {
        "CatBoost": CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, eval_metric="R2", verbose=0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42),
        "Ridge": Ridge(),
        "Linear": LinearRegression()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        logger.info(f"🔍 {name}: R² = {r2:.4f}")

    # === Стекинг ===
    stack = StackingRegressor(
        estimators=[
            ("cat", models["CatBoost"]),
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

    # === Финальная модель CatBoost ===
    final_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        eval_metric='R2',
        verbose=0,
        random_state=42
    )
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, plot=False)

    y_pred = final_model.predict(X)
    logger.info(f"📈 R²:   {r2_score(y, y_pred):.4f}")
    logger.info(f"📉 RMSE: {mean_squared_error(y, y_pred):.4f}")
    logger.info(f"📊 MAE:  {mean_absolute_error(y, y_pred):.4f}")

    # === Кривые обучения ===
    evals_result = final_model.get_evals_result()
    metric = list(evals_result['learn'].keys())[0]

    plt.figure(figsize=(8, 5))
    plt.plot(np.log1p(evals_result['learn'][metric]), label="Train (log loss)")
    plt.plot(np.log1p(evals_result['validation'][metric]), label="Test (log loss)")
    plt.title(f"Log Loss по эпохам ({TASK_NAME})")
    plt.xlabel("Итерации")
    plt.ylabel("log(1 + loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/log_loss_curve_{TASK_NAME}.png")
    plt.close()

    train_r2, test_r2 = [], []
    for i in range(1, final_model.tree_count_ + 1):
        train_r2.append(r2_score(y_train, final_model.predict(X_train, ntree_end=i)))
        test_r2.append(r2_score(y_test, final_model.predict(X_test, ntree_end=i)))

    plt.figure(figsize=(8, 5))
    plt.plot(train_r2, label="Train R²")
    plt.plot(test_r2, label="Test R²")
    plt.title(f"R² по эпохам ({TASK_NAME})")
    plt.xlabel("Итерации")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/r2_curve_{TASK_NAME}.png")
    plt.close()

    # === SHAP-анализ ===
    explainer = shap.TreeExplainer(final_model)
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
    sfm = SelectFromModel(final_model, threshold="median", prefit=True)
    selected = X.columns[sfm.get_support()]
    pd.DataFrame({"feature": selected}).to_csv(FEATURES_OUT, index=False)
    logger.info(f"✅ Отобрано признаков: {len(selected)}")
    logger.info(f"📂 Сохранено: {FEATURES_OUT}")

if __name__ == "__main__":
    main()
