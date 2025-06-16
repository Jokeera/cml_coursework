# === reg_cc50.py ===

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

# === Логгирование ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# === Константы ===
TARGET = "log1p_CC50_nM"
TASK_NAME = "reg_log1p_CC50_nM"
DATA_FILE = "data/data_final_reg.csv"
X_FILE = "data/scaled/X_scaled_reg.csv"
PLOTS_DIR = f"plots/shap/{TASK_NAME}"
FEATURES_OUT = f"data/features/selected_by_catboost_{TASK_NAME}.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("data/features", exist_ok=True)

def main():
    logger.info("--- SHAP-анализ и сравнение моделей для задачи регрессии ---")

    # === Загрузка данных ===
    try:
        X = pd.read_csv(X_FILE)
        y = pd.read_csv(DATA_FILE)[TARGET]
    except FileNotFoundError:
        logger.error("❌ Не найден файл X_scaled_reg.csv или data_final_reg.csv")
        return

    # === Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Сравнение моделей ===
    models = {
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(),
        "Linear": LinearRegression(),
    }

    param_grids = {
        "CatBoost": {"depth": [4, 6], "learning_rate": [0.03, 0.05], "iterations": [800, 1000]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "Ridge": {"alpha": [0.1, 1.0]},
        "Linear": {},
    }

    best_models = {}
    best_scores = {}

    for name, model in models.items():
        try:
            if param_grids[name]:
                grid = GridSearchCV(model, param_grids[name], cv=3, scoring="r2", n_jobs=-1)
                grid.fit(X_train, y_train)
                best_models[name] = grid.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_models[name] = model
            preds = best_models[name].predict(X_test)
            score = r2_score(y_test, preds)
            best_scores[name] = score
            logger.info(f"🔍 {name}: R² = {score:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка в модели {name}: {e}")
            continue

    # === Стекинг моделей ===
    stack_model = StackingRegressor(
        estimators=[
            ('cat', best_models['CatBoost']),
            ('rf', best_models['RandomForest']),
            ('gb', best_models['GradientBoosting']),
        ],
        final_estimator=Ridge(),
        cv=10,
        n_jobs=-1
    )
    stack_model.fit(X_train, y_train)
    stack_preds = stack_model.predict(X_test)
    stack_r2 = r2_score(y_test, stack_preds)
    logger.info(f"🧱 StackingRegressor: R² = {stack_r2:.4f}")

    # === Финальная модель CatBoost ===
    model = best_models["CatBoost"]
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, plot=False)

    evals_result = model.get_evals_result()
    metric_name = list(evals_result["learn"].keys())[0]

    plt.figure(figsize=(8, 5))
    plt.plot(np.log1p(evals_result["learn"][metric_name]), label="Train (log loss)")
    plt.plot(np.log1p(evals_result["validation"][metric_name]), label="Test (log loss)")
    plt.title(f"Log Loss по эпохам ({TASK_NAME})")
    plt.xlabel("Итерации")
    plt.ylabel("log(1 + loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/log_loss_curve_{TASK_NAME}.png")
    plt.close()

    train_r2, test_r2 = [], []
    for i in range(1, model.tree_count_ + 1):
        y_train_pred = model.predict(X_train, ntree_end=i)
        y_test_pred = model.predict(X_test, ntree_end=i)
        train_r2.append(r2_score(y_train, y_train_pred))
        test_r2.append(r2_score(y_test, y_test_pred))

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

    # === Метрики на всём X ===
    y_pred = model.predict(X)
    logger.info(f"📈 R²:   {r2_score(y, y_pred):.4f}")
    logger.info(f"📉 RMSE: {mean_squared_error(y, y_pred):.4f}")
    logger.info(f"📊 MAE:  {mean_absolute_error(y, y_pred):.4f}")

    # === SHAP-анализ ===
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_bar_{TASK_NAME}_top20.png")
    plt.close()

    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_beeswarm_{TASK_NAME}_top20.png")
    plt.close()

    top_features = X.columns[np.argsort(np.abs(shap_values).mean(0))[-3:]]
    for feature in top_features:
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_dependence_{TASK_NAME}_{feature}.png")
        plt.close()

    # === Отбор признаков ===
    sfm = SelectFromModel(model, prefit=True, threshold="median")
    selected_features = X.columns[sfm.get_support()]
    df_selected = pd.DataFrame({"feature": selected_features})
    df_selected.to_csv(FEATURES_OUT, index=False)
    logger.info(f"✅ Отобрано признаков: {len(selected_features)}")
    logger.info(f"📂 Сохранено: {FEATURES_OUT}")

if __name__ == "__main__":
    main()
