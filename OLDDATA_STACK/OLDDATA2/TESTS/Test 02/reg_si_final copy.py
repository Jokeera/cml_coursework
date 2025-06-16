# === REG_SI_FINAL.PY — ПОЛНЫЙ ПАЙПЛАЙН ДЛЯ РЕГРЕССИИ ЗАДАЧИ log1p(SI) ===

import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

# === Параметры ===
RANDOM_STATE = 42
TARGET = "log1p_SI"
TASK_NAME = f"reg_{TARGET}"
DATA_FILE = "data/data_final.csv"
X_FILE = "data/scaled/X_scaled.csv"
FEATURES_OUT = f"data/features/selected_by_catboost_{TARGET}2.csv"
PLOTS_DIR = f"plots/shap/{TASK_NAME}"
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Шаг 1: Загрузка данных ===
df = pd.read_csv(DATA_FILE)
X = pd.read_csv(X_FILE)
y = df[TARGET]

# === Шаг 2: Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# === Шаг 3: Обучение модели CatBoost ===
model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.01,
    depth=6,
    eval_metric='R2',
    random_state=RANDOM_STATE,
    verbose=0
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# === Шаг 4: Оценка качества модели ===
y_pred = model.predict(X)
print(f"\n✅ R2:   {r2_score(y, y_pred):.4f}")
print(f"✅ RMSE: {mean_squared_error(y, y_pred):.4f}")
print(f"✅ MAE:  {mean_absolute_error(y, y_pred):.4f}")

# === Шаг 5: Метрики по эпохам ===
evals_result = model.get_evals_result()
metric_name = list(evals_result["learn"].keys())[0]

plt.figure(figsize=(8, 5))
plt.plot(np.log1p(evals_result["learn"][metric_name]), label="Train (log loss)")
plt.plot(np.log1p(evals_result["validation"][metric_name]), label="Test (log loss)")
plt.title(f"Log Loss по эпохам ({TASK_NAME})")
plt.xlabel("Итерации"); plt.ylabel("log(1 + loss)")
plt.grid(); plt.legend(); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/log_loss_curve.png")
plt.close()

train_r2, test_r2 = [], []
for i in range(1, model.tree_count_ + 1):
    train_r2.append(r2_score(y_train, model.predict(X_train, ntree_end=i)))
    test_r2.append(r2_score(y_test, model.predict(X_test, ntree_end=i)))

plt.figure(figsize=(8, 5))
plt.plot(train_r2, label="Train R²")
plt.plot(test_r2, label="Test R²")
plt.title(f"R² по эпохам ({TASK_NAME})")
plt.xlabel("Итерации"); plt.ylabel("R²")
plt.grid(); plt.legend(); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/r2_curve.png")
plt.close()

# === Шаг 6: SHAP-анализ ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/shap_bar_top20.png")
plt.close()

shap.summary_plot(shap_values, X, max_display=20, show=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/shap_beeswarm_top20.png")
plt.close()

# SHAP dependence plots (top-3)
top_features = X.columns[np.argsort(np.abs(shap_values).mean(0))[-3:]]
for feature in top_features:
    shap.dependence_plot(feature, shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_dependence_{feature}.png")
    plt.close()

# === Шаг 7: Отбор признаков ===
sfm = SelectFromModel(model, prefit=True, threshold="median")
selected_features = X.columns[sfm.get_support()]
df_selected = pd.DataFrame({"feature": selected_features})
df_selected.to_csv(FEATURES_OUT, index=False)
print(f"\n📂 Сохранено: {FEATURES_OUT} ({len(selected_features)} признаков)")

# === Шаг 8: Сохранение модели ===
joblib.dump(model, f"models/{TASK_NAME}.pkl")
print(f"✅ Модель сохранена: models/{TASK_NAME}.pkl")
