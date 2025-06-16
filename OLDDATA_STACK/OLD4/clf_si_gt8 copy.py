# clf_si_gt8.py

import os
import joblib
import shap
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, GridSearchCV, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# === Настройки ===
RANDOM_STATE = 42
N_SPLITS_CV = 5
N_TOP_FEATURES_TO_SELECT = 45
TASK_NAME = "clf_si_gt8"
DATA_FILE = "data/eda_gen/data_final.csv"
PLOTS_DIR = f"plots/classification/{TASK_NAME}"
MODELS_DIR = f"models/{TASK_NAME}"
FEATURES_DIR = "features"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # === Загрузка данных ===
    df = pd.read_csv(DATA_FILE)

    # === Формирование таргета ===
    if "SI_corrected" not in df.columns:
        logger.error("Колонка 'SI_corrected' не найдена в данных")
        return

    y = (df["SI_corrected"] > 8).astype(int)
    y.name = "target"
    logger.info(f"🎯 Цель: SI_gt_8 (бинарный классификатор), Баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    # === Удаление потенциальных утечек ===
    forbidden_cols = [
        # CC50-related
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM", "CC50_gt_median",

        # IC50-related
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM", "IC50_gt_median",

        # SI-related
        "SI", "SI_corrected", "log_SI", "log1p_SI", "log1p_SI_corrected",
        "SI_original", "SI_diff", "SI_diff_check", "SI_check", "SI_gt_median", "SI_gt_8",

        # Other leakage-related
        "ratio_IC50_CC50", "Unnamed: 0"
    ]

    removed_cols = [col for col in forbidden_cols if col in df.columns]
    df = df.drop(columns=removed_cols)
    X_all = df.select_dtypes(include="number").copy()
    logger.info(f"✅ Удалены потенциально утечные признаки: {removed_cols}")


    # === Отбор признаков по MI ===
    selector = SelectKBest(score_func=mutual_info_classif, k=N_TOP_FEATURES_TO_SELECT)
    selector.fit(X_all, y)
    selected_features = X_all.columns[selector.get_support()].tolist()
    X = X_all[selected_features].copy()
    logger.info(f"✅ Отобрано признаков по MI: {len(selected_features)}")

    # === A/B тестирование скейлеров ===
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler()
    }

    best_scaler = None
    best_score = -np.inf
    for name, scaler in scalers.items():
        pipe = make_pipeline(scaler, CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))
        score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        logger.info(f"{name} ROC AUC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_scaler = scaler

    # === GridSearchCV моделей ===
    class_counts = y.value_counts()
    class_weights = [len(y) / (2 * class_counts[i]) for i in [0, 1]]

    models = {
        "catboost": {
            "model": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, class_weights=class_weights),
            "params": {
                "classifier__iterations": [200],
                "classifier__learning_rate": [0.01],
                "classifier__depth": [4, 5]
            }
        },
        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                scale_pos_weight=2.13  # class 0 : class 1 ≈ 603 / 283
            ),

            "params": {
                "classifier__n_estimators": [200, 400],
                "classifier__learning_rate": [0.01],
                "classifier__max_depth": [3, 5]
            }
        }
    }

    tuned_models = []
    for name, config in models.items():
        pipe = Pipeline([
            ("scaler", best_scaler),
            ("classifier", config["model"])
        ])
        gs = GridSearchCV(
            pipe, config["params"], cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        gs.fit(X, y)
        logger.info(f"{name} ROC AUC: {gs.best_score_:.4f}, Параметры: {gs.best_params_}")
        tuned_models.append((name, gs.best_estimator_))

    # === Стекинг ===
    estimators = [(n, m) for n, m in tuned_models]
    final_estimator = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1
    )
    logger.info("📊 Получение OOF-прогнозов (честная оценка модели)...")
    y_proba = cross_val_predict(
        stack_model, X, y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]
    
    # === ЭТАП 9.1: OOF-метрики и оптимальный порог ===
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    j_scores = tpr - fpr
    j_best_idx = np.argmax(j_scores)
    best_thresh = thresholds[j_best_idx]
    logger.info(f"📌 Оптимальный порог (Youden's J): {best_thresh:.4f} (J-стат: {j_scores[j_best_idx]:.4f})")

    y_pred_bin = (y_proba >= best_thresh).astype(int)
    y_true = y.values  # Явно для совместимости

    # === ROC-кривая (OOF) ===
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Stacking OOF AUC = {roc_auc_score(y, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая — OOF Stacking")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{TASK_NAME}_roc_oof.png")
    plt.close()

    # === Метрики на OOF ===
    roc_auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin)
    rec = recall_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)

    logger.info(f"ROC AUC:  {roc_auc:.4f}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision:{prec:.4f}")
    logger.info(f"Recall:   {rec:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info("\n" + classification_report(y_true, y_pred_bin))

    # === Confusion Matrix (нормализованная) ===
    cm = confusion_matrix(y_true, y_pred_bin, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='viridis', colorbar=False)
    plt.title(f"Confusion Matrix (normalized)\n{TASK_NAME}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{TASK_NAME}_confusion_matrix.png"))
    plt.close()

    # === ЭТАП 9.2: Hold-out проверка ===
    logger.info("🔁 Hold-out проверка после SHAP-отбора...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    stack_model.fit(X_train, y_train)
    y_test_proba = stack_model.predict_proba(X_test)[:, 1]
    y_test_bin = (y_test_proba >= best_thresh).astype(int)

    roc_auc_hold = roc_auc_score(y_test, y_test_proba)
    acc_hold = accuracy_score(y_test, y_test_bin)
    prec_hold = precision_score(y_test, y_test_bin)
    rec_hold = recall_score(y_test, y_test_bin)
    f1_hold = f1_score(y_test, y_test_bin)

    logger.info(f"🎯 Hold-out метрики (после SHAP):")
    logger.info(f"ROC AUC:  {roc_auc_hold:.4f}")
    logger.info(f"Accuracy: {acc_hold:.4f}")
    logger.info(f"Precision:{prec_hold:.4f}")
    logger.info(f"Recall:   {rec_hold:.4f}")
    logger.info(f"F1-score: {f1_hold:.4f}")

    # === Сохранение Hold-out метрик ===
    with open(os.path.join(MODELS_DIR, f"metrics_{TASK_NAME}_holdout.txt"), "w") as f:
        f.write(f"ROC AUC:  {roc_auc_hold:.4f}\n")
        f.write(f"Accuracy: {acc_hold:.4f}\n")
        f.write(f"Precision:{prec_hold:.4f}\n")
        f.write(f"Recall:   {rec_hold:.4f}\n")
        f.write(f"F1-score: {f1_hold:.4f}\n")


    # === SHAP-анализ
    try:
        model_for_shap = tuned_models[0][1].named_steps["classifier"]
        explainer = shap.Explainer(model_for_shap)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plt.savefig(f"{PLOTS_DIR}/shap_bar.png", bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X, show=False, plot_type="violin")
        plt.savefig(f"{PLOTS_DIR}/shap_beeswarm.png", bbox_inches="tight")
        plt.close()

        threshold = np.quantile(np.abs(shap_values.values).mean(axis=0), 0.3)
        selector = SelectFromModel(model_for_shap, threshold=threshold, prefit=True)
        X_selected = selector.transform(X)
        selected_features_final = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, columns=selected_features_final)
        logger.info(f"🔍 SHAP-признаков: {len(selected_features_final)}")
        with open(os.path.join(FEATURES_DIR, f"selected_by_shap_{TASK_NAME}.txt"), "w") as f:
            f.write("\n".join(selected_features_final))
    except Exception as e:
        logger.warning(f"❌ SHAP-анализ не выполнен: {e}")

    # Переобучение на SHAP-признаках
    stack_model.fit(X, y)  # Обнови модель перед сохранением
    joblib.dump(stack_model, os.path.join(MODELS_DIR, f"model_{TASK_NAME}.joblib"))

    # === Сохранение
    joblib.dump(stack_model, os.path.join(MODELS_DIR, f"model_{TASK_NAME}.joblib"))
    y.to_csv(os.path.join(MODELS_DIR, "target.csv"), index=False)
    with open(os.path.join(MODELS_DIR, f"metrics_{TASK_NAME}.txt"), "w") as f:
        f.write(f"ROC AUC:  {roc_auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision:{prec:.4f}\n")
        f.write(f"Recall:   {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Best threshold (Youden): {best_thresh:.4f}\n")


if __name__ == "__main__":
    main()
