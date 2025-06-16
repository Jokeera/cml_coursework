# clf_si_median copy 2.py
print("clf_si_median copy 2 - newdir_oldfeatures")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from utils import (
    save_model_artifacts,
    setup_logging,
    get_logger,
    plot_roc_curve,
    PLOTS_DIR,
    RANDOM_STATE,
    N_SPLITS_CV
)

# === Инициализация ===
setup_logging()
logger = get_logger(__name__)

# --- Конфигурация ---
TASK_NAME_PREFIX = 'clf_si_median'
USE_FEATURE_SELECTION = True
N_TOP_FEATURES_TO_SELECT = 20
TUNE_BASE_MODELS = True

TASK_NAME = f"{TASK_NAME_PREFIX}"
if USE_FEATURE_SELECTION:
    TASK_NAME += f"_mi_top{N_TOP_FEATURES_TO_SELECT}"
else:
    TASK_NAME += "_allfeatures"
if TUNE_BASE_MODELS:
    TASK_NAME += "_tuned_stack"

TASK_PLOTS_DIR = os.path.join(PLOTS_DIR, "classification", TASK_NAME)
os.makedirs(TASK_PLOTS_DIR, exist_ok=True)


def find_optimal_threshold(y_true, y_proba_oof):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_oof)
    if not (tpr.size and fpr.size and thresholds.size):
        logger.warning("Недостаточно данных или классов для ROC-кривой. Порог 0.5.")
        return 0.5
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]
    logger.info(f"Оптимальный порог (Youden's J): {optimal_threshold:.4f} (J-стат: {j_scores[optimal_idx]:.4f})")
    return optimal_threshold


def plot_shap_summary_custom(shap_values, features_df, plot_type="bar", model_name_suffix=""):
    plt.figure()
    max_display = min(N_TOP_FEATURES_TO_SELECT if USE_FEATURE_SELECTION else 20, features_df.shape[1])
    if max_display <= 0:
        logger.warning(f"Нет признаков для отображения в SHAP plot ({model_name_suffix}).")
        plt.close()
        return
    shap.summary_plot(shap_values, features_df, plot_type=plot_type, show=False, max_display=max_display)
    plt.title(f"SHAP Summary ({plot_type}) - {TASK_NAME}{model_name_suffix}")
    plt.savefig(os.path.join(TASK_PLOTS_DIR, f"shap_{plot_type}{model_name_suffix}.png"), bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP summary ({plot_type}{model_name_suffix}) сохранен.")


def main():
    logger.info(f"--- Задача: {TASK_NAME} ---")

    # === Загрузка данных ===
    X = pd.read_csv("data/eda_gen/scaled/data_scaled.csv")
    df = pd.read_csv("data/eda_gen/data_final.csv")

    # === Целевой вектор ===
    if "SI_gt_median" not in df.columns:
        logger.error("Целевая колонка 'SI_gt_median' не найдена в data_final.csv")
        return

    median_val = df["SI_gt_median"].median()
    y = (df["SI_gt_median"] > median_val).astype(int)
    y.name = "target"
    logger.info(f"Цель: SI_gt_median > {median_val:.2f}, Баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    # === Удаление признаков-утечек ===
    forbidden_cols = [
         # CC50-related
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM", "CC50_gt_median",

        # IC50-related
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM", "IC50_gt_median",

        # SI-related
        "SI", "SI_corrected", "log_SI", "log1p_SI", "log1p_SI_corrected",
        "SI_original", "SI_diff", "SI_diff_check", "SI_check", "SI_gt_median",	"SI_gt_8",

        # Other leakage-related
        "ratio_IC50_CC50", "Unnamed: 0"
    ]
    X = X.drop(columns=[col for col in forbidden_cols if col in X.columns], errors="ignore")
    feature_names_for_model = X.columns.tolist()

    # === Отбор признаков по MI ===
    if USE_FEATURE_SELECTION:
        logger.info(f"Отбор top-{N_TOP_FEATURES_TO_SELECT} признаков по Mutual Information...")
        if X.isnull().sum().sum() > 0:
            X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X.columns, index=X.index)
        mi_selector = SelectKBest(mutual_info_classif, k=min(N_TOP_FEATURES_TO_SELECT, X.shape[1]))
        mi_selector.fit(X, y)
        selected_indices = mi_selector.get_support(indices=True)
        feature_names_for_model = X.columns[selected_indices].tolist()
        X = X[feature_names_for_model]
        logger.info(f"Отобрано признаков: {X.shape[1]} — {feature_names_for_model[:5]}...")
    else:
        logger.info(f"Используются все признаки: {X.shape[1]}.")

    if X.empty:
        logger.error("Набор признаков X пуст.")
        return

    # === Предобработка ===
    preprocessor = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])

    # === Модели и гиперпараметры ===
    catboost_grid = {'iterations': [200, 400], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.05]}
    xgboost_grid = {'n_estimators': [200, 400], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05], 'subsample': [0.7, 1.0], 'colsample_bytree': [0.7, 1.0]}

    base_learners_config = [
        ('catboost', CatBoostClassifier(random_state=RANDOM_STATE, verbose=0), catboost_grid),
        ('xgboost', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'), xgboost_grid)
    ]

    tuned_models = []
    best_oof_scores = {}

    for name, model, grid in base_learners_config:
        logger.info(f"Тюнинг модели: {name}")
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        param_grid = {f'classifier__{k}': v for k, v in grid.items()}
        gs = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
                          scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X, y)
        logger.info(f"Best ROC AUC {name}: {gs.best_score_:.4f}, Параметры: {gs.best_params_}")
        tuned_models.append((name, gs.best_estimator_))

        # OOF
        y_proba_oof = cross_val_predict(gs.best_estimator_, X, y, method="predict_proba",
                                        cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 10),
                                        n_jobs=-1)[:, 1]
        best_oof_scores[name] = roc_auc_score(y, y_proba_oof)
        logger.info(f"OOF ROC AUC {name}: {best_oof_scores[name]:.4f}")
        plot_roc_curve(y, y_proba_oof, task_name=TASK_NAME, model_name=f"OOF_Best_{name}")

    # === Stacking ===
    stacking_model = StackingClassifier(
        estimators=tuned_models,
        final_estimator=LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE),
        cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 1),
        stack_method="predict_proba",
        n_jobs=-1
    )

    logger.info("Тюнинг StackingClassifier...")
    gs_stacking = GridSearchCV(stacking_model, {'final_estimator__C': [0.01, 0.1, 1.0, 10.0]},
                               cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 2),
                               scoring='roc_auc', n_jobs=-1, verbose=1)
    gs_stacking.fit(X, y)
    best_stacking_model = gs_stacking.best_estimator_

    y_proba_oof_stack = cross_val_predict(best_stacking_model, X, y, method="predict_proba",
                                          cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 3),
                                          n_jobs=-1)[:, 1]
    auc_stack = roc_auc_score(y, y_proba_oof_stack)
    logger.info(f"OOF ROC AUC (Stacking): {auc_stack:.4f}")
    threshold = find_optimal_threshold(y, y_proba_oof_stack)
    y_pred_opt = (y_proba_oof_stack >= threshold).astype(int)
    logger.info(f"Classification Report:\n{classification_report(y, y_pred_opt, zero_division=0)}")
    plot_roc_curve(y, y_proba_oof_stack, task_name=TASK_NAME, model_name="OOF_Stacking")

    # === CM ===
    cm = confusion_matrix(y, y_pred_opt)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold={threshold:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(TASK_PLOTS_DIR, "confusion_matrix_oof_stacking.png"), bbox_inches='tight')
    plt.close()

    # === SHAP ===
    try:
        shap_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
        X_shap = pd.DataFrame(shap_pipeline.fit_transform(X), columns=feature_names_for_model)
        for i, base in enumerate(best_stacking_model.estimators_):
            model_name = base_learners_config[i][0]
            try:
                model = base.named_steps['classifier']
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                if isinstance(shap_values, list):  # For CatBoost
                    shap_values = shap_values[1]
                plot_shap_summary_custom(shap_values, X_shap, "bar", model_name_suffix=f"_{model_name}")
                plot_shap_summary_custom(shap_values, X_shap, "beeswarm", model_name_suffix=f"_{model_name}")
            except Exception as e:
                logger.warning(f"Ошибка SHAP для {model_name}: {e}")
    except Exception as e:
        logger.error(f"Ошибка SHAP-анализов: {e}")

    # === Сохранение артефактов ===
    artifacts = {
        'model': best_stacking_model,
        'features': feature_names_for_model,
        'optimal_threshold_oof': threshold,
        'preprocessor': shap_pipeline
    }
    save_model_artifacts(artifacts, TASK_NAME, "classification")
    logger.info(f"Задача {TASK_NAME} завершена. Артефакты сохранены.")


if __name__ == "__main__":
    main()
