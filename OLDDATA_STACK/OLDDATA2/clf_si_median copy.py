# clf_si_median copy.py
print("clf_si_median copy - olddir")

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
    load_prepared_data,
    save_model_artifacts,
    setup_logging,
    get_logger,
    plot_roc_curve,
    PLOTS_DIR,
    RANDOM_STATE,
    N_SPLITS_CV
)

setup_logging()
logger = get_logger(__name__)

# --- Конфигурация Эксперимента ---
TARGET_SI_COLUMN = 'SI_corrected'
USE_FEATURE_SELECTION = True
N_TOP_FEATURES_TO_SELECT = 45
TUNE_BASE_MODELS = True

TASK_NAME_PREFIX = f'clf_si_median'
TASK_NAME = f'{TASK_NAME_PREFIX}'
if USE_FEATURE_SELECTION:
    TASK_NAME += f'_mi_top{N_TOP_FEATURES_TO_SELECT}'
else:
    TASK_NAME += '_allfeatures'
if TUNE_BASE_MODELS:
    TASK_NAME += '_tuned_stack'

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
        plt.show()
        return
    shap.summary_plot(shap_values, features_df, plot_type=plot_type, show=False, max_display=max_display)
    plt.title(f"SHAP Summary ({plot_type}) - {TASK_NAME}{model_name_suffix}")
    plt.savefig(os.path.join(TASK_PLOTS_DIR, f"shap_{plot_type}{model_name_suffix}.png"), bbox_inches='tight')
    plt.show()
    logger.info(f"SHAP summary ({plot_type}{model_name_suffix}) сохранен.")


# --- Основная логика ---
def main():
    logger.info(f"--- Начало задачи: {TASK_NAME} ---")

    df = load_prepared_data()
    if df is None:
        return

    if TARGET_SI_COLUMN not in df.columns:
        logger.error(f"Целевая колонка '{TARGET_SI_COLUMN}' не найдена. Доступные колонки: {df.columns.tolist()}")
        return

    median_val = df[TARGET_SI_COLUMN].median()
    y = (df[TARGET_SI_COLUMN] > median_val).astype(int)
    y.name = "target"
    median_display = f"{median_val:.2f}"
    logger.info(f"Задача: '{TARGET_SI_COLUMN} > {median_display}'. Баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    df = df.drop(columns=[TARGET_SI_COLUMN], errors='ignore')
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

    removed_leakage = [col for col in forbidden_cols if col in df.columns]
    if removed_leakage:
        logger.info(f"Удаляются признаки-утечки: {removed_leakage}")
        df = df.drop(columns=removed_leakage)

    X_source = df.select_dtypes(include=np.number).copy()
    feature_names_for_model = X_source.columns.tolist()

    if USE_FEATURE_SELECTION:
        logger.info(f"Выполняется отбор топ-{N_TOP_FEATURES_TO_SELECT} признаков по Mutual Information...")
        if X_source.isnull().sum().sum() > 0:
            temp_imputer = SimpleImputer(strategy='median')
            X_source_np = temp_imputer.fit_transform(X_source)
            X_source = pd.DataFrame(X_source_np, columns=X_source.columns, index=X_source.index)
        mi_selector = SelectKBest(mutual_info_classif, k=min(N_TOP_FEATURES_TO_SELECT, X_source.shape[1]))
        mi_selector.fit(X_source, y)
        selected_indices = mi_selector.get_support(indices=True)
        feature_names_for_model = X_source.columns[selected_indices].tolist()
        X = X_source[feature_names_for_model].copy()
        logger.info(f"Отобрано признаков: {X.shape[1]} — {feature_names_for_model[:5]}...")
        print(feature_names_for_model)
    else:
        X = X_source.copy()
        logger.info(f"Используются все признаки: {X.shape[1]}.")

    if X.empty:
        logger.error("Набор признаков X пуст.")
        return

    preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])

    catboost_grid = {'iterations': [200, 400], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.05]}
    xgboost_grid = {'n_estimators': [200, 400], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05], 'subsample': [0.7, 1.0], 'colsample_bytree': [0.7, 1.0]}

    base_learners_config = [
        ('catboost', CatBoostClassifier(random_state=RANDOM_STATE, verbose=0), catboost_grid),
        ('xgboost', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'), xgboost_grid)
    ]
    tuned_processed_base_learners = []
    best_single_models_oof_scores = {}

    logger.info("--- Тюнинг базовых моделей ---")
    for name, model, param_grid in base_learners_config:
        logger.info(f"Тюнинг {name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        grid_search_params = {'classifier__' + k: v for k, v in param_grid.items()}
        gs = GridSearchCV(pipe, grid_search_params,
                          cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
                          scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X, y)
        best_estimator_pipeline = gs.best_estimator_
        logger.info(f"Лучший ROC AUC ({name}, CV): {gs.best_score_:.4f}. Параметры: {gs.best_params_}")
        tuned_processed_base_learners.append((name, best_estimator_pipeline))

        logger.info(f"Получение OOF-оценки для лучшей модели {name}...")
        y_proba_oof_single = cross_val_predict(best_estimator_pipeline, X, y,
                                               cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 4),
                                               method='predict_proba', n_jobs=-1)
        if y_proba_oof_single.ndim == 2:
            y_proba_oof_single = y_proba_oof_single[:, 1]
        oof_roc_auc_single = roc_auc_score(y, y_proba_oof_single)
        best_single_models_oof_scores[name] = oof_roc_auc_single
        logger.info(f"OOF ROC AUC для лучшей модели {name}: {oof_roc_auc_single:.4f}")
        plot_roc_curve(y, y_proba_oof_single, task_name=TASK_NAME, model_name=f"OOF_Best_{name}")

    meta_learner = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced')
    stacking_clf = StackingClassifier(
        estimators=tuned_processed_base_learners,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=max(2, N_SPLITS_CV - 1), shuffle=True, random_state=RANDOM_STATE + 1),
        stack_method='predict_proba', n_jobs=-1, passthrough=False)

    params_stacking = {'final_estimator__C': [0.01, 0.1, 1.0, 10.0]}
    logger.info(f"Тюнинг StackingClassifier... Сетка: {params_stacking}")
    gs_stacking = GridSearchCV(stacking_clf, params_stacking,
                               cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 2),
                               scoring='roc_auc', n_jobs=-1, verbose=1)
    gs_stacking.fit(X, y)
    best_stacking_model = gs_stacking.best_estimator_
    logger.info(f"Лучший ROC AUC Stacking (CV): {gs_stacking.best_score_:.4f}. Параметры: {gs_stacking.best_params_}")

    y_proba_oof_stacking = cross_val_predict(best_stacking_model, X, y,
                                             cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 3),
                                             method='predict_proba', n_jobs=-1)
    if y_proba_oof_stacking.ndim == 2:
        y_proba_oof_stacking = y_proba_oof_stacking[:, 1]

    oof_roc_auc_stacking = roc_auc_score(y, y_proba_oof_stacking)
    logger.info(f"OOF ROC AUC (Stacking): {oof_roc_auc_stacking:.4f}")
    optimal_thr_stacking = find_optimal_threshold(y, y_proba_oof_stacking)
    y_pred_oof_opt_stacking = (y_proba_oof_stacking >= optimal_thr_stacking).astype(int)
    logger.info(f"Отчет классификации (Stacking OOF, порог={optimal_thr_stacking:.4f}):\n{classification_report(y, y_pred_oof_opt_stacking, zero_division=0)}")

    cm_stacking = confusion_matrix(y, y_pred_oof_opt_stacking)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'CM (Stacking OOF, T={optimal_thr_stacking:.2f})\n{TASK_NAME}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(TASK_PLOTS_DIR, "confusion_matrix_oof_stacking.png"), bbox_inches='tight')
    plt.show()
    logger.info("CM (Stacking OOF) сохранена.")
    plot_roc_curve(y, y_proba_oof_stacking, task_name=TASK_NAME, model_name="OOF_Stacking")

    logger.info("--- Сводка OOF ROC AUC ---")
    for model_name_key, score in best_single_models_oof_scores.items():
        logger.info(f"Лучшая одиночная модель {model_name_key}: {score:.4f}")
    logger.info(f"Stacking Classifier: {oof_roc_auc_stacking:.4f}")

    final_trained_model = best_stacking_model

    logger.info("SHAP анализ для StackingClassifier (базовые модели)...")
    try:
        shap_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
        shap_preprocessor.fit(X, y)
        X_processed_shap_np = shap_preprocessor.transform(X)
        X_df_shap = pd.DataFrame(X_processed_shap_np, columns=feature_names_for_model)

        if hasattr(final_trained_model, 'estimators_') and final_trained_model.estimators_:
            for i, base_pipeline_estimator in enumerate(final_trained_model.estimators_):
                base_est_original_name = base_learners_config[i][0]
                try:
                    actual_model = base_pipeline_estimator.named_steps['classifier']
                    if isinstance(actual_model, (CatBoostClassifier, XGBClassifier)):
                        explainer = shap.TreeExplainer(actual_model)
                        shap_values = explainer.shap_values(X_df_shap)
                        plot_shap_values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
                        plot_shap_summary_custom(plot_shap_values, X_df_shap, "bar", model_name_suffix=f"_{base_est_original_name}Base")
                        plot_shap_summary_custom(plot_shap_values, X_df_shap, "beeswarm", model_name_suffix=f"_{base_est_original_name}Base")
                except Exception as e:
                    logger.warning(f"SHAP ошибка для {base_est_original_name}: {e}")
    except Exception as e:
        logger.error(f"Ошибка SHAP: {e}")

    artifacts = {'model': final_trained_model, 'features': feature_names_for_model,
                 'optimal_threshold_oof': optimal_thr_stacking, 'preprocessor': shap_preprocessor}
    save_model_artifacts(artifacts, TASK_NAME, "classification")
    logger.info(f"Артефакты сохранены. Задача {TASK_NAME} завершена.")


if __name__ == '__main__':
    main()
