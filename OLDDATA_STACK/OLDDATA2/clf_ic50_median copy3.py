# clf_ic50_median.py

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
TARGET_IC50_COLUMN = 'IC50_nM'
USE_FEATURE_SELECTION = True
N_TOP_FEATURES_TO_SELECT = 99
TUNE_BASE_MODELS = True

DATA_FILE = "data/eda_gen/data_final.csv"
FEATURES_FILE = "data/eda_gen/features/clf_IC50_gt_median.txt"

# --- Генерация имени задачи ---
task_identifier = TARGET_IC50_COLUMN.lower().replace("_", "")
TASK_NAME_PREFIX = f'clf_{task_identifier}_median'
TASK_NAME = f'{TASK_NAME_PREFIX}'
if USE_FEATURE_SELECTION:
    TASK_NAME += f'_mi_top{N_TOP_FEATURES_TO_SELECT}'
else:
    TASK_NAME += '_allfeatures'
if TUNE_BASE_MODELS:
    TASK_NAME += '_tuned_stack'

TASK_PLOTS_DIR = os.path.join(PLOTS_DIR, "classification", TASK_NAME)
os.makedirs(TASK_PLOTS_DIR, exist_ok=True)

# --- Вспомогательные Функции ---
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
    max_display = min(N_TOP_FEATURES_TO_SELECT if USE_FEATURE_SELECTION else 99, features_df.shape[1])
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

    # === Загрузка данных ===
    df_full = pd.read_csv(DATA_FILE)
    logger.info(f"✅ Данные загружены: {df_full.shape}")

    # === Бинаризация таргета ===
    TARGET_CC50_COLUMN = "CC50_nM"
    median_val = df_full[TARGET_CC50_COLUMN].median()
    y = (df_full[TARGET_CC50_COLUMN] > median_val).astype(int)
    y.name = "target"
    logger.info(f"🎯 Цель: {TARGET_CC50_COLUMN} > {median_val:.2f}, баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

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
    X_full = df_full.drop(columns=forbidden_cols, errors="ignore")

    # === Загрузка признаков ===
    if not os.path.exists(FEATURES_FILE):
        logger.error(f"❌ Файл признаков не найден: {FEATURES_FILE}")
        return
    with open(FEATURES_FILE, "r") as f:
        selected_features = [line.strip() for line in f if line.strip() in X_full.columns]

    selected_features = selected_features[:N_TOP_FEATURES_TO_SELECT]
    logger.info(f"✅ Используются признаки (top-{N_TOP_FEATURES_TO_SELECT}): {len(selected_features)}")
    X = X_full[selected_features].copy()

    # === MI отбор признаков (опционально) ===
    if USE_FEATURE_SELECTION:
        selector = SelectKBest(score_func=mutual_info_classif, k="all")
        selector.fit(X, y)
        mi_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        selected_features = mi_scores.head(N_TOP_FEATURES_TO_SELECT).index.tolist()
        X = X[selected_features]
        logger.info(f"📊 MI отбор: выбрано признаков: {len(selected_features)}")


    # === Загрузка признаков из файла ===
    if not os.path.exists(FEATURES_FILE):
        logger.error(f"Файл признаков не найден: {feature_list_file}")
        return

    with open(FEATURES_FILE, "r") as f:
        selected_features = [line.strip() for line in f if line.strip() in X_full.columns]

    # ✂️ Ограничение числа признаков (если нужно)
    selected_features = selected_features[:N_TOP_FEATURES_TO_SELECT]
    logger.info(f"✅ Используются признаки (top-{N_TOP_FEATURES_TO_SELECT}): {selected_features[:5]} ...")

    # === Проверка на пустой список признаков ===
    if not selected_features:
        logger.error("❌ Ни один признак не был выбран. Завершение.")
        return


    X = X_full[selected_features].copy()
    feature_names_for_model = selected_features
    logger.info(f"Используются признаки ({len(feature_names_for_model)}): {feature_names_for_model[:5]} ...")

    if X.empty:
        logger.error("Набор признаков X пуст.")
        return


    # === Препроцессинг и модели ===
    preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])

    # === Модели и гиперпараметры ===
    catboost_grid = {'iterations': [200, 400, 600], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.03, 0.05, 0.1]}
    xgboost_grid = {'n_estimators': [200, 400, 600], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05, 0.1], 'subsample': [0.7, 0.9, 1.0], 'colsample_bytree': [0.7, 0.9, 1.0]}

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

        y_proba_oof_single = cross_val_predict(best_estimator_pipeline, X, y,
                                               cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE + 4),
                                               method='predict_proba', n_jobs=-1)
        if y_proba_oof_single.ndim == 2:
            y_proba_oof_single = y_proba_oof_single[:, 1]
        oof_roc_auc_single = roc_auc_score(y, y_proba_oof_single)
        best_single_models_oof_scores[name] = oof_roc_auc_single
        logger.info(f"OOF ROC AUC для лучшей модели {name}: {oof_roc_auc_single:.4f}")
        plot_roc_curve(y, y_proba_oof_single, task_name=TASK_NAME, model_name=f"OOF_Best_{name}")

    # === Стекинг ===
    meta_learner = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced')
    stacking_clf = StackingClassifier(
        estimators=tuned_processed_base_learners,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=max(2, N_SPLITS_CV - 1), shuffle=True, random_state=RANDOM_STATE + 1),
        stack_method='predict_proba', n_jobs=-1, passthrough=False)

    params_stacking = {'final_estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
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
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'CM (Stacking OOF, T={optimal_thr_stacking:.2f})\n{TASK_NAME}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.savefig(os.path.join(TASK_PLOTS_DIR, "confusion_matrix_oof_stacking.png"), bbox_inches='tight')
    plt.show()
    logger.info(f"CM (Stacking OOF) сохранена.")
    plot_roc_curve(y, y_proba_oof_stacking, task_name=TASK_NAME, model_name="OOF_Stacking")

    logger.info("--- Сводка OOF ROC AUC ---")
    for model_name_key, score in best_single_models_oof_scores.items():
        logger.info(f"Лучшая одиночная модель {model_name_key}: {score:.4f}")
    logger.info(f"Stacking Classifier: {oof_roc_auc_stacking:.4f}")

    final_trained_model = best_stacking_model

    logger.info("SHAP-анализ для базовых моделей стекера...")

    try:
        shap_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', RobustScaler())])
        shap_preprocessor.fit(X, y)
        X_processed = shap_preprocessor.transform(X)
        X_df_shap = pd.DataFrame(X_processed, columns=feature_names_for_model)

        if not hasattr(final_trained_model, 'estimators_') or not final_trained_model.estimators_:
            logger.warning("❌ Stacking model has no base estimators. SHAP анализ невозможен.")
        else:
            for i, base_pipeline in enumerate(final_trained_model.estimators_):
                base_name = base_learners_config[i][0]
                try:
                    base_model = base_pipeline.named_steps['classifier']
                    logger.info(f"🎯 SHAP для базовой модели: {base_name} ({type(base_model).__name__})")

                    if isinstance(base_model, (CatBoostClassifier, XGBClassifier)):
                        explainer = shap.TreeExplainer(base_model)
                        shap_values = explainer.shap_values(X_df_shap)

                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]

                        mean_abs_shap = np.abs(shap_values).mean()
                        logger.info(f"🔎 Среднее |SHAP| значение: {mean_abs_shap:.5f}")

                        if mean_abs_shap == 0 or np.allclose(shap_values, 0):
                            logger.warning(f"❌ SHAP значения для {base_name} нулевые — графики не построены.")
                            continue

                        # Bar plot
                        shap.summary_plot(shap_values, X_df_shap,
                                        plot_type="bar", show=False, max_display=min(len(feature_names_for_model), 99))
                        bar_path = os.path.join(TASK_PLOTS_DIR, f"shap_bar_{base_name}.png")
                        plt.savefig(bar_path, bbox_inches='tight')
                        plt.show()
                        logger.info(f"✅ SHAP bar plot сохранен: {bar_path}")

                        # Beeswarm plot
                        shap.summary_plot(shap_values, X_df_shap,
                                        plot_type="violin", show=False, max_display=min(len(feature_names_for_model), 99))
                        swarm_path = os.path.join(TASK_PLOTS_DIR, f"shap_beeswarm_{base_name}.png")
                        plt.savefig(swarm_path, bbox_inches='tight')
                        plt.show()
                        logger.info(f"✅ SHAP beeswarm plot сохранен: {swarm_path}")

                    else:
                        logger.warning(f"❌ SHAP не поддерживает тип {type(base_model).__name__} для модели {base_name}")

                except Exception as e:
                    logger.error(f"❌ Ошибка при SHAP-анализе {base_name}: {e}")

    except Exception as e:
        logger.error(f"❌ Общая ошибка SHAP: {e}")



    # === Hold-out валидация ===
    logger.info("🔁 Hold-out проверка (20%)...")
    from sklearn.model_selection import train_test_split

    X_hold_train, X_hold_test, y_hold_train, y_hold_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE + 123
    )

    final_trained_model.fit(X_hold_train, y_hold_train)
    y_hold_proba = final_trained_model.predict_proba(X_hold_test)[:, 1]
    y_hold_pred = (y_hold_proba >= optimal_thr_stacking).astype(int)

    auc_hold = roc_auc_score(y_hold_test, y_hold_proba)
    acc_hold = (y_hold_pred == y_hold_test).mean()
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision_hold = precision_score(y_hold_test, y_hold_pred, zero_division=0)
    recall_hold = recall_score(y_hold_test, y_hold_pred, zero_division=0)
    f1_hold = f1_score(y_hold_test, y_hold_pred, zero_division=0)

    logger.info(f"🎯 Hold-out метрики (20%):")
    logger.info(f"ROC AUC:  {auc_hold:.4f}")
    logger.info(f"Accuracy: {acc_hold:.4f}")
    logger.info(f"Precision:{precision_hold:.4f}")
    logger.info(f"Recall:   {recall_hold:.4f}")
    logger.info(f"F1-score: {f1_hold:.4f}")

    # Confusion Matrix — Hold-out
    cm_hold = confusion_matrix(y_hold_test, y_hold_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_hold, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title(f'CM (Hold-out, T={optimal_thr_stacking:.2f})\n{TASK_NAME}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(TASK_PLOTS_DIR, "confusion_matrix_holdout.png"))
    plt.show()
    logger.info("CM (Hold-out) сохранена.")

    # Сохранение holdout-метрик
    holdout_metrics_path = os.path.join(TASK_PLOTS_DIR, "holdout_metrics.txt")
    with open(holdout_metrics_path, "w") as f:
        f.write(f"ROC AUC:  {auc_hold:.4f}\n")
        f.write(f"Accuracy: {acc_hold:.4f}\n")
        f.write(f"Precision:{precision_hold:.4f}\n")
        f.write(f"Recall:   {recall_hold:.4f}\n")
        f.write(f"F1-score: {f1_hold:.4f}\n")
    logger.info(f"📄 Hold-out метрики сохранены в: {holdout_metrics_path}")






    artifacts = {
        'model': final_trained_model,
        'features': selected_features
,
        'optimal_threshold_oof': optimal_thr_stacking,
        'preprocessor': shap_preprocessor
    }
    save_model_artifacts(artifacts, TASK_NAME, "classification")
    logger.info(f"Артефакты сохранены. Задача {TASK_NAME} завершена.")

if __name__ == '__main__':
    main()