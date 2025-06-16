print("clf_si_median - active")

# === Импорты ===
import os
import joblib
import shap
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from utils import (
    # Эти функции предполагаются существующими в вашем utils.py
    # load_prepared_data,
    # save_model_artifacts,
    # setup_logging,
    get_logger,
    # plot_roc_curve,
    PLOTS_DIR,
    RANDOM_STATE,
    N_SPLITS_CV
)


# === Логгирование ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_clf_si_median():

    # === Константы ===
    # RANDOM_STATE = 42 # Используется из импорта
    # N_SPLITS_CV = 5 # Используется из импорта
    N_TOP_FEATURES_TO_SELECT = 45
    TASK_NAME = "clf_si_median"
    DATA_FILE = "data/eda_gen/data_final.csv"
    # FEATURE_FILE = "data/eda_gen/features/clf_SI_gt_median.txt" # Не используется, т.к. отбор внутри
    SCALE_METHODS = {"standard": StandardScaler(), "robust": RobustScaler()}
    TASK_PLOTS_DIR = os.path.join(PLOTS_DIR, "classification", f"{TASK_NAME}_mi_top{N_TOP_FEATURES_TO_SELECT}_tuned_stack")
    MODELS_DIR = f"models/{TASK_NAME}"
    FEATURES_DIR = "features"

    os.makedirs(TASK_PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # === Загрузка данных ===
    df = pd.read_csv(DATA_FILE)
    
    # === Формирование таргета ===
    if "SI_corrected" not in df.columns:
        logger.error("Колонка 'SI_corrected' не найдена в данных")
        return # ИЗМЕНЕНО: exit() на return

    median_val = df["SI_corrected"].median()
    y = (df["SI_corrected"] > median_val).astype(int)
    y.name = "target"
    logger.info(f"Цель: SI_gt_median (бинарный классификатор), Баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    # === Удаление целевой переменной и потенциальных утечек ===
    forbidden_cols = [
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM", "CC50_gt_median",
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM", "IC50_gt_median",
        "SI", "SI_corrected", "log_SI", "log1p_SI", "log1p_SI_corrected",
        "SI_original", "SI_diff", "SI_diff_check", "SI_check", "SI_gt_median", "SI_gt_8",
        "ratio_IC50_CC50", "Unnamed: 0"
    ]
    removed_cols = [col for col in forbidden_cols if col in df.columns]
    X_all = df.drop(columns=removed_cols)
    X_all = X_all.select_dtypes(include="number").copy()
    logger.info(f"✅ Удалены потенциально утечные признаки: {removed_cols}")

    # === Отбор признаков по Mutual Information ===
    logger.info("Отбор признаков по Mutual Information (MI)...")
    selector = SelectKBest(score_func=mutual_info_classif, k=N_TOP_FEATURES_TO_SELECT)
    selector.fit(X_all, y)
    
    # ИСПРАВЛЕНИЕ: Сразу создаем DataFrame X_df и работаем только с ним
    selected_features = X_all.columns[selector.get_support()].tolist()
    X_df = X_all[selected_features].copy()
    logger.info(f"✅ Отобрано признаков по MI: {len(selected_features)} из {X_all.shape[1]}")

    # === A/B тестирование двух скейлеров ===
    logger.info("🔍 A/B-тест скейлеров: StandardScaler vs RobustScaler...")
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scalers = {"StandardScaler": StandardScaler(), "RobustScaler": RobustScaler()}
    best_scaler_name, best_roc_auc, best_scaler = None, -np.inf, None
    for name, scaler in scalers.items():
        pipe = make_pipeline(scaler, CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))
        roc_auc = cross_val_score(pipe, X_df, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean() # ИСПРАВЛЕНО: X -> X_df
        logger.info(f"ROC AUC ({name}): {roc_auc:.4f}")
        if roc_auc > best_roc_auc:
            best_roc_auc, best_scaler_name, best_scaler = roc_auc, name, scaler
    logger.info(f"✅ Выбран лучший скейлер: {best_scaler_name} (ROC AUC = {best_roc_auc:.4f})")

    # === Тюнинг моделей ===
    class_counts = y.value_counts()
    total = len(y)
    class_weights = [total / (2 * class_counts[0]), total / (2 * class_counts[1])]

    models = {
        "catboost": {"model": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, class_weights=class_weights, l2_leaf_reg=10.0), "params": {"iterations": [200], "learning_rate": [0.01], "depth": [5]}},
        "xgboost": {"model": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE), "params": {"n_estimators": [200, 400], "learning_rate": [0.01], "max_depth": [5], "subsample": [0.7], "colsample_bytree": [1.0]}}
    }
    
    tuned_models = []
    for name, config in models.items():
        logger.info(f"Тюнинг модели: {name}")
        pipe = Pipeline([("scaler", best_scaler), ("classifier", config["model"])])
        grid_search_params = {'classifier__' + k: v for k, v in config["params"].items()}
        gs = GridSearchCV(pipe, grid_search_params, cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE), scoring="roc_auc", n_jobs=-1, verbose=0)
        gs.fit(X_df, y)
        logger.info(f"Best ROC AUC {name}: {gs.best_score_:.4f}, Параметры: {gs.best_params_}")
        tuned_models.append((name, gs.best_estimator_))

    # === Стекинг ===
    estimators = [("cat", dict(tuned_models)["catboost"]), ("xgb", dict(tuned_models)["xgboost"])]
    final_estimator = LogisticRegression(max_iter=1000, solver="liblinear")
    stack_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
    stack_model.fit(X_df, y)

    # === Сравнение и выбор лучшей модели ===
    logger.info("📊 Сравнение моделей (Stacking vs XGBoost vs CatBoost)...")
    models_to_compare = {"stack": stack_model, "xgboost": dict(tuned_models)["xgboost"], "catboost": dict(tuned_models)["catboost"]}
    results = {}
    for name, model in models_to_compare.items():
        # Переобучать здесь не обязательно, модель уже обучена
        y_prob = model.predict_proba(X_df)[:, 1]
        auc = roc_auc_score(y, y_prob)
        results[name] = auc
        logger.info(f"{name} ROC AUC = {auc:.4f}")

    best_model_name = max(results, key=results.get)
    best_model = models_to_compare[best_model_name]
    logger.info(f"✅ Выбрана лучшая модель: {best_model_name.upper()} (ROC AUC = {results[best_model_name]:.4f})")
    
    # === Hold-out валидация ===
    logger.info("Проверка модели на отложенной выборке (hold-out 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    best_model.fit(X_train, y_train)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)
    roc_auc_holdout = roc_auc_score(y_test, y_test_proba)
    logger.info(f"=== Hold-out метрики (Test 20%) ===")
    logger.info(f"ROC AUC:  {roc_auc_holdout:.4f}")
    
    # === SHAP-анализ и сохранение артефактов ===
    logger.info("SHAP-анализ...")
    # ИСПРАВЛЕНИЕ 2: Вся логика SHAP и сохранения перестроена для создания СОВМЕСТИМЫХ артефактов
    model_for_shap = best_model.named_steps.get("classifier", None) # Безопасно получаем классификатор
    
    final_features = X_df.columns.tolist() # По умолчанию - исходные MI-признаки
    model_to_save = best_model # По умолчанию - исходная лучшая модель

    if isinstance(model_for_shap, StackingClassifier):
        logger.warning("❌ SHAP не поддерживает StackingClassifier напрямую — сохраняем модель и признаки до SHAP-отбора.")
    elif model_for_shap is not None:
        try:
            X_transformed = best_model.named_steps['scaler'].fit_transform(X_df)
            explainer = shap.Explainer(model_for_shap, X_transformed)
            shap_values = explainer(X_transformed)
            
            vals = np.abs(shap_values.values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X_df.columns, vals)), columns=['feature', 'importance'])
            feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
            
            # Отбираем ВСЕ признаки, отсортированные по SHAP
            final_features = feature_importance['feature'].tolist() 
            X_final = X_df[final_features]
            logger.info(f"🔍 Финальное число признаков после SHAP-отбора: {len(final_features)}")

            # Переобучаем лучшую модель на этом финальном наборе
            logger.info(f"Переобучение лучшей модели ({best_model_name}) на {len(final_features)} финальных признаках...")
            model_to_save = clone(best_model)
            model_to_save.fit(X_final, y)

        except Exception as e:
            logger.warning(f"❌ SHAP-анализ не выполнен: {e}. Сохраняем модель и признаки до SHAP-отбора.")
    else:
        logger.warning("Не удалось извлечь классификатор для SHAP. Сохраняем модель и признаки до SHAP-отбора.")


    # --- Сохранение финальных, совместимых артефактов ---
    logger.info("💾 Сохранение артефактов...")
    # 1. Сохраняем модель (переобученную на SHAP-признаках, если удалось)
    model_filename = f"model_{TASK_NAME}_{best_model_name}.joblib".replace(f"_mi_top{N_TOP_FEATURES_TO_SELECT}", "") # Убираем лишнее из имени
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model_to_save, model_path)
    logger.info(f"Модель сохранена: {model_path}")

    # 2. Сохраняем финальный список признаков
    features_filename = f"selected_by_shap_{TASK_NAME}.txt"
    features_path = os.path.join(FEATURES_DIR, features_filename)
    pd.DataFrame(final_features, columns=['feature']).to_csv(features_path, index=False, header=True)
    logger.info(f"Сохранено {len(final_features)} признаков: {features_path}")

    logger.info(f"✅ Успешно: {os.path.basename(__file__)}")

if __name__ == "__main__":
    run_clf_si_median()