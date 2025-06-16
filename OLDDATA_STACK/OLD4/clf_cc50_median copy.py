import os
import joblib
import shap
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
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

# Конфигурация Эксперимента
TARGET_CC50_COLUMN = 'CC50_nM'
USE_FEATURE_SELECTION = True
N_TOP_FEATURES_TO_SELECT = 99
TASK_NAME = "clf_cc50_median"
TUNE_BASE_MODELS = True

DATA_FILE = "data/eda_gen/data_final.csv"
FEATURES_FILE = "data/eda_gen/features/clf_CC50_gt_median.txt"  # Новый файл признаков для CC50
MODELS_DIR = f"models/{TASK_NAME}"

# Генерация имени задачи
task_identifier = TARGET_CC50_COLUMN.lower().replace("_", "")
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

# Основная логика
def main():
    logger.info(f"--- Начало задачи: {TASK_NAME} ---")

    # Загрузка данных
    df_full = pd.read_csv(DATA_FILE)
    logger.info(f"✅ Данные загружены: {df_full.shape}")

    # Бинаризация таргета
    median_val = df_full[TARGET_CC50_COLUMN].median()
    y = (df_full[TARGET_CC50_COLUMN] > median_val).astype(int)
    y.name = "target"
    logger.info(f"🎯 Цель: {TARGET_CC50_COLUMN} > {median_val:.2f}, баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    # Удаление потенциальных утечек
    forbidden_cols = [
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM", "CC50_gt_median",
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM", "IC50_gt_median",
        "SI", "SI_corrected", "log_SI", "log1p_SI", "log1p_SI_corrected",
        "SI_original", "SI_diff", "SI_diff_check", "SI_check", "SI_gt_median", "SI_gt_8",
        "ratio_IC50_CC50", "Unnamed: 0"
    ]
    X_full = df_full.drop(columns=forbidden_cols, errors="ignore")
    logger.info(f"Удалены потенциально утечные признаки: {[col for col in forbidden_cols if col in df_full.columns]}")

    # Загрузка признаков
    if not os.path.exists(FEATURES_FILE):
        logger.error(f"❌ Файл признаков не найден: {FEATURES_FILE}")
        return
    with open(FEATURES_FILE, "r") as f:
        selected_features = [line.strip() for line in f if line.strip() in X_full.columns]

    selected_features = selected_features[:N_TOP_FEATURES_TO_SELECT]
    logger.info(f"✅ Используются признаки (top-{N_TOP_FEATURES_TO_SELECT}): {len(selected_features)}")
    X = X_full[selected_features].copy()

    # Создание X_df
    X_df = X.copy()


    # === Определение лучшего числа признаков с кросс-валидацией ---
    logger.info("🔍 Поиск оптимального числа признаков...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    max_k = 99  # Максимальное количество признаков
    roc_scores = []  # Список для сохранения результатов

    # Цикл по количеству признаков
    for k in range(10, max_k + 1, 10):
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_k = selector.fit_transform(X, y)  # Применение выбора признаков
        pipe = make_pipeline(StandardScaler(), CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))  # Пайплайн
        score = cross_val_score(pipe, X_k, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()  # Кросс-валидация
        roc_scores.append((k, score))  # Сохранение результатов
        logger.info(f"k = {k}, ROC AUC = {score:.4f}")

    # Построение графика зависимости качества от количества признаков
    ks, scores = zip(*roc_scores)  # Разделение на k и roc_auc
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker="o")
    plt.xlabel("Число признаков (k)")
    plt.ylabel("ROC AUC (CV)")
    plt.title("Подбор оптимального количества признаков")
    plt.grid(True)
    plt.tight_layout()

    # Сохранение графика
    roc_curve_path = os.path.join(PLOTS_DIR, f"{TASK_NAME}_feature_selection_curve.png")
    plt.savefig(roc_curve_path)
    logger.info(f"График зависимости качества от числа признаков сохранен: {roc_curve_path}")
    plt.close()




    # MI отбор признаков
    if USE_FEATURE_SELECTION:
        selector = SelectKBest(score_func=mutual_info_classif, k="all")
        selector.fit(X, y)
        mi_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        selected_features = mi_scores.head(N_TOP_FEATURES_TO_SELECT).index.tolist()
        X = X[selected_features]
        logger.info(f"📊 MI отбор: выбрано признаков: {len(selected_features)}")

    # A/B тестирование двух скейлеров
    logger.info("🔍 A/B-тест скейлеров: StandardScaler vs RobustScaler...")

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler()
    }

    best_scaler_name = None
    best_roc_auc = -np.inf
    best_scaler = None

    for name, scaler in scalers.items():
        pipe = make_pipeline(
            scaler,
            CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
        )
        roc_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        logger.info(f"ROC AUC ({name}): {roc_auc:.4f}")
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_scaler_name = name
            best_scaler = scaler

    logger.info(f"✅ Выбран лучший скейлер: {best_scaler_name} (ROC AUC = {best_roc_auc:.4f})")

    # Подсчёт class_weights вручную для CC50
    class_counts = y.value_counts()
    total = len(y)
    class_weights = [
        total / (2 * class_counts[0]),
        total / (2 * class_counts[1])
    ]

    # Параметры для CatBoost
    param_grid_catboost = {
        "classifier__iterations": [200, 400, 600],
        "classifier__depth": [4, 6, 8],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1]
    }

    # Параметры для XGBoost
    param_grid_xgboost = {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.03, 0.05],
        "classifier__subsample": [0.7, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.9, 1.0]
    }

    # Список моделей с параметрами для GridSearch
    models = {
        "catboost": {
            "model": CatBoostClassifier(
                verbose=0,
                random_state=RANDOM_STATE,
                class_weights=class_weights,
                l2_leaf_reg=10.0  # регуляризация для борьбы с переобучением
            ),
            "params": param_grid_catboost
        },
        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ),
            "params": param_grid_xgboost
        }
    }

    tuned_models = []

    # Тюнинг базовых моделей
    for name, config in models.items():
        logger.info(f"Тюнинг модели: {name}")
        pipe = Pipeline([
            ("scaler", best_scaler),
            ("classifier", config["model"])
        ])

        # Использование параметров для GridSearchCV
        gs = GridSearchCV(
            pipe, config["params"],
            cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", n_jobs=-1, verbose=1
        )

        # Обучение модели с подбором гиперпараметров
        gs.fit(X, y)
        logger.info(f"Best ROC AUC {name}: {gs.best_score_:.4f}, Параметры: {gs.best_params_}")
        tuned_models.append((name, gs.best_estimator_))

    # Стеккинг
    # Обучение StackingClassifier
    final_estimator = LogisticRegression(max_iter=1000, solver="liblinear")
    stack_model = StackingClassifier(
        estimators=[("cat", tuned_models[0][1]), ("xgb", tuned_models[1][1])],
        final_estimator=final_estimator,
        cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )

    stack_model.fit(X, y)




    # Сравнение моделей
    logger.info("📊 Сравнение моделей (Stacking vs XGBoost vs CatBoost)...")
    models_to_compare = {
        "stack": stack_model,
        "xgboost": [m for name, m in tuned_models if name == "xgboost"][0],
        "catboost": [m for name, m in tuned_models if name == "catboost"][0],
    }

    results = {}
    for name, model in models_to_compare.items():
        model.fit(X_df, y)
        y_prob = model.predict_proba(X_df)[:, 1]
        auc = roc_auc_score(y, y_prob)
        results[name] = auc
        logger.info(f"{name} ROC AUC = {auc:.4f}")

        # Предсказания для confusion_matrix
        y_pred = model.predict(X_df)
        cm = confusion_matrix(y, y_pred)
        logger.info(f"{name} Confusion Matrix:\n{cm}")

    best_model_name = max(results, key=results.get)
    best_model = models_to_compare[best_model_name]
    logger.info(f"✅ Выбрана лучшая модель: {best_model_name.upper()} (ROC AUC = {results[best_model_name]:.4f})")


    # Повторное обучение финальной модели на всём датасете
    logger.info("Обучение финальной модели на всём датасете...")
    best_model.fit(X_df, y)

    # Предсказания
    y_pred = best_model.predict(X_df)
    y_proba = best_model.predict_proba(X_df)[:, 1]

    # Метрики
    y_pred = best_model.predict(X_df)
    y_proba = best_model.predict_proba(X_df)[:, 1]

    roc_auc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    logger.info(f"=== Финальные метрики (OOF, full data) ===")
    logger.info(f"ROC AUC:  {roc_auc:.4f}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")



    # === ROC-кривая
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Final Model")
    plt.legend()
    roc_path = os.path.join(PLOTS_DIR, f"{TASK_NAME}_OOF_Best_{best_model_name}_roc_curve.png")
    plt.savefig(roc_path)
    logger.info(f"ROC-кривая сохранена: {roc_path}")




    # === Hold-out валидация (20%)

    logger.info("Проверка модели на отложенной выборке (hold-out 20%)...")

    # Разделение на тренировочную и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Переобучение модели на тренировочной выборке
    best_model.fit(X_train, y_train)

    # Предсказания на тестовой выборке
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    # Метрики hold-out
    roc_auc_holdout = roc_auc_score(y_test, y_test_proba)
    acc_holdout = accuracy_score(y_test, y_test_pred)
    prec_holdout = precision_score(y_test, y_test_pred)
    rec_holdout = recall_score(y_test, y_test_pred)
    f1_holdout = f1_score(y_test, y_test_pred)

    logger.info(f"=== Hold-out метрики (Test 20%) ===")
    logger.info(f"ROC AUC:  {roc_auc_holdout:.4f}")
    logger.info(f"Accuracy: {acc_holdout:.4f}")
    logger.info(f"Precision: {prec_holdout:.4f}")
    logger.info(f"Recall:    {rec_holdout:.4f}")
    logger.info(f"F1-score:  {f1_holdout:.4f}")

    # Сохранение hold-out метрик
    metrics_holdout_path = os.path.join(MODELS_DIR, f"metrics_{best_model_name}_holdout.txt")
    os.makedirs(MODELS_DIR, exist_ok=True)  # Создание директории для моделей, если она не существует
    with open(metrics_holdout_path, "w") as f:
        f.write(f"ROC AUC:  {roc_auc_holdout:.4f}\n")
        f.write(f"Accuracy: {acc_holdout:.4f}\n")
        f.write(f"Precision: {prec_holdout:.4f}\n")
        f.write(f"Recall:    {rec_holdout:.4f}\n")
        f.write(f"F1-score:  {f1_holdout:.4f}\n")

    logger.info(f"Hold-out метрики сохранены: {metrics_holdout_path}")

    # Сравнение с базовыми моделями на hold-out
    logger.info("📊 Сравнение с базовыми моделями на hold-out...")

    base_models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear")
    }

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, preds)
        acc = accuracy_score(y_test, preds > 0.5)
        logger.info(f"{name} ROC AUC: {roc:.4f}, Accuracy: {acc:.4f}")

    # === SHAP-анализ
    logger.info("SHAP-анализ...")

    model_for_shap = best_model.named_steps["classifier"]
    shap_supported = False  # Изначально установка на False
    if isinstance(model_for_shap, StackingClassifier):
        logger.warning("❌ SHAP не поддерживает StackingClassifier напрямую — пропуск анализа")
    else:
        try:
            explainer = shap.Explainer(model_for_shap)
            shap_values = explainer(X_df)
            shap_supported = True
        except Exception as e:
            logger.warning(f"❌ SHAP-анализ не выполнен: {e}")

    if shap_supported:
        # SHAP summary (bar)
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
        bar_path = os.path.join(PLOTS_DIR, f"{TASK_NAME}_SHAP_bar.png")
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()

        # SHAP summary (beeswarm)
        shap.summary_plot(shap_values, X_df, plot_type="violin", show=False)
        bee_path = os.path.join(PLOTS_DIR, f"{TASK_NAME}_SHAP_beeswarm.png")
        plt.savefig(bee_path, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP-графики сохранены: {bar_path}, {bee_path}")

        # === Отбор признаков по SHAP
        threshold_val = np.quantile(np.abs(shap_values.values).mean(axis=0), 0.30)
        selector = SelectFromModel(model_for_shap, threshold=threshold_val, prefit=True)
        X_selected = selector.transform(X_df)
        selected_features = X_df.columns[selector.get_support()].tolist()

        X_df = pd.DataFrame(X_selected, columns=selected_features)
        logger.info(f"🔍 Финальное число признаков после SHAP-отбора: {len(selected_features)}")

        # ✅ Создание директории, если не существует
        FEATURES_DIR = "features"
        os.makedirs(FEATURES_DIR, exist_ok=True)

        features_path = os.path.join(FEATURES_DIR, f"selected_by_shap_{TASK_NAME}.txt")
        pd.Series(selected_features).to_csv(features_path, index=False)
        logger.info(f"Сохранено {len(selected_features)} признаков: {features_path}")

    # === Сохранение модели
    model_path = os.path.join(MODELS_DIR, f"model_{TASK_NAME}_{best_model_name}.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Модель сохранена: {model_path}")

    # === Сохранение целевой переменной
    y.to_csv(os.path.join(MODELS_DIR, "target.csv"), index=False)

    # === Сохранение метрик
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{best_model_name}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"ROC AUC:  {roc_auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
    logger.info(f"Метрики сохранены: {metrics_path}")

    # === 📈 График переобучения: OOF vs Hold-out ===

    # Значения
    oof_auc = roc_auc  # из финальной модели
    holdout_auc = roc_auc_holdout  # из hold-out теста

    # Визуализация
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["OOF (Train CV)", "Hold-out (Test)"], [oof_auc, holdout_auc],
                color=["#1f77b4", "#ff7f0e"])

    # Подписи значений
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

    # Настройки
    plt.ylim(0.0, 1.05)
    plt.ylabel("ROC AUC")
    plt.title("Train vs Hold-out ROC AUC")
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Сохранение
    plt.savefig(os.path.join(PLOTS_DIR, f"{TASK_NAME}_overfitting_check_auc.png"))
    plt.close()

    # ROC-кривая на hold-out (XGBoost)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_test, y_test_proba):.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая (Hold-out)")
    plt.legend()
    roc_path = os.path.join(PLOTS_DIR, f"{TASK_NAME}_holdout_ROC.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"✅ ROC-кривая на hold-out сохранена: {roc_path}")



if __name__ == '__main__':
    main()
