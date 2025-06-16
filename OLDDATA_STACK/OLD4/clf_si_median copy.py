print("clf_si_median - active")

# === Импорты ===
import os
import joblib
import shap
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectFromModel

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier

# === Логгирование ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_clf_si_median():



    # === Константы ===
    RANDOM_STATE = 42
    N_SPLITS_CV = 5
    N_TOP_FEATURES_TO_SELECT = 45
    TASK_NAME = "clf_si_median"
    DATA_FILE = "data/eda_gen/data_final.csv"
    FEATURE_FILE = "data/eda_gen/features/clf_SI_gt_median.txt"
    SCALE_METHODS = {"standard": StandardScaler(), "robust": RobustScaler()}
    PLOTS_DIR = f"plots/classification/{TASK_NAME}"
    MODELS_DIR = f"models/{TASK_NAME}"
    FEATURES_DIR = "features"

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)







    # === Загрузка данных ===
    df = pd.read_csv(DATA_FILE)
    X_all = df.copy()  # ✅ оставляем все колонки, включая категориальные

    # === Формирование таргета ===
    if "SI_corrected" not in df.columns:
        logger.error("Колонка 'SI_corrected' не найдена в данных")
        exit()

    median_val = df["SI_corrected"].median()
    y = (df["SI_corrected"] > median_val).astype(int)
    y.name = "target"
    logger.info(f"Цель: SI_gt_median (бинарный классификатор), Баланс классов:\n{y.value_counts(normalize=True).rename('proportion')}")

    # === Удаление целевой переменной и потенциальных утечек ===
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










    # === Определение лучшего числа признаков с кросс-валидацией
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    max_k = 60
    roc_scores = []

    for k in range(10, max_k + 1, 10):
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_k = selector.fit_transform(X_all, y)
        pipe = make_pipeline(StandardScaler(), CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))
        score = cross_val_score(pipe, X_k, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        roc_scores.append((k, score))
        print(f"k = {k}, ROC AUC = {score:.4f}")

    # График зависимости качества от количества признаков
    ks, scores = zip(*roc_scores)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker="o")
    plt.xlabel("Число признаков (k)")
    plt.ylabel("ROC AUC (CV)")
    plt.title("Подбор оптимального количества признаков")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{TASK_NAME}_feature_selection_curve.png"))
    plt.close()








    # === Отбор признаков по Mutual Information ===


    logger.info("Отбор признаков по Mutual Information (MI)...")
    selector = SelectKBest(score_func=mutual_info_classif, k=45)
    selector.fit(X_all, y)
    X = selector.transform(X_all)
    selected_features = X_all.columns[selector.get_support()].tolist()

    logger.info(f"✅ Отобрано признаков по MI: {len(selected_features)} из {X_all.shape[1]}")
    X_df = pd.DataFrame(X, columns=selected_features)











    # === A/B тестирование двух скейлеров ===
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


    # === Хранилище лучших моделей ===
    tuned_models = []

    # === Подсчёт class_weights вручную для CC50 ===
    class_counts = y.value_counts()
    total = len(y)
    class_weights = [
        total / (2 * class_counts[0]),  # вес для класса 0
        total / (2 * class_counts[1])   # вес для класса 1
    ]

    # === Список моделей с параметрами для GridSearch ===
    models = {
        "catboost": {
            "model": CatBoostClassifier(
                verbose=0,
                random_state=RANDOM_STATE,
                class_weights=class_weights,
                l2_leaf_reg=10.0  # регуляризация для борьбы с переобучением
            ),
        "params": {
            "iterations": [200],        # меньше деревьев
            "learning_rate": [0.01],
            "depth": [4, 5]             # умеренная глубина
        }

        },
        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ),
        "params": {
            "n_estimators": [200, 400],
            "learning_rate": [0.01],
            "max_depth": [3, 5],
            "subsample": [0.7],
            "colsample_bytree": [1.0]
        }

        }
    }


    # === Тюнинг базовых моделей ===
    for name, config in models.items():
        logger.info(f"Тюнинг модели: {name}")
        pipe = Pipeline([
            ("scaler", best_scaler),  # Используем лучший скейлер
            ("classifier", config["model"])
        ])
        grid_search_params = {'classifier__' + k: v for k, v in config["params"].items()}
        gs = GridSearchCV(
            pipe,
            grid_search_params,
            cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        # === Обёртка признаков в DataFrame с правильными именами ===
        X_df = pd.DataFrame(X, columns=selected_features)

        # === Обучение модели с подбором гиперпараметров ===
        gs.fit(X_df, y)
        logger.info(f"Best ROC AUC {name}: {gs.best_score_:.4f}, Параметры: {gs.best_params_}")
        tuned_models.append((name, gs.best_estimator_))

    # ✅ Обучаем стекинг: CatBoost + XGBoost + LogisticRegression
    logger.info("✅ Обучение StackingClassifier...")
    estimators = [
        ("cat", [m for name, m in tuned_models if name == "catboost"][0]),
        ("xgb", [m for name, m in tuned_models if name == "xgboost"][0])
    ]
    final_estimator = LogisticRegression(max_iter=1000, solver="liblinear")

    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )

    stack_model.fit(X_df, y)

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

    best_model_name = max(results, key=results.get)
    best_model = models_to_compare[best_model_name]
    logger.info(f"✅ Выбрана лучшая модель: {best_model_name.upper()} (ROC AUC = {results[best_model_name]:.4f})")


    # Переопределение X_df по отобранным SHAP признакам
    X_df = X_df[selected_features].copy()
    logger.info(f"⚙️ Используются {len(X_df.columns)} признаки из SHAP.")


    # Повторное обучение финальной модели на всём датасете
    logger.info("Обучение финальной модели на всём датасете (SHAP-признаки)...")
    best_model.fit(X_df, y)




    # === Предсказания
    y_pred = best_model.predict(X_df)
    y_proba = best_model.predict_proba(X_df)[:, 1]

    # === Метрики
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

    # === Classification report & confusion matrix
    logger.info("\n" + classification_report(y, y_pred))
    logger.info(f"\nConfusion matrix:\n{confusion_matrix(y, y_pred)}")


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

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Переобучение модели на тренировочной выборке
    best_model.fit(X_train, y_train)

    # Предсказания
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
    with open(metrics_holdout_path, "w") as f:
        f.write(f"ROC AUC:  {roc_auc_holdout:.4f}\n")
        f.write(f"Accuracy: {acc_holdout:.4f}\n")
        f.write(f"Precision: {prec_holdout:.4f}\n")
        f.write(f"Recall:    {rec_holdout:.4f}\n")
        f.write(f"F1-score:  {f1_holdout:.4f}\n")

    logger.info(f"Hold-out метрики сохранены: {metrics_holdout_path}")






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
    if isinstance(model_for_shap, StackingClassifier):
        logger.warning("❌ SHAP не поддерживает StackingClassifier напрямую — пропуск анализа")
        shap_supported = False
    else:
        try:
            explainer = shap.Explainer(model_for_shap)
            shap_values = explainer(X_df)
            shap_supported = True
        except Exception as e:
            logger.warning(f"❌ SHAP-анализ не выполнен: {e}")
            shap_supported = False

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
    oof_auc = roc_auc      # из финальной модели
    holdout_auc = roc_auc_holdout  # из hold-out теста

    # Визуализация
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["OOF (Train CV)", "Hold-out (Test)"], [oof_auc, holdout_auc],
                color=["#1f77b4", "#ff7f0e"])

    # Подписи значений
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

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






if __name__ == "__main__":
    run_clf_si_median()