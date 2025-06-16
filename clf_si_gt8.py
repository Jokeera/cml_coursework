import os
import joblib
import shap
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.base import clone
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
        "CC50", "CC50_mM", "CC50_nM", "log_CC50", "log1p_CC50", "log1p_CC50_nM", "CC50_gt_median",
        "IC50", "IC50_mM", "IC50_nM", "log_IC50", "log1p_IC50", "log1p_IC50_nM", "IC50_gt_median",
        "SI", "SI_corrected", "log_SI", "log1p_SI", "log1p_SI_corrected",
        "SI_original", "SI_diff", "SI_diff_check", "SI_check", "SI_gt_median", "SI_gt_8",
        "ratio_IC50_CC50", "Unnamed: 0"
    ]

    removed_cols = [col for col in forbidden_cols if col in df.columns]
    df = df.drop(columns=removed_cols)
    X_all = df.select_dtypes(include="number").copy()
    logger.info(f"✅ Удалены потенциально утечные признаки: {removed_cols}")


    # === Отбор признаков по MI ===
    selector = SelectKBest(score_func=mutual_info_classif, k=N_TOP_FEATURES_TO_SELECT)
    selector.fit(X_all, y)
    initial_features = X_all.columns[selector.get_support()].tolist()
    X = X_all[initial_features].copy()
    logger.info(f"✅ Отобрано признаков по MI: {len(initial_features)}")

    # === A/B тестирование скейлеров ===
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scalers = {"StandardScaler": StandardScaler(), "RobustScaler": RobustScaler()}
    best_scaler, best_score = None, -np.inf
    for name, scaler in scalers.items():
        pipe = make_pipeline(scaler, CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))
        score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        logger.info(f"{name} ROC AUC: {score:.4f}")
        if score > best_score:
            best_score, best_scaler = score, scaler

    # === GridSearchCV моделей ===
    class_counts = y.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    models = {
        "catboost": {"model": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight), "params": {"classifier__iterations": [200], "classifier__learning_rate": [0.01], "classifier__depth": [5]}},
        "xgboost": {"model": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight), "params": {"classifier__n_estimators": [200], "classifier__learning_rate": [0.01], "classifier__max_depth": [5]}}
    }
    tuned_models = []
    for name, config in models.items():
        pipe = Pipeline([("scaler", best_scaler), ("classifier", config["model"])])
        gs = GridSearchCV(pipe, config["params"], cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        logger.info(f"{name} ROC AUC: {gs.best_score_:.4f}, Параметры: {gs.best_params_}")
        tuned_models.append((name, gs.best_estimator_))

    # === Стекинг и оценка OOF ===
    estimators = [(n, m) for n, m in tuned_models]
    final_estimator = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    stack_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=cv, n_jobs=-1)
    
    logger.info("📊 Получение OOF-прогнозов (честная оценка модели)...")
    y_proba = cross_val_predict(stack_model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    
    # === Определение оптимального порога ===
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    logger.info(f"📌 Оптимальный порог (Youden's J): {best_thresh:.4f}")
    y_pred_bin = (y_proba >= best_thresh).astype(int)
    
    logger.info("\n" + classification_report(y, y_pred_bin))
    
    # === SHAP-анализ и финальный отбор признаков ===
    final_features = X.columns.tolist() # по умолчанию используем исходные MI-признаки
    try:
        model_for_shap = tuned_models[0][1] # Берем лучшую базовую модель для анализа
        X_transformed = model_for_shap.named_steps['scaler'].fit_transform(X)
        explainer = shap.Explainer(model_for_shap.named_steps['classifier'], X_transformed)
        shap_values = explainer(X_transformed)
        
        vals = np.abs(shap_values.values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'importance'])
        feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
        
        final_features = feature_importance['feature'].tolist() # Берем все признаки, отсортированные по SHAP
        logger.info(f"🔍 SHAP-признаков отсортировано: {len(final_features)}")
    except Exception as e:
        logger.warning(f"❌ SHAP-анализ не выполнен: {e}")
    
    # === Переобучение финальной модели на финальных признаках ===
    X_final = X[final_features].copy()
    logger.info(f"⚙️ Обучение финальной модели на {len(final_features)} признаках...")
    final_model_to_save = clone(stack_model) # Клонируем стек, чтобы обучить на финальных признаках
    final_model_to_save.fit(X_final, y)

    # === Hold-out проверка ===
    logger.info("🔁 Hold-out проверка...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    holdout_model = clone(final_model_to_save)
    holdout_model.fit(X_train, y_train)
    y_test_proba = holdout_model.predict_proba(X_test)[:, 1]
    y_test_bin = (y_test_proba >= best_thresh).astype(int)

    logger.info(f"🎯 Hold-out метрики:")
    logger.info(f"ROC AUC:  {roc_auc_score(y_test, y_test_proba):.4f}")
    logger.info(f"F1-score: {f1_score(y_test, y_test_bin):.4f}")

    # === Сохранение всех артефактов для sanity_check.py ===
    logger.info("💾 Сохранение артефактов...")
    
    # 1. Финальная модель
    joblib.dump(final_model_to_save, os.path.join(MODELS_DIR, "model_clf_si_gt8.joblib"))
    # 2. Финальный список признаков
    joblib.dump(final_features, os.path.join(MODELS_DIR, "features.joblib"))
    # 3. Препроцессор (скейлер)
    joblib.dump(best_scaler, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    # 4. Оптимальный порог
    joblib.dump(best_thresh, os.path.join(MODELS_DIR, "optimal_threshold.joblib"))
    
    logger.info(f"✅ Все артефакты сохранены в {MODELS_DIR}")
    
    logger.info(f"✅ Успешно: {os.path.basename(__file__)}")


if __name__ == "__main__":
    main()