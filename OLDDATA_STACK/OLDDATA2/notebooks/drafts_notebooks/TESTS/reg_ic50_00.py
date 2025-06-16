import os
import logging # Логгирование остается, но настраивается через utils
import joblib # Будет использоваться через utils.save_model_artifacts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV # ДОБАВЛЕНО: RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    roc_curve
)
from sklearn.pipeline import Pipeline # ДОБАВЛЕНО: Pipeline
from sklearn.linear_model import LogisticRegression # ДОБАВЛЕНО: LogisticRegression
from catboost import CatBoostClassifier

# === ИМПОРТЫ ИЗ UTILS ===
from utils import (
    load_prepared_data,
    save_model_artifacts,
    setup_logging, # Функция для настройки логирования
    N_SPLITS_CV,    # Общее количество фолдов
    RANDOM_STATE,   # Общий random_state
    PLOTS_DIR,      # Общая папка для графиков
    MODELS_DIR      # Общая папка для моделей (используется в save_model_artifacts)
)

# === Настройка логгера (через utils) ===
setup_logging() # Вызываем функцию настройки из utils
logger = logging.getLogger(__name__) # Получаем логгер для текущего файла

# === Константы и Настройки для текущей задачи ===
TARGET_COLUMN_ORIGINAL = 'IC50' # Оригинальное имя колонки для вычисления таргета
TASK_PREFIX = "ic50_median"      # Префикс для имен файлов и логов
# N_SPLITS_CV и RANDOM_STATE теперь импортируются из utils

# ДОБАВЛЕНО: Директория для графиков этой задачи
TASK_PLOTS_DIR = os.path.join(PLOTS_DIR, "classification", TASK_PREFIX)
os.makedirs(TASK_PLOTS_DIR, exist_ok=True)


# === Подготовка признаков и целевой переменной ===
def prepare_feature_target(df: pd.DataFrame, target_column_original: str, task_prefix_log: str):
    """
    Готовит признаки (X) и целевую переменную (y) для задачи классификации "> медианы".
    """
    logger.info(f"Подготовка признаков и таргета для задачи: {task_prefix_log}")

    # Расчет медианы для таргета
    # ВАЖНО: Медиана рассчитывается по всему доступному датасету df (который загружен)
    # Это обычная практика для создания бинарного таргета, который затем используется в CV.
    # Для "1000% безупречности" при HPO, где порог может влиять на выбор модели,
    # этот шаг можно было бы делать внутри каждого фолда HPO.
    # Но для данной задачи (классификация > фиксированной медианы) это приемлемо.
    if target_column_original not in df.columns:
        logger.error(f"Целевая колонка '{target_column_original}' не найдена в DataFrame.")
        raise ValueError(f"Целевая колонка '{target_column_original}' не найдена.")

    median_val = df[target_column_original].median()
    logger.info(f"Для задачи '{task_prefix_log}' используется медиана '{target_column_original}' = {median_val:.4f} как порог.")
    y_target = (df[target_column_original] > median_val).astype(int)

    # Исключение целевых и потенциально "протекающих" признаков
    # data_prepared.csv УЖЕ содержит log_IC50, log_CC50, log_SI
    cols_to_exclude = ['IC50', 'CC50', 'SI', 'log_IC50', 'log_CC50', 'log_SI']
    
    # Убедимся, что все колонки из cols_to_exclude, которые есть в df, будут удалены
    actual_cols_to_drop = [col for col in cols_to_exclude if col in df.columns]
    
    X_features = df.drop(columns=actual_cols_to_drop)
    feature_names = X_features.columns.tolist()

    # Проверка на наличие нечисловых колонок в X_features (кроме тех, что CatBoost может обработать как категориальные)
    # Для простоты сейчас ожидаем, что все признаки числовые или CatBoost сам разберется.
    # Если есть строковые object колонки, которые не являются категориальными, SimpleImputer упадет.
    # eda.py уже должен был подготовить числовые признаки.

    logger.info(f"Количество признаков для X: {len(feature_names)}")
    logger.info(f"Распределение классов в y_target для {task_prefix_log}: {y_target.value_counts(normalize=True)}")
    
    if y_target.nunique() < 2:
        logger.warning(f"В целевой переменной y_target для {task_prefix_log} только один класс. Могут возникнуть проблемы с обучением/оценкой.")

    return X_features, y_target, feature_names, median_val


# === Обучение и оценка моделей ===
def train_evaluate_model(X: pd.DataFrame, y: pd.Series, feature_names: list, task_prefix: str,
                         model_name: str, model_pipeline: Pipeline, cv_splits: int,
                         is_catboost: bool = False):
    """
    Обучает и оценивает модель с использованием кросс-валидации.
    """
    logger.info(f"--- Кросс-валидация для модели: {model_name} ({task_prefix}) ---")
    
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    fold_metrics = {
        'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.debug(f"Фолд {fold + 1}/{cv_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Обучаем пайплайн (который включает импьютацию, масштабирование и модель)
        model_pipeline.fit(X_train, y_train)
        
        y_pred = model_pipeline.predict(X_val)
        y_proba = model_pipeline.predict_proba(X_val)[:, 1]

        fold_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        fold_metrics['f1'].append(f1_score(y_val, y_pred))
        fold_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        if len(np.unique(y_val)) > 1 : # roc_auc требует как минимум два класса в y_val
             fold_metrics['roc_auc'].append(roc_auc_score(y_val, y_proba))
        else:
            fold_metrics['roc_auc'].append(np.nan) # или 0.5, или пропустить

    logger.info(f"Средние CV метрики для {model_name} ({task_prefix}):")
    for metric_name, values in fold_metrics.items():
        mean_val = np.nanmean(values) # Используем nanmean на случай NaN в ROC AUC
        std_val = np.nanstd(values)
        logger.info(f"  {metric_name.capitalize()}: {mean_val:.4f} (+/- {std_val:.4f})")
    
    return fold_metrics # Возвращаем метрики для возможного дальнейшего анализа


def find_best_catboost_params(X: pd.DataFrame, y: pd.Series, feature_names: list):
    """
    Ищет лучшие гиперпараметры для CatBoostClassifier с помощью RandomizedSearchCV.
    """
    logger.info("--- Поиск лучших гиперпараметров для CatBoost ---")

    # Пайплайн для CatBoost внутри RandomizedSearchCV
    # Важно: CatBoost может сам обрабатывать пропуски, но для консистентности с другими моделями
    # и чтобы RobustScaler не падал, используем SimpleImputer.
    # CatBoost также может работать с категориальными признаками без явного кодирования,
    # но сейчас все признаки предполагаются числовыми после eda.py.
    
    # Рассчитываем scale_pos_weight для всего набора данных y
    # Это будет использоваться как фиксированное значение в сетке поиска, если не тюним его отдельно
    # или если не используем auto_class_weights
    count_neg = np.sum(y == 0)
    count_pos = np.sum(y == 1)
    scale_pos_weight_val = count_neg / count_pos if count_pos > 0 else 1.0
    logger.info(f"Рассчитанный scale_pos_weight для CatBoost HPO: {scale_pos_weight_val:.2f}")

    catboost_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('catboost', CatBoostClassifier(
            random_state=RANDOM_STATE,
            verbose=0, # Отключаем вывод CatBoost во время HPO
            # scale_pos_weight=scale_pos_weight_val # Можно установить здесь или использовать auto_class_weights
            # auto_class_weights='Balanced' # Попробуем этот вариант, он проще
        ))
    ])

    # Сетка параметров для RandomizedSearchCV
    # Убрал iterations из сетки, лучше задать достаточно большое значение и использовать early_stopping
    # если CatBoost используется напрямую в CV. Для RandomizedSearchCV просто задаем iterations.
    param_dist = {
        'catboost__iterations': [500, 1000, 1500], # Увеличил варианты
        'catboost__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'catboost__depth': [4, 6, 8], # Уменьшил максимальную глубину для скорости
        'catboost__l2_leaf_reg': [1, 3, 5, 7, 9],
        'catboost__auto_class_weights': ['Balanced', None] # ДОБАВЛЕНО: Тюнинг балансировки
        # Если auto_class_weights=None, можно добавить 'catboost__scale_pos_weight': [scale_pos_weight_val, 1.0, (count_neg/count_pos)*0.5, (count_neg/count_pos)*1.5]
        # но это усложнит сетку. 'Balanced' должен хорошо работать.
    }
    
    # RandomizedSearchCV будет использовать свой внутренний CV
    # N_ITER_SEARCH - количество комбинаций параметров для проверки
    # CV_FOLDS_HPO - количество фолдов для внутреннего CV в RandomizedSearchCV
    N_ITER_SEARCH = 20 # Можно увеличить для более тщательного поиска
    CV_FOLDS_HPO = 3   # Можно увеличить, но замедлит HPO
    
    random_search = RandomizedSearchCV(
        estimator=catboost_pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=StratifiedKFold(n_splits=CV_FOLDS_HPO, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc', # Фокусируемся на ROC AUC для HPO
        n_jobs=-1, # Используем все доступные ядра
        random_state=RANDOM_STATE,
        verbose=1 # Вывод информации о ходе RandomizedSearchCV
    )

    random_search.fit(X, y)

    logger.info(f"Лучшие найденные параметры для CatBoost: {random_search.best_params_}")
    logger.info(f"Лучший ROC AUC по итогам HPO (на внутренних фолдах): {random_search.best_score_:.4f}")
    
    # Извлекаем параметры именно для шага 'catboost' в пайплайне
    best_catboost_params = {k.split('catboost__', 1)[1]: v 
                            for k, v in random_search.best_params_.items() 
                            if 'catboost__' in k}
    return best_catboost_params


def plot_feature_importances(importances_series: pd.Series, task_prefix: str, top_n: int = 20):
    """Строит и сохраняет график важности признаков."""
    plt.figure(figsize=(10, max(6, top_n // 2))) # Адаптивный размер
    importances_series.head(top_n).plot(kind='barh')
    plt.title(f'Топ-{top_n} важных признаков ({task_prefix.replace("_", " ").title()})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(TASK_PLOTS_DIR, f"feature_importance_{task_prefix}.png")
    plt.savefig(plot_path)
    logger.info(f"График важности признаков сохранен: {plot_path}")
    plt.close()

def plot_final_roc_curve(y_true, y_proba, task_prefix):
    """Строит и сохраняет ROC-кривую для финальной модели."""
    if len(np.unique(y_true)) < 2:
        logger.warning(f"В y_true для {task_prefix} только один класс, ROC-кривая не строится.")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Final Model ({task_prefix.replace("_", " ").title()})')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    roc_plot_path = os.path.join(TASK_PLOTS_DIR, f"roc_final_model_{task_prefix}.png")
    plt.savefig(roc_plot_path)
    logger.info(f"ROC-кривая для финальной модели сохранена: {roc_plot_path}")
    plt.close()


# === Основная функция ===
def main():
    logger.info(f"--- Запуск скрипта: {TASK_PREFIX} (Классификация '{TARGET_COLUMN_ORIGINAL}' > медианы) ---")
    
    df_full = load_prepared_data()
    if df_full is None:
        logger.error("Не удалось загрузить данные. Завершение скрипта.")
        return

    try:
        X_features, y_target, feature_names, _ = prepare_feature_target(
            df_full, TARGET_COLUMN_ORIGINAL, TASK_PREFIX
        )
    except ValueError as e_prep:
        logger.error(f"Ошибка подготовки данных для {TASK_PREFIX}: {e_prep}. Завершение скрипта.")
        return

    # --- 1. Baseline модель: Logistic Regression ---
    log_reg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('logreg', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced'))
    ])
    train_evaluate_model(X_features, y_target, feature_names, TASK_PREFIX, 
                         "Logistic Regression (Baseline)", log_reg_pipeline, N_SPLITS_CV)

    # --- 2. Поиск лучших гиперпараметров для CatBoost ---
    best_catboost_hpo_params = find_best_catboost_params(X_features, y_target, feature_names)
    
    # --- 3. Оценка CatBoost с лучшими параметрами на CV ---
    # Создаем пайплайн CatBoost с лучшими параметрами из HPO
    catboost_best_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('catboost', CatBoostClassifier(
            **best_catboost_hpo_params, # Используем лучшие параметры
            random_state=RANDOM_STATE,
            verbose=0 # Отключаем вывод для CV оценки
        ))
    ])
    train_evaluate_model(X_features, y_target, feature_names, TASK_PREFIX, 
                         "CatBoost (Best HPO Params)", catboost_best_pipeline, N_SPLITS_CV, is_catboost=True)

    # --- 4. Обучение финальной модели CatBoost на всех данных с лучшими параметрами ---
    logger.info(f"--- Обучение финальной модели CatBoost ({TASK_PREFIX}) на всех данных ---")
    final_imputer = SimpleImputer(strategy='median')
    final_scaler = RobustScaler()
    
    X_imputed = final_imputer.fit_transform(X_features)
    X_scaled = final_scaler.fit_transform(X_imputed)
    
    final_catboost_model = CatBoostClassifier(
        **best_catboost_hpo_params,
        random_state=RANDOM_STATE,
        verbose=100 # Можно выводить информацию каждые 100 итераций
    )
    final_catboost_model.fit(X_scaled, y_target) # Обучаем на всех данных
    logger.info("Финальная модель CatBoost обучена.")

    # --- 5. Важность признаков для финальной модели CatBoost ---
    try:
        importances = final_catboost_model.get_feature_importance()
        importances_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        logger.info(f"Топ-10 важных признаков ({TASK_PREFIX}):\n{importances_series.head(10)}")
        plot_feature_importances(importances_series, TASK_PREFIX)
    except Exception as e_fi:
        logger.error(f"Ошибка при получении или отображении важности признаков: {e_fi}")

    # --- 6. ROC-кривая для финальной модели CatBoost ---
    # (это было упущено в оригинальном clf_ic50_median.py)
    try:
        y_proba_final = final_catboost_model.predict_proba(X_scaled)[:, 1]
        plot_final_roc_curve(y_target, y_proba_final, TASK_PREFIX)
    except Exception as e_roc:
        logger.error(f"Ошибка при построении ROC-кривой для финальной модели: {e_roc}")

    # --- 7. Сохранение артефактов финальной модели CatBoost ---
    artifacts_to_save = {
        'model': final_catboost_model,
        'imputer': final_imputer,
        'scaler': final_scaler,
        'features': feature_names
    }
    save_model_artifacts(artifacts_to_save, TASK_PREFIX, model_type_dir="classification")
    
    logger.info(f"--- Скрипт {TASK_PREFIX} завершен ---")

# === Точка входа ===
if __name__ == '__main__':
    main()