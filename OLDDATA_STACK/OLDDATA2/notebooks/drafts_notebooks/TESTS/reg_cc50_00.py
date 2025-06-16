import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold # Используем KFold для регрессии
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# === Логгирование ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Константы и Настройки ===
TARGET_COLUMN_LOG = 'log_CC50' # ИЗМЕНЕНО: Логарифмированная целевая колонка
TARGET_COLUMN_ORIGINAL = 'CC50' # Для удаления выбросов
TASK_PREFIX = "cc50_reg"       # ИЗМЕНЕНО: Префикс для имен файлов и логов
N_SPLITS_CV = 10                # Количество фолдов для кросс-валидации

# === Загрузка данных и специфичная предобработка (удаление выбросов) ===
def load_and_preprocess_data(path="data/data_prepared.csv", target_original_for_outliers=TARGET_COLUMN_ORIGINAL):
    """
    Загружает данные, добавляет признаки, удаляет выбросы по указанной колонке.
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов из '{path}'")
    except FileNotFoundError:
        logger.error(f"Файл не найден по пути: {path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {path}: {e}")
        raise

    # Создание логарифмированных и производных признаков (некоторые будут исключены из X)
    if 'IC50' in df.columns: df["log_IC50"] = np.log1p(df["IC50"])
    if 'CC50' in df.columns: df["log_CC50"] = np.log1p(df["CC50"]) # Это будет наш таргет
    if 'SI' in df.columns: df["log_SI"] = np.log1p(df["SI"])

    if 'IC50' in df.columns and 'CC50' in df.columns:
        df["ratio_IC50_CC50"] = df["IC50"] / (df["CC50"] + 1e-6)
    else: logger.warning("'IC50' или 'CC50' отсутствуют, 'ratio_IC50_CC50' не создан.")

    if 'Chi0' in df.columns and 'Chi1' in df.columns:
        df["chi_ratio"] = df["Chi0"] / (df["Chi1"] + 1e-6) 
    else: logger.warning("'Chi0' или 'Chi1' отсутствуют, 'chi_ratio' не создан.")
        
    # Удаление выбросов по ОРИГИНАЛЬНОЙ колонке CC50 (как в вашем коде)
    if target_original_for_outliers in df.columns:
        initial_rows = df.shape[0]
        threshold = df[target_original_for_outliers].quantile(0.99)
        df = df[df[target_original_for_outliers] <= threshold].copy() # .copy() для избежания SettingWithCopyWarning
        rows_after_outliers = df.shape[0]
        logger.info(f"Удалены выбросы по '{target_original_for_outliers}' > {threshold:.2f} (было {initial_rows} строк, стало {rows_after_outliers} строк).")
    else:
        logger.warning(f"Колонка '{target_original_for_outliers}' для удаления выбросов не найдена.")

    logger.debug(f"Колонки в df после load_and_preprocess_data: {df.columns.tolist()}")
    return df

# === Подготовка признаков X и целевой переменной y ===
def prepare_feature_target(df, target_log_col_name, task_prefix):
    """
    Выбирает признаки (X) и формирует целевую переменную (y) для регрессии.
    """
    COLS_TO_EXCLUDE_FROM_FEATURES = [
        'IC50', 'CC50', 'SI',
        'log_IC50', 'log_CC50', 'log_SI', # Все лог-таргеты исключаем из признаков
        'ratio_IC50_CC50',
        'Unnamed: 0' 
    ]
    
    all_df_columns = df.columns.tolist()
    features_x = [col for col in all_df_columns if col not in COLS_TO_EXCLUDE_FROM_FEATURES and col in df.columns]
    
    if not features_x:
        logger.error(f"Для {task_prefix}: список признаков (features_x) пуст!")
        raise ValueError("Список признаков X не может быть пустым.")
    logger.info(f"Для {task_prefix}: Количество используемых признаков X: {len(features_x)}. Первые несколько: {features_x[:5]}")
    logger.debug(f"Для {task_prefix}: Полный список признаков X: {features_x}")

    # Формирование целевой переменной y (логарифмированное значение)
    if target_log_col_name not in df.columns:
        msg = f"Для {task_prefix}: целевая колонка '{target_log_col_name}' отсутствует в DataFrame."
        logger.error(msg)
        raise ValueError(msg)
    y = df[target_log_col_name].copy() 
    
    # Проверка на NaN в целевой переменной после всех манипуляций
    if y.isnull().any():
        logger.warning(f"Для {task_prefix}: в целевой переменной '{target_log_col_name}' есть NaN значения. Удаляем строки с NaN в таргете.")
        nan_mask = y.isnull()
        y = y[~nan_mask]
        X_df_temp = df[features_x].copy()
        X = X_df_temp[~nan_mask]
        logger.info(f"Для {task_prefix}: после удаления NaN в таргете, X: {X.shape}, y: {y.shape}")
    else:
        X = df[features_x].copy()

    if X.empty or y.empty:
        msg = f"Для {task_prefix}: X или y пусты после обработки NaN в таргете."
        logger.error(msg)
        raise ValueError(msg)
        
    return X, y, features_x

# === Обучение регрессионной модели с кросс-валидацией и обучение финальной модели ===
def train_model_cv_and_final(X_df, y_series, task_prefix, n_splits=N_SPLITS_CV):
    """
    Обучает регрессионную модель с использованием KFold, оценивает производительность,
    а затем обучает финальную модель на всех данных.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = {"mae": [], "rmse": [], "r2": []}

    # Параметры модели RandomForestRegressor
    # n_estimators=200 как в вашем коде, можно подобрать через HPO
    rf_params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1} 

    logger.info(f"Для {task_prefix}: Начало кросс-валидации ({n_splits} фолдов)...")

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_df, y_series)):
        logger.info(f"Для {task_prefix}: Фолд {fold_num + 1}/{n_splits}")
        
        X_train_fold, X_test_fold = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train_fold, y_test_fold = y_series.iloc[train_idx], y_series.iloc[test_idx]

        imputer_fold = SimpleImputer(strategy="median")
        X_train_fold_imp = imputer_fold.fit_transform(X_train_fold)
        X_test_fold_imp = imputer_fold.transform(X_test_fold)
        
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_imp)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold_imp)
        
        model_fold = RandomForestRegressor(**rf_params)
        
        try:
            model_fold.fit(X_train_fold_scaled, y_train_fold)
        except Exception as e_fit_fold:
            logger.error(f"Фолд {fold_num+1} ({task_prefix}): ошибка обучения модели: {e_fit_fold}")
            for metric_key in fold_metrics: fold_metrics[metric_key].append(np.nan)
            continue

        y_pred_fold = model_fold.predict(X_test_fold_scaled)

        fold_metrics["mae"].append(mean_absolute_error(y_test_fold, y_pred_fold))
        fold_metrics["rmse"].append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
        fold_metrics["r2"].append(r2_score(y_test_fold, y_pred_fold))

    logger.info(f"=== Средние метрики по CV для {task_prefix} ===")
    for metric_name, scores in fold_metrics.items():
        logger.info(f"{metric_name.upper()}: {np.nanmean(scores):.4f}")
    
    logger.info(f"Для {task_prefix}: Обучение финальной регрессионной модели на всех данных...")
    
    final_imputer = SimpleImputer(strategy="median")
    X_df_imputed = final_imputer.fit_transform(X_df)
    
    final_scaler = RobustScaler()
    X_df_scaled = final_scaler.fit_transform(X_df_imputed)
    
    final_model = RandomForestRegressor(**rf_params)
    try:
        final_model.fit(X_df_scaled, y_series) 
        logger.info(f"Для {task_prefix}: Финальная регрессионная модель обучена.")
    except Exception as e_fit_final:
        logger.error(f"Для {task_prefix}: ошибка обучения финальной модели: {e_fit_final}")
        return None, None, None

    return final_model, final_imputer, final_scaler

# === Сохранение артефактов ===
def save_artifacts(model, imputer, scaler, feature_names, prefix):
    """Сохраняет финальную модель и артефакты препроцессинга для регрессии."""
    if model is None or imputer is None or scaler is None:
        logger.error(f"Один из артефактов для {prefix} (регрессия) равен None, сохранение отменено.")
        return
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model,     f"models/{prefix}_reg_model.joblib") # _reg_ для регрессии
        joblib.dump(imputer,   f"models/{prefix}_reg_imputer.joblib")
        joblib.dump(scaler,    f"models/{prefix}_reg_scaler.joblib")
        joblib.dump(feature_names,  f"models/{prefix}_reg_features.joblib")
        logger.info(f"Сохранены артефакты для {prefix} (регрессия): models/{prefix}_reg_*.joblib")
    except Exception as e:
        logger.error(f"Ошибка при сохранении артефактов для {prefix} (регрессия): {e}")

# === Точка входа ===
def main():
    logger.info(f"--- Запуск скрипта для задачи: {TASK_PREFIX} (Регрессия '{TARGET_COLUMN_LOG}' с CV) ---")
    
    try:
        df_processed_for_outliers = load_and_preprocess_data(target_original_for_outliers=TARGET_COLUMN_ORIGINAL)
        X_clean_features, y_target, feature_names_list = prepare_feature_target(df_processed_for_outliers, TARGET_COLUMN_LOG, TASK_PREFIX)
    except Exception as e_data_prep:
        logger.error(f"Критическая ошибка на этапе загрузки или подготовки признаков/таргета для {TASK_PREFIX}: {e_data_prep}")
        return

    if X_clean_features.empty or y_target.empty:
        logger.error(f"Для {TASK_PREFIX}: X или y пусты перед обучением модели. Прерывание.")
        return

    final_model_trained, final_imputer_trained, final_scaler_trained = train_model_cv_and_final(
        X_clean_features, y_target, TASK_PREFIX, n_splits=N_SPLITS_CV
    )
    
    if final_model_trained:
        save_artifacts(final_model_trained, final_imputer_trained, final_scaler_trained, feature_names_list, prefix=TASK_PREFIX)
        
        # Построение scatter-графика для финальной модели на всех данных
        if final_imputer_trained and final_scaler_trained:
            logger.info(f"Построение scatter-графика для финальной модели {TASK_PREFIX} (на всех данных)...")
            X_processed_all = final_scaler_trained.transform(final_imputer_trained.transform(X_clean_features))
            try:
                y_pred_all = final_model_trained.predict(X_processed_all)
                
                plt.figure(figsize=(8, 6))
                plt.scatter(y_target, y_pred_all, alpha=0.6, edgecolors='w', linewidth=0.5)
                # Добавляем линию y=x
                min_val = min(y_target.min(), y_pred_all.min())
                max_val = max(y_target.max(), y_pred_all.max())
                plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal fit (y=x)")
                
                plt.xlabel(f"Настоящие значения {TARGET_COLUMN_LOG}")
                plt.ylabel(f"Предсказанные значения {TARGET_COLUMN_LOG}")
                plt.title(f"Регрессия (финальная модель на всех данных) — {TASK_PREFIX.replace('_reg', '').upper()}")
                plt.legend()
                plt.grid(True)
                plots_dir = "plots"
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(f"{plots_dir}/scatter_final_model_{TASK_PREFIX}.png")
                plt.close()
                logger.info(f"Scatter-график для финальной модели {TASK_PREFIX} сохранен.")
            except Exception as e_scatter_final:
                logger.warning(f"Не удалось построить/сохранить scatter-график для финальной модели {TASK_PREFIX}: {e_scatter_final}")
    else:
        logger.warning(f"Финальная модель для {TASK_PREFIX} не была обучена, артефакты не сохранены.")
    
    logger.info(f"--- Скрипт для задачи {TASK_PREFIX} завершен ---")

if __name__ == "__main__":
    main()