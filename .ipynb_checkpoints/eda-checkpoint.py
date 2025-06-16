# eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import umap.umap_ as umap
import plotly.express as px # Для интерактивных 3D-графиков

# Импортируем функции и константы из utils
from utils import setup_logging, get_logger, PLOTS_DIR, DATA_DIR, DATA_PREPARED_PATH

# === Настройки ===
setup_logging() # Настраиваем логирование
logger = get_logger(__name__)

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

# Определяем BASE_DIR здесь, так как он нужен для поиска исходного файла
# Это определение BASE_DIR предполагает, что eda.py находится в корневой папке проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Используем PLOTS_DIR из utils, создаем подпапку для EDA
EDA_PLOTS_DIR = os.path.join(PLOTS_DIR, "eda")
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

# Константы для EDA
RANDOM_STATE_UMAP = 42
IQR_MULTIPLIER = 1.5
VARIANCE_THRESHOLD_VALUE = 0.01 * (1 - 0.01) # Удаляем признаки, где одно значение встречается в >99% или <1% случаев


# === Функция для анализа выбросов (IQR) ===
def analyze_outliers_iqr(data, feature, log_scale=False):
    """Анализирует выбросы с использованием IQR и выводит информацию."""
    if feature not in data.columns:
        logger.warning(f"Колонка '{feature}' для анализа выбросов не найдена.")
        return None
        
    target_data = data[feature].copy() # Работаем с копией
    if target_data.isnull().all(): # Если все значения NaN
        logger.warning(f"Все значения в колонке '{feature}' являются NaN. Анализ выбросов невозможен.")
        return None

    if log_scale:
        # Применяем log1p только к положительным значениям, NaN остаются NaN
        target_data_positive = target_data[target_data > 0]
        if not target_data_positive.empty:
            target_data.loc[target_data_positive.index] = np.log1p(target_data_positive)
        feature_name = f"log1p({feature})"
    else:
        feature_name = feature

    q1 = target_data.quantile(0.25)
    q3 = target_data.quantile(0.75)
    
    if pd.isna(q1) or pd.isna(q3): # Если квантили не могут быть посчитаны (например, слишком много NaN)
        logger.warning(f"Не удалось рассчитать квантили для '{feature_name}'. Анализ выбросов пропущен.")
        return None

    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr
    
    # Фильтруем выбросы, игнорируя NaN в target_data при сравнении
    outliers = data[((target_data < lower_bound) | (target_data > upper_bound)) & pd.notna(target_data)]
    
    logger.info(f"Анализ выбросов для '{feature_name}': Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
                f"Границы=[{lower_bound:.2f}, {upper_bound:.2f}], Найдено выбросов: {len(outliers)}")
    return outliers

def main():
    logger.info("--- Начало EDA и подготовки данных ---")

    # === Загрузка данных ===
    try:
        raw_data_path_excel = os.path.join(BASE_DIR, 'Данные_для_курсовои_Классическое_МО.xlsx')
        if not os.path.exists(raw_data_path_excel):
             raw_data_path_excel = os.path.join(DATA_DIR, 'Данные_для_курсовои_Классическое_МО.xlsx')

        df = pd.read_excel(raw_data_path_excel)
        logger.info(f"Исходные данные загружены. Размер: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Исходный файл Excel не найден. Проверьте путь: {raw_data_path_excel} или {os.path.join(DATA_DIR, 'Данные_для_курсовои_Классическое_МО.xlsx')}")
        return
    except Exception as e:
        logger.error(f"Ошибка при загрузке исходных данных: {e}")
        return

    df = df.rename(columns={'IC50, mM': 'IC50_mM', 'CC50, mM': 'CC50_mM', 'SMILES': 'SMILES_orig'})
    # Удаляем колонку 'Unnamed: 0', если она есть (обычно это индекс из Excel)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        logger.info("Удалена колонка 'Unnamed: 0'.")
    logger.info(f"Колонки после переименования и удаления 'Unnamed: 0': {df.columns.tolist()}")


    # === Шаг 1: Корректировка IC50 и расчет SI ===
    logger.info("Шаг 1: Корректировка IC50 и расчет SI")
    if 'IC50_mM' not in df.columns or 'CC50_mM' not in df.columns:
        logger.error("Необходимые колонки 'IC50_mM' или 'CC50_mM' не найдены. Прекращение работы.")
        return

    if 'SI' not in df.columns:
        logger.warning("Исходная колонка 'SI' не найдена. Сравнение с оригинальным SI будет пропущено.")
        df['SI_original'] = np.nan
    else:
        df = df.rename(columns={'SI': 'SI_original'})

    df['IC50_nM'] = df['IC50_mM'] * 1_000_000
    logger.info(f"Создана колонка 'IC50_nM'. Пример значений: {df['IC50_nM'].head().values}")

    denominator_ic50_mm = df['IC50_mM'].replace(0, np.nan)
    df['SI_corrected'] = df['CC50_mM'] / denominator_ic50_mm
    
    si_corrected_nan_count = df['SI_corrected'].isnull().sum()
    si_corrected_inf_count = np.isinf(df['SI_corrected']).sum()
    if si_corrected_nan_count > 0:
        logger.info(f"В 'SI_corrected' найдено {si_corrected_nan_count} NaN значений (до импутации).")
    if si_corrected_inf_count > 0:
        logger.warning(f"В 'SI_corrected' найдено {si_corrected_inf_count} Inf значений.")
        df['SI_corrected'].replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info("Inf значения в 'SI_corrected' заменены на NaN.")

    logger.info(f"Создана колонка 'SI_corrected'. Пример значений: {df['SI_corrected'].head().values}")
    if 'SI_original' in df.columns and not df['SI_original'].isnull().all():
        df['SI_diff_check'] = np.abs(df['SI_original'] - df['SI_corrected'])
        # Логируем только если есть значительные расхождения
        significant_diff_count = (df['SI_diff_check'] > 0.1).sum()
        if significant_diff_count > 0:
            logger.info(f"Проверка SI: {significant_diff_count} значений 'SI_corrected' отличаются от 'SI_original' > 0.1.")
        else:
            logger.info("Проверка SI: 'SI_corrected' совпадает с 'SI_original' (расхождения <= 0.1).")


    # === Определение признаков (X) ===
    # Колонки, которые НЕ являются признаками
    non_feature_cols = [
        'ID_internal', 'ID_external', 'SMILES_orig', # Идентификаторы и текст
        'IC50_mM', 'CC50_mM', 'IC50_nM',            # Целевые и их производные
        'SI_original', 'SI_corrected', 'SI_diff_check' # SI и его проверки
    ]
    potential_feature_names = [col for col in df.columns if col not in non_feature_cols]
    
    # Выбираем только числовые признаки из потенциальных
    X_df_initial = df[potential_feature_names].select_dtypes(include=np.number)
    initial_feature_names = X_df_initial.columns.tolist()
    logger.info(f"Начальное количество потенциальных числовых признаков: {len(initial_feature_names)}")

    # === Шаг 2: Удаление низкодисперсионных признаков ===
    logger.info(f"Шаг 2: Удаление низкодисперсионных признаков (порог VarianceThreshold: {VARIANCE_THRESHOLD_VALUE:.5f})")
    if not X_df_initial.empty:
        # Перед VarianceThreshold нужно обработать NaN, иначе будет ошибка
        imputer_for_variance_check = SimpleImputer(strategy='median')
        X_for_variance_check = imputer_for_variance_check.fit_transform(X_df_initial)
        X_for_variance_check_df = pd.DataFrame(X_for_variance_check, columns=initial_feature_names)

        selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD_VALUE)
        selector.fit(X_for_variance_check_df)
        
        selected_features_mask = selector.get_support()
        X_df_variant = X_df_initial.loc[:, selected_features_mask] # Применяем маску к оригинальному X_df_initial с NaN
        
        removed_cols_variance = set(initial_feature_names) - set(X_df_variant.columns.tolist())
        logger.info(f"Удалено {len(removed_cols_variance)} низкодисперсионных признаков: {list(removed_cols_variance)[:10]}{'...' if len(removed_cols_variance) > 10 else ''}")
        logger.info(f"Осталось признаков после удаления низкодисперсионных: {X_df_variant.shape[1]}")
        current_feature_names = X_df_variant.columns.tolist()
    else:
        logger.warning("Нет числовых признаков для удаления по дисперсии.")
        X_df_variant = pd.DataFrame()
        current_feature_names = []

    # === Шаг 3: Импутация пропущенных значений в оставшихся признаках ===
    logger.info("Шаг 3: Импутация пропущенных значений в признаках (медианой)")
    if not X_df_variant.empty:
        imputer = SimpleImputer(strategy='median')
        # Применяем импьютер к X_df_variant, который мог содержать NaN
        X_imputed_np = imputer.fit_transform(X_df_variant)
        X_df_imputed = pd.DataFrame(X_imputed_np, columns=current_feature_names, index=X_df_variant.index) # Восстанавливаем индекс
        nan_after_imputation = X_df_imputed.isnull().sum().sum()
        if nan_after_imputation == 0:
            logger.info("Все пропуски в признаках успешно импутированы.")
        else:
            logger.warning(f"Осталось {nan_after_imputation} NaN в признаках после импутации! Проверьте данные.")
    else:
        logger.warning("Нет признаков для импутации.")
        X_df_imputed = pd.DataFrame()

    # === Формирование финального датафрейма для сохранения ===
    # Колонки, которые должны быть в data_prepared.csv:
    # Все целевые и важные информационные + очищенные признаки
    final_cols_for_prepared_df = ['IC50_nM', 'IC50_mM', 'CC50_mM', 'SI_corrected'] # Основные таргеты/инфо

    # Создаем df_prepared из оригинального df (df), чтобы сохранить корректные индексы и нечисловые колонки, если они были
    # Выбираем только существующие колонки из final_cols_for_prepared_df
    existing_final_cols = [col for col in final_cols_for_prepared_df if col in df.columns]
    df_prepared = df[existing_final_cols].copy()

    if not X_df_imputed.empty:
        # Присоединяем импутированные признаки к df_prepared
        # Убедимся, что индексы df_prepared и X_df_imputed совпадают
        if not df_prepared.index.equals(X_df_imputed.index):
            logger.warning("Индексы df_prepared и X_df_imputed не совпадают. Сброс индексов перед конкатенацией.")
            df_prepared = pd.concat([df_prepared.reset_index(drop=True), X_df_imputed.reset_index(drop=True)], axis=1)
        else:
            df_prepared = pd.concat([df_prepared, X_df_imputed], axis=1)
        
        df_prepared = df_prepared.loc[:,~df_prepared.columns.duplicated()] # Удаляем дубликаты колонок, если возникли

    # Импутация для целевых/информационных переменных в df_prepared (если в них остались NaN)
    for col_target_imp in existing_final_cols:
        if df_prepared[col_target_imp].isnull().any():
            median_val_target = df_prepared[col_target_imp].median()
            if pd.notna(median_val_target): # Убедимся, что медиана не NaN
                df_prepared[col_target_imp].fillna(median_val_target, inplace=True)
                logger.info(f"Пропуски в колонке '{col_target_imp}' заполнены медианой ({median_val_target:.2f}).")
            else:
                logger.warning(f"Не удалось рассчитать медиану для '{col_target_imp}' (все значения могут быть NaN). Пропуски не заполнены.")
    
    # === Сохранение data_prepared.csv ===
    try:
        df_prepared.to_csv(DATA_PREPARED_PATH, index=False)
        logger.info(f"Подготовленные данные сохранены в: {DATA_PREPARED_PATH}. Размер: {df_prepared.shape}")
        logger.info(f"Колонки в data_prepared.csv: {df_prepared.columns.tolist()}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении {DATA_PREPARED_PATH}: {e}")
        return # Выход, если не удалось сохранить

    # === Аналитическая часть EDA (визуализации) ===
    logger.info("--- Начало аналитической части EDA (визуализации на основе df_prepared) ---")
    
    key_cols_to_plot = ['IC50_nM', 'CC50_mM', 'SI_corrected'] # Также можно добавить 'IC50_mM'
    if 'IC50_mM' in df_prepared.columns and 'IC50_mM' not in key_cols_to_plot: # Для полноты картины
        key_cols_to_plot.insert(1, 'IC50_mM') 

    for col in key_cols_to_plot:
        if col in df_prepared.columns:
            plt.figure(figsize=(12, 4))
            # График 1: Распределение исходной колонки
            plt.subplot(1, 2, 1)
            sns.histplot(df_prepared[col].dropna(), kde=True, bins=50) # dropna() на случай, если импутация не сработала
            plt.title(f'Распределение {col}')
            # График 2: Распределение логарифмированной колонки
            plt.subplot(1, 2, 2)
            # Применяем log1p только к положительным значениям для корректного логарифмирования
            log_transformed_data = df_prepared[col][df_prepared[col] > 0].copy()
            if not log_transformed_data.empty:
                log_transformed_data = np.log1p(log_transformed_data)
                sns.histplot(log_transformed_data.dropna(), kde=True, bins=50)
            plt.title(f'Распределение log1p({col}) (только >0)')
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_PLOTS_DIR, f'distribution_{col}.png'))
            plt.close()
            analyze_outliers_iqr(df_prepared, col)
            analyze_outliers_iqr(df_prepared, col, log_scale=True)
        else:
            logger.warning(f"Колонка {col} для построения распределения не найдена в df_prepared.")

    if not df_prepared.empty and any(col in df_prepared.columns for col in key_cols_to_plot):
        existing_key_cols = [col for col in key_cols_to_plot if col in df_prepared.columns]
        desc_stats = df_prepared[existing_key_cols].describe().T
        logger.info(f"Описательная статистика для ключевых колонок:\n{desc_stats}")
    
    if all(col in df_prepared.columns for col in key_cols_to_plot): # Используем исходный key_cols_to_plot для консистентности
        pairplot_data = df_prepared[key_cols_to_plot].copy()
        for col in key_cols_to_plot:
            # log1p для положительных значений
            positive_values = pairplot_data[col][pairplot_data[col] > 0]
            if not positive_values.empty:
                 pairplot_data[f'log1p_{col}'] = np.nan # Инициализация
                 pairplot_data.loc[positive_values.index, f'log1p_{col}'] = np.log1p(positive_values)
            else:
                pairplot_data[f'log1p_{col}'] = np.nan

        log_pairplot_cols = [f'log1p_{col}' for col in key_cols_to_plot if f'log1p_{col}' in pairplot_data.columns]
        if len(log_pairplot_cols) >=2 : # Нужно хотя бы 2 колонки для pairplot
            plt.figure()
            # Удаляем строки где все значения log1p NaN, чтобы pairplot не падал
            sns.pairplot(pairplot_data[log_pairplot_cols].dropna(how='all'), diag_kind='kde', corner=True)
            plt.suptitle('Pairplot для log1p(ключевых колонок)', y=1.02)
            plt.savefig(os.path.join(EDA_PLOTS_DIR, 'pairplot_log_targets.png'))
            plt.close()
        else:
            logger.warning("Недостаточно колонок с логарифмированными значениями для pairplot.")


    # Признаки для корреляции и MI - все, что не key_cols_to_plot в df_prepared
    features_for_analysis = [f for f in df_prepared.columns if f not in key_cols_to_plot]
    
    if features_for_analysis and not df_prepared[features_for_analysis].empty:
        X_for_eda_plots = df_prepared[features_for_analysis]

        plt.figure(figsize=(12, 10))
        correlation_matrix = X_for_eda_plots.corr(method='spearman') # Spearman лучше для нелинейных связей и выбросов
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".1f", vmin=-1, vmax=1)
        plt.title('Корреляционная матрица признаков (Spearman, из data_prepared)')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_PLOTS_DIR, 'correlation_matrix_prepared.png'))
        plt.close()

        if 'IC50_nM' in df_prepared.columns:
            # MI требует, чтобы X не содержал NaN. X_for_eda_plots уже импутирован.
            # Целевая переменная также не должна содержать NaN. np.log1p(df_prepared['IC50_nM']) может вернуть NaN, если IC50_nM < 0
            target_for_mi = np.log1p(df_prepared['IC50_nM'][df_prepared['IC50_nM'] >= 0]) # Убираем отрицательные перед логарифмом
            X_for_mi = X_for_eda_plots.loc[target_for_mi.index] # Согласуем X с таргетом
            target_for_mi = target_for_mi.fillna(target_for_mi.median()) # Импутируем NaN в таргете, если остались

            if not X_for_mi.empty and not target_for_mi.empty:
                mi_ic50 = mutual_info_regression(X_for_mi, target_for_mi, random_state=RANDOM_STATE_UMAP, n_neighbors=3) # n_neighbors может помочь с непрерывными
                mi_ic50_series = pd.Series(mi_ic50, index=X_for_mi.columns).sort_values(ascending=False)
                plt.figure(figsize=(10, max(8, len(mi_ic50_series)//3)))
                mi_ic50_series.head(min(30, len(mi_ic50_series))).plot(kind='barh') # Показываем топ-30 или меньше
                plt.title('Top MI с log1p(IC50_nM)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(EDA_PLOTS_DIR, 'mi_ic50nm.png'))
                plt.close()
            else:
                logger.warning("Недостаточно данных для расчета MI для IC50_nM.")


        if X_for_eda_plots.shape[0] > 0 and X_for_eda_plots.shape[1] > 1 :
            logger.info("Выполняется PCA и UMAP для визуализации...")
            scaler_viz = RobustScaler()
            X_scaled_viz = scaler_viz.fit_transform(X_for_eda_plots) # X_for_eda_plots уже импутирован

            pca_viz = PCA(n_components=2, random_state=RANDOM_STATE_UMAP).fit_transform(X_scaled_viz)
            try:
                umap_viz_2d = umap.UMAP(n_components=2, random_state=RANDOM_STATE_UMAP, n_jobs=1, n_neighbors=15, min_dist=0.1).fit_transform(X_scaled_viz)
                
                for col_target_viz in key_cols_to_plot: # Используем key_cols_to_plot
                    if col_target_viz in df_prepared.columns:
                        # log1p для положительных значений
                        valid_target_data = df_prepared[col_target_viz][df_prepared[col_target_viz] >= 0]
                        if not valid_target_data.empty:
                            log_target_viz = np.log1p(valid_target_data)
                            # Согласуем индексы для раскраски
                            pca_subset = pca_viz[valid_target_data.index]
                            umap_subset = umap_viz_2d[valid_target_data.index]

                            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                            axs[0].scatter(pca_subset[:, 0], pca_subset[:, 1], c=log_target_viz, cmap='viridis', alpha=0.7, s=10)
                            axs[0].set_title(f'PCA — log1p({col_target_viz})')
                            axs[1].scatter(umap_subset[:, 0], umap_subset[:, 1], c=log_target_viz, cmap='plasma', alpha=0.7, s=10)
                            axs[1].set_title(f'UMAP — log1p({col_target_viz})')
                            for ax_item in axs: ax_item.set_xticks([]); ax_item.set_yticks([])
                            fig.tight_layout()
                            plt.savefig(os.path.join(EDA_PLOTS_DIR, f'projection_2d_log1p_{col_target_viz}.png'))
                            plt.close()
            except Exception as e_umap:
                logger.error(f"Ошибка при выполнении 2D UMAP: {e_umap}. UMAP 2D визуализации будут пропущены.")
        else:
            logger.warning("Недостаточно данных или признаков для PCA/UMAP 2D визуализаций.")

        # Интерактивные 3D-графики (Plotly)
        if X_for_eda_plots.shape[0] > 0 and X_for_eda_plots.shape[1] >= 3: # Нужно хотя бы 3 признака для 3D
            logger.info("Выполняется интерактивные 3D PCA и UMAP...")
            scaler_inter = RobustScaler()
            X_scaled_inter = scaler_inter.fit_transform(X_for_eda_plots)

            pca_3d_obj = PCA(n_components=3, random_state=RANDOM_STATE_UMAP)
            X_pca_3d = pca_3d_obj.fit_transform(X_scaled_inter)
            
            try:
                umap_3d_obj = umap.UMAP(n_components=3, random_state=RANDOM_STATE_UMAP, n_jobs=1, n_neighbors=15, min_dist=0.1)
                X_umap_3d = umap_3d_obj.fit_transform(X_scaled_inter)

                for target_col_3d in key_cols_to_plot: # Используем key_cols_to_plot
                    if target_col_3d in df_prepared.columns:
                        # log1p для положительных значений
                        valid_target_data_3d = df_prepared[target_col_3d][df_prepared[target_col_3d] >= 0]
                        if not valid_target_data_3d.empty:
                            color_data_3d = np.log1p(valid_target_data_3d)
                            
                            # Согласуем индексы
                            df_plot_3d = pd.DataFrame(index=valid_target_data_3d.index)
                            df_plot_3d['PC1'] = X_pca_3d[valid_target_data_3d.index, 0]
                            df_plot_3d['PC2'] = X_pca_3d[valid_target_data_3d.index, 1]
                            df_plot_3d['PC3'] = X_pca_3d[valid_target_data_3d.index, 2]
                            df_plot_3d['U1'] = X_umap_3d[valid_target_data_3d.index, 0]
                            df_plot_3d['U2'] = X_umap_3d[valid_target_data_3d.index, 1]
                            df_plot_3d['U3'] = X_umap_3d[valid_target_data_3d.index, 2]
                            df_plot_3d[f"log1p({target_col_3d})"] = color_data_3d

                            fig_pca_3d = px.scatter_3d(df_plot_3d, x='PC1', y='PC2', z='PC3',
                                                    color=f"log1p({target_col_3d})",
                                                    title=f"Interactive 3D PCA — log1p({target_col_3d})",
                                                    color_continuous_scale=px.colors.sequential.Viridis)
                            fig_pca_3d.write_html(os.path.join(EDA_PLOTS_DIR, f'interactive_3d_pca_{target_col_3d}.html'))

                            fig_umap_3d = px.scatter_3d(df_plot_3d, x='U1', y='U2', z='U3',
                                                     color=f"log1p({target_col_3d})",
                                                     title=f"Interactive 3D UMAP — log1p({target_col_3d})",
                                                     color_continuous_scale=px.colors.sequential.Plasma)
                            fig_umap_3d.write_html(os.path.join(EDA_PLOTS_DIR, f'interactive_3d_umap_{target_col_3d}.html'))
                logger.info("Интерактивные 3D-графики (Plotly) сохранены в HTML.")
            except Exception as e_umap_3d:
                 logger.error(f"Ошибка при выполнении 3D UMAP или Plotly: {e_umap_3d}")
        else:
            logger.warning("Недостаточно данных или признаков для интерактивных 3D визуализаций.")
    else:
        logger.warning("Нет признаков для анализа (корреляция, MI, проекции).")

    logger.info("--- EDA и подготовка данных завершены ---")

if __name__ == '__main__':
    main()