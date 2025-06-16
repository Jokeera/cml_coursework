# === eda_general.py ===

# === ЭТАП 0: Инициализация окружения и базовая загрузка ===

# === ИМПОРТЫ И НАСТРОЙКИ ===

# 📦 Базовые библиотеки
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

# 🧪 Статистика и визуализация
import scipy.stats as stats
from matplotlib import gridspec

# 🧠 ML / Feature Selection / Метрики
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 🌐 Визуализация проекций
import umap.umap_ as umap

# 🛠️ Настройки вывода и графиков
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", '{:.4f}'.format)
sns.set(style="whitegrid")

def main():

    # === СОЗДАНИЕ КАТАЛОГОВ ===
    os.makedirs("data/eda_gen", exist_ok=True)
    os.makedirs("plots/eda_gen", exist_ok=True)
    os.makedirs("data/eda_gen/scaled", exist_ok=True)
    os.makedirs("data/eda_gen/features", exist_ok=True)
    os.makedirs("plots/eda_gen/projections_task/pca", exist_ok=True)
    os.makedirs("plots/eda_gen/projections_task/umap", exist_ok=True)
    os.makedirs("plots/eda_gen/projections_task/lda", exist_ok=True)
    os.makedirs("plots/eda_gen/projections_variance", exist_ok=True)
    os.makedirs("plots/eda_gen/targets/log_transform", exist_ok=True)
    os.makedirs("plots/eda_gen/targets/strip", exist_ok=True)
    os.makedirs("plots/eda_gen/classification_targets", exist_ok=True)
    os.makedirs("plots/eda_gen/classification_targets/analysis", exist_ok=True)
    os.makedirs("plots/eda_gen/classification_targets/dummy_reports", exist_ok=True)
    os.makedirs("plots/eda_gen/outliers", exist_ok=True)
    os.makedirs("plots/dummy", exist_ok=True)
    os.makedirs("plots/eda_gen/features", exist_ok=True)
    os.makedirs("plots/eda_gen/feature_importance", exist_ok=True)


    # 📥 Загрузка исходного Excel-файла
    df = pd.read_excel("data/Данные_для_курсовои_Классическое_МО.xlsx")

    # ✅ Пути к финальным объектам
    DATA_PATH = "data/eda_gen/data_final.csv"
    X_SCALED_PATH = "data/eda_gen/scaled/X_scaled.csv"
    SCALER_PATH = "data/eda_gen/scaled/scaler_clf.pkl"

    # 🎯 Целевые переменные для классификации
    TARGET_COLUMNS = [
        "IC50_gt_median", "CC50_gt_median",
        "SI_gt_median", "SI_gt_8"
    ]

    # 🚫 Признаки, которые нельзя использовать как фичи
    FORBIDDEN_COLUMNS = [
        "IC50_nM", "CC50_nM", "SI_corrected",
        "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"
    ] + TARGET_COLUMNS

    # 🖨️ Общая информация о датасете
    print("✅ ЭТАП 0: Загрузка и первичный осмотр данных")
    print("📐 Размерность:", df.shape)
    print("📋 Первые строки:")
    print(df.head(3))
    print("📦 Типы данных:")
    print(df.dtypes.value_counts())
    print("🔍 Object-колонки:")
    print(df.select_dtypes(include='object').columns.tolist())
    print("📌 Дубликаты:", df.duplicated().sum())
    print("📌 Строк с >20 NaN:", (df.isnull().sum(axis=1) > 20).sum())

    # 📊 Подсчёт NaN по столбцам
    nan_stats = df.isnull().sum()
    nan_stats = nan_stats[nan_stats > 0].sort_values(ascending=False)
    if nan_stats.empty:
        print("✅ Пропусков нет")
    else:
        print("⚠️ Найдены пропуски:")
        print(nan_stats)

    print("✅ ЭТАП 0 завершён: Импорты и загрузка данных выполнены")


    # === ЭТАП 1: Подготовка целевых переменных и признаков ===


    # === Шаг 1: Очистка названий колонок ===
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.strip().str.replace(",", "").str.replace(" ", "_")

    # === Шаг 2: Удаление дубликатов и пропусков ===
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    # === Шаг 3: Преобразование в наноМоли ===
    df["IC50_nM"] = df["IC50_mM"] * 1e6
    df["CC50_nM"] = df["CC50_mM"] * 1e6

    # === Логарифмирование ===
    df["log1p_IC50_nM"] = np.log1p(df["IC50_nM"])
    df["log1p_CC50_nM"] = np.log1p(df["CC50_nM"])

    # === Расчёт SI и логарифм ===
    df["SI_corrected"] = df["CC50_nM"] / df["IC50_nM"]
    df["SI_corrected"] = df["SI_corrected"].replace([np.inf, -np.inf], np.nan)
    df["log1p_SI"] = np.log1p(df["SI_corrected"])

    # === Удаление NaN в критичных таргетах ===
    df = df.dropna(subset=["log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"])

    # === Удаление исходных SI/IC50/CC50 после использования ===
    df = df.drop(columns=["IC50_mM", "CC50_mM", "SI"], errors="ignore")

    # === Бинаризация таргетов ===
    df["IC50_gt_median"] = (df["IC50_nM"] > df["IC50_nM"].median()).astype(int)
    df["CC50_gt_median"] = (df["CC50_nM"] > df["CC50_nM"].median()).astype(int)
    df["SI_gt_median"] = (df["SI_corrected"] > df["SI_corrected"].median()).astype(int)
    df["SI_gt_8"] = (df["SI_corrected"] > 8).astype(int)


    # === Исходные и лог-преобразованные значения ===
    targets = {
        "IC50": ("IC50_nM", "log1p_IC50_nM"),
        "CC50": ("CC50_nM", "log1p_CC50_nM"),
        "SI": ("SI_corrected", "log1p_SI")
    }

    # === Функция визуализации до/после ===
    def plot_comparison(before, after, label, save_name):
        fig = plt.figure(figsize=(16, 5))
        spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

        ax0 = fig.add_subplot(spec[0, 0])
        sns.histplot(before, kde=True, bins=30, ax=ax0, color="skyblue")
        ax0.set_title(f"{label} - Гистограмма + KDE")

        ax1 = fig.add_subplot(spec[0, 1])
        sns.boxplot(y=before, ax=ax1, color="lightgreen")
        ax1.set_title(f"{label} - Boxplot")

        ax2 = fig.add_subplot(spec[0, 2])
        stats.probplot(before, dist="norm", plot=ax2)
        ax2.set_title("QQ Plot")

        mean = before.mean()
        std = before.std()
        skew = before.skew()
        kurt = before.kurt()
        iqr = before.quantile(0.75) - before.quantile(0.25)
        val_range = (before.min(), before.max())

        stats_text = f"""
        Mean      = {mean:.3f}
        Std       = {std:.3f}
        Skewness  = {skew:.3f}
        Kurtosis  = {kurt:.3f}
        IQR       = {iqr:.3f}
        Min-Max   = {val_range[0]:.3f} → {val_range[1]:.3f}
        """

        ax3 = fig.add_subplot(spec[0, 3])
        ax3.text(0.1, 0.5, stats_text, fontsize=12)
        ax3.axis("off")
        ax3.set_title("Статистики")

        plt.suptitle(f"{label}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"plots/eda_gen/targets/log_transform/{save_name}.png")
        plt.close()

    # === Генерация графиков до/после логарифмирования ===
    for name, (raw_col, log_col) in targets.items():
        plot_comparison(df[raw_col], df[raw_col], f"{name} — ДО логарифмирования", f"{name}_before_log")
        plot_comparison(df[log_col], df[log_col], f"{name} — ПОСЛЕ логарифмирования", f"{name}_after_log")

    # === Удаление утечек ===
    df = df.drop(columns=["IC50_nM", "CC50_nM", "SI_corrected"], errors="ignore")

    # === Проверки ===
    print("✅ Размерность после обработки:", df.shape)
    print("\n📊 Описание лог-таргетов:")
    print(df[["log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"]].describe())

    print("\n📊 Распределение классов:")
    for col in ["IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"]:
        print(f"{col}:")
        print(df[col].value_counts(normalize=True).rename_axis("class").reset_index(name="fraction"))
        print()

    # === Сохраняем результат ===
    df.to_csv("data/eda_gen/data_clean.csv", index=False)

    # === Проверка пропусков ===
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]

    if nan_counts.empty:
        print("✅ Пропущенных значений в датафрейме НЕТ.")
    else:
        print("🧨 Найдены пропуски:")
        print(nan_counts.sort_values(ascending=False))

    print("💾 Сохранено: data/eda_gen/data_clean.csv")






    # === ЭТАП 2: Анализ распределений лог-таргетов + удаление выбросов ===


    # === Настройки ===
    df = pd.read_csv("data/eda_gen/data_clean.csv")

    target_cols = ["log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"]

    # === Функции ===
    def plot_distribution(col, data, stage, path_prefix):
        fig = plt.figure(figsize=(16, 5))
        spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

        ax0 = fig.add_subplot(spec[0, 0])
        sns.histplot(data[col], kde=True, bins=30, ax=ax0, color="skyblue")
        ax0.set_title(f"{col} — Histogram ({stage})")

        ax1 = fig.add_subplot(spec[0, 1])
        sns.boxplot(y=data[col], ax=ax1, color="lightgreen")
        ax1.set_title(f"{col} — Boxplot ({stage})")

        ax2 = fig.add_subplot(spec[0, 2])
        stats.probplot(data[col], dist="norm", plot=ax2)
        ax2.set_title(f"{col} — QQ Plot ({stage})")

        ax3 = fig.add_subplot(spec[0, 3])
        stats_text = f"""
        Mean      = {data[col].mean():.3f}
        Std       = {data[col].std():.3f}
        Skewness  = {data[col].skew():.3f}
        Kurtosis  = {data[col].kurt():.3f}
        IQR       = {(data[col].quantile(0.75) - data[col].quantile(0.25)):.3f}
        Min-Max   = {data[col].min():.3f} → {data[col].max():.3f}
        """
        ax3.text(0.1, 0.5, stats_text, fontsize=12)
        ax3.axis("off")
        ax3.set_title("Статистика")

        plt.suptitle(f"EDA: {col} ({stage})", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"{path_prefix}/{col}_distribution_{stage}.png")
        plt.close()

    def remove_outliers_iqr(df, col):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = df.shape[0]
        print(f"🧹 {col}: удалено {before - after} выбросов по IQR")
        return df

    # === ЭТАП 2: Анализ распределений, выбросов, DummyRegressor и stripplot ===


    # === Константы ===
    target_cols = ["log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"]
    binary_targets = ["IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"]
    forbidden_cols = [
        "IC50_nM", "CC50_nM", "SI_corrected",
        "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI",
        "IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"
    ]


    # === Функции ===

    def get_stats_text(series):
        return f"""
        Mean      = {series.mean():.3f}
        Std       = {series.std():.3f}
        Skewness  = {series.skew():.3f}
        Kurtosis  = {series.kurt():.3f}
        IQR       = {series.quantile(0.75) - series.quantile(0.25):.3f}
        Min-Max   = {series.min():.3f} → {series.max():.3f}
        """

    def get_quantiles_robust(series):
        skew = series.skew()
        kurt = series.kurt()
        n = len(series)
        if n < 200:
            return 0.25, 0.75, 1.5, "IQR"
        if abs(skew) > 2 or kurt > 5:
            return None, None, 3, "MAD"
        if abs(skew) > 1.2 or kurt > 2.5:
            return 0.2, 0.6, 1.0, "IQR"
        if abs(skew) > 0.5 or kurt > 1.5:
            return 0.22, 0.68, 1.5, "IQR"
        return 0.25, 0.75, 1.5, "IQR"

    def plot_ecdf(series, title, path):
        x = np.sort(series)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.figure(figsize=(5, 4))
        plt.plot(x, y, marker='.', linestyle='none', color='blue')
        plt.xlabel(title)
        plt.ylabel("Доля ≤ значения")
        plt.title(f"ECDF: {title}")
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    # === 2.1: Распределения и удаление выбросов ===
    for col in target_cols:
        print(f"\n📊 Обработка переменной: {col}")

        # До удаления выбросов
        fig = plt.figure(figsize=(24, 5))
        spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

        ax0 = fig.add_subplot(spec[0, 0])
        sns.histplot(df[col], kde=True, bins=30, color="skyblue", ax=ax0)
        ax0.set_title(f"{col} — Histogram (Before)")

        ax1 = fig.add_subplot(spec[0, 1])
        sns.boxplot(y=df[col], color="salmon", ax=ax1)
        ax1.set_title(f"{col} — Boxplot (Before)")

        ax2 = fig.add_subplot(spec[0, 2])
        stats.probplot(df[col], dist="norm", plot=ax2)
        ax2.set_title(f"{col} — QQ Plot (Before)")

        ax3 = fig.add_subplot(spec[0, 3])
        ax3.text(0.05, 0.5, get_stats_text(df[col]), fontsize=12, verticalalignment='center')
        ax3.axis("off")
        ax3.set_title("Stats (Before)")

        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/outliers/{col}_before_outliers.png")
        plt.close()

        # ECDF до
        plot_ecdf(df[col], f"{col} (Before)", f"plots/eda_gen/outliers/{col}_ecdf_before.png")

        # Удаление выбросов по фиксированным квантилям
        Q1 = df[col].quantile(0.22)
        Q3 = df[col].quantile(0.68)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        method = "IQR"

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"🧨 Удаляется выбросов по {col}: {outliers.shape[0]} строк (метод: {method}, множитель=1.5)")
        print(f"  → Квантили: Q1=0.22, Q3=0.68")

        outliers.to_csv(f"plots/eda_gen/outliers/removed_{col}.csv", index=False)
        df = df[(df[col] >= lower) & (df[col] <= upper)]

        # После удаления выбросов
        fig = plt.figure(figsize=(24, 5))
        spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

        ax0 = fig.add_subplot(spec[0, 0])
        sns.histplot(df[col], kde=True, bins=30, color="lightgreen", ax=ax0)
        ax0.set_title(f"{col} — Histogram (After)")

        ax1 = fig.add_subplot(spec[0, 1])
        sns.boxplot(y=df[col], color="lightblue", ax=ax1)
        ax1.set_title(f"{col} — Boxplot (After)")

        ax2 = fig.add_subplot(spec[0, 2])
        stats.probplot(df[col], dist="norm", plot=ax2)
        ax2.set_title(f"{col} — QQ Plot (After)")

        ax3 = fig.add_subplot(spec[0, 3])
        ax3.text(0.05, 0.5, get_stats_text(df[col]), fontsize=12, verticalalignment='center')
        ax3.axis("off")
        ax3.set_title("Stats (After)")

        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/outliers/{col}_after_outliers.png")
        plt.close()

        # ECDF после
        plot_ecdf(df[col], f"{col} (After)", f"plots/eda_gen/outliers/{col}_ecdf_after.png")

    # === 2.2: Dummy Regressor ===
    for target in target_cols:
        print(f"\n📊 Dummy Regressor Report — {target}")

        y = df[target]
        X = df.drop(columns=forbidden_cols, errors="ignore").select_dtypes(include="number")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")

        # Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", color="red")
        plt.xlabel("True Values")
        plt.ylabel("Predicted")
        plt.title(f"Dummy: {target}")
        plt.tight_layout()
        plt.savefig(f"plots/dummy/{target}_scatter.png")
        plt.close()

        # Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=30, color="skyblue", edgecolor="black")
        plt.axvline(0, color="red", linestyle="--")
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title(f"Residuals: {target}")
        plt.tight_layout()
        plt.savefig(f"plots/dummy/{target}_residuals.png")
        plt.close()

    # === 2.3: Stripplot по бинарным классам ===
    for hue_target in binary_targets:
        for reg_col in target_cols:
            plt.figure(figsize=(8, 4))
            sns.stripplot(
                x=df[hue_target].astype(str),
                y=df[reg_col],
                jitter=True,
                alpha=0.5,
                palette="Set2",
                edgecolor="gray",
                linewidth=0.3
            )
            plt.title(f"{reg_col} по классам {hue_target}")
            plt.xlabel(hue_target)
            plt.ylabel(reg_col)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"plots/eda_gen/targets/strip/{reg_col}_by_{hue_target}.png")
            plt.close()







    # === ЭТАП 3: Анализ бинарных классификационных меток ===
    print("\n=== ЭТАП 3: Анализ бинарных классификационных меток ===")




    # === Бинарные таргеты ===
    binary_targets = ["IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"]

    # === ЭТАП 3.1: Распределение классов ===
    for col in binary_targets:
        counts = df[col].value_counts().sort_index()
        percentages = counts / counts.sum() * 100

        plt.figure(figsize=(5, 4))
        ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette="Set2")
        for i, (v, p) in enumerate(zip(counts.values, percentages.values)):
            ax.text(i, v + 2, f"{v} ({p:.1f}%)", ha="center", fontsize=10)
        plt.title(f"Распределение классов: {col}")
        plt.xlabel("Класс")
        plt.ylabel("Количество")
        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/classification_targets/{col}_distribution.png")
        plt.close()

        print(f"📊 {col}:")
        for cls in counts.index:
            print(f"  Класс {cls}: {counts[cls]} ({percentages[cls]:.2f}%)")
        print("-" * 50)

    # === ЭТАП 3.2: Корреляция между бинарными метками ===
    plt.figure(figsize=(6, 5))
    corr = df[binary_targets].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Корреляции между бинарными метками")
    plt.tight_layout()
    plt.savefig("plots/eda_gen/classification_targets/analysis/binary_targets_corr_heatmap.png")
    plt.close()

    # === ЭТАП 3.3: Violinplot по MolLogP / SI_gt_8 ===
    if "MolLogP" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x="SI_gt_8", y="MolLogP", data=df, palette="Set2")
        plt.title("Распределение MolLogP по классам SI_gt_8")
        plt.tight_layout()
        plt.savefig("plots/eda_gen/classification_targets/analysis/MolLogP_by_SI_gt_8.png")
        plt.close()

    # === ЭТАП 3.4: DummyClassifier sanity-check ===
    for target in binary_targets:
        print(f"\n📌 DummyClassifier: {target}")

        # Защита от утечек
        forbidden_cols = binary_targets + [
            "IC50_nM", "CC50_nM", "SI_corrected",
            "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"
        ]
        X = df.drop(columns=forbidden_cols, errors="ignore").select_dtypes(include="number")
        y = df[target]

        # Stratified split
        stratify_y = y if y.nunique() >= 2 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_y
        )

        # Dummy model
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # === Отчет ===
        print(classification_report(y_test, y_pred, zero_division=0))

        # === Матрица ошибок ===
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"DummyClassifier — {target}")
        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/classification_targets/dummy_reports/{target}_conf_matrix.png")
        plt.close()





    # === ЭТАП 4: Feature Engineering и фильтрация признаков ===

    print("\n=== ЭТАП 4: Feature Engineering и фильтрация признаков ===")


    # === Целевые переменные (не признаки) ===
    target_cols = [
        "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI",
        "IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"
    ]

    # === Feature Engineering ===
    print("\n📌 Добавляем новые признаки:")
    if "MaxEStateIndex" in df.columns and "MinEStateIndex" in df.columns:
        df["EState_Delta"] = df["MaxEStateIndex"] - df["MinEStateIndex"]
        print("✅ EState_Delta")
    if "NumHAcceptors" in df.columns and "NumHDonors" in df.columns:
        df["HAcceptors_to_HDonors_Ratio"] = df["NumHAcceptors"] / (df["NumHDonors"] + 1e-6)
        print("✅ HAcceptors_to_HDonors_Ratio")
    if "MolLogP" in df.columns:
        df["MolLogP_sq"] = df["MolLogP"] ** 2
        print("✅ MolLogP_sq")
    if "MolWt" in df.columns and "TPSA" in df.columns:
        df["MolWt_x_TPSA"] = df["MolWt"] * df["TPSA"]
        print("✅ MolWt_x_TPSA")

    # === Предварительный отбор числовых признаков ===
    X = df.select_dtypes(include=[np.number]).drop(columns=target_cols, errors="ignore")
    print(f"\n🔢 Числовых признаков до фильтрации: {X.shape[1]}")

    # === Удаление признаков с >30% NaN ===
    nan_ratio = X.isna().mean()
    nan_features = nan_ratio[nan_ratio > 0.3].index.tolist()
    print(f"⚠️ Признаков с >30% NaN: {len(nan_features)}")

    # === Константные признаки ===
    constant_features = X.columns[X.nunique(dropna=False) <= 1].tolist()
    print(f"❌ Константные признаки: {len(constant_features)}")

    # === Низковариативные признаки (std < 0.01) ===
    low_variance_features = X.columns[X.std() < 0.01].tolist()
    print(f"⚠️ Низковариативные признаки (< 0.01): {len(low_variance_features)}")

    # === Финальный список к удалению ===
    bad_features = sorted(set(nan_features + constant_features + low_variance_features))
    print(f"🧹 Удаляется всего: {len(bad_features)} признаков")

    with open("data/eda_gen/features/features_to_remove_preliminary.txt", "w") as f:
        for feat in bad_features:
            f.write(f"{feat}\n")

    # === Удаляем признаки из датафрейма ===
    df = df.drop(columns=bad_features, errors="ignore")
    print(f"✅ После удаления: {df.shape[1]} колонок")

    # === Визуализация std распределений ===
    plt.figure(figsize=(10, 4))
    X_std = df.select_dtypes(include=[np.number]).drop(columns=target_cols, errors="ignore").std()
    sns.histplot(X_std, bins=30, kde=True)
    plt.axvline(0.01, color='red', linestyle='--', label='Порог 0.01')
    plt.title("Распределение стандартного отклонения признаков")
    plt.xlabel("Std")
    plt.ylabel("Частота")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/eda_gen/features/std_distribution.png")
    plt.close()

    # === Heatmap пропусков ===
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Карта пропущенных значений")
    plt.tight_layout()
    plt.savefig("plots/eda_gen/features/missing_heatmap.png")
    plt.close()

    # === Импутация NaN средним ===
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df[num_cols].isnull().any().any():
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])
        print("✅ Пропущенные значения числовых признаков заполнены медианой.")
    else:
        print("✅ Пропущенных значений в числовых признаках нет.")

    # === Сохраняем очищенные данные ===
    df.to_csv("data/eda_gen/data_clean_pruned.csv", index=False)
    print("💾 Сохранено: data/eda_gen/data_clean_pruned.csv")

    # === Финальный список признаков ===
    remaining_features = df.drop(columns=target_cols, errors='ignore').select_dtypes(include='number').columns.tolist()
    print(f"🔎 Признаков после фильтрации: {len(remaining_features)}")

    # Проверка удалённых признаков
    print("\n🧾 Проверка удалённых признаков:")
    with open("data/eda_gen/features/features_to_remove_preliminary.txt") as f:
        bad_features = [line.strip() for line in f]
    print(f"Удалено признаков: {len(bad_features)}")
    for col in bad_features[:10]:
        print(f"  • {col}")










    # === ЭТАП 5: Автоматический отбор признаков по MI с визуализацией ===
    print("\n=== ЭТАП 5: Автоматический отбор признаков по MI с визуализацией ===")



    # === Загрузка данных ===
    df = pd.read_csv("data/eda_gen/data_clean_pruned.csv")

    # === Словари задач ===
    tasks = {
        "reg_log1p_IC50_nM": df["log1p_IC50_nM"],
        "reg_log1p_CC50_nM": df["log1p_CC50_nM"],
        "reg_log1p_SI": df["log1p_SI"],
        "clf_IC50_gt_median": df["IC50_gt_median"],
        "clf_CC50_gt_median": df["CC50_gt_median"],
        "clf_SI_gt_median": df["SI_gt_median"],
        "clf_SI_gt_8": df["SI_gt_8"]
    }

    forbidden_cols = [
        "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI",
        "IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8",
        "IC50_nM", "CC50_nM", "SI_corrected"
    ]

    # === Признаки без таргетов ===
    X = df.drop(columns=forbidden_cols, errors="ignore")
    print(f"✅ Число признаков для MI: {X.shape[1]}")

    # === Расчёт MI по всем задачам ===
    mi_all = {}
    for task_name, y in tasks.items():
        is_clf = task_name.startswith("clf_")
        mi = mutual_info_classif(X, y, random_state=42) if is_clf else mutual_info_regression(X, y, random_state=42)

        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        mi_all[task_name] = mi_series

        cumulative_mi = mi_series.cumsum() / mi_series.sum()
        optimal_k = (cumulative_mi < 0.95).sum() + 1
        top_features = mi_series.head(optimal_k)

        # 💾 Сохраняем
        top_features.to_csv(f"data/eda_gen/features/topMI_{task_name}.csv", header=["mutual_info"])
        with open(f"data/eda_gen/features/{task_name}.txt", "w") as f:
            f.writelines([f"{feat}\n" for feat in top_features.index])

        # 📊 Top-25 Barplot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_features.head(25).values, y=top_features.head(25).index, palette="viridis")
        plt.title(f"Top-25 MI для: {task_name}")
        plt.xlabel("Mutual Information")
        plt.ylabel("Признаки")
        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/feature_importance/top25_{task_name}.png")
        plt.close()

        # 📈 Кумулятивный график
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_mi)+1), cumulative_mi.values, marker='o')
        plt.axhline(0.95, color='r', linestyle='--', label='95% порог')
        plt.axvline(optimal_k, color='g', linestyle='--', label=f"K = {optimal_k}")
        plt.title(f"Кумулятивная важность MI для: {task_name}")
        plt.xlabel("Число признаков")
        plt.ylabel("Кумулятивная доля MI")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/feature_importance/cumulative_MI_{task_name}.png")
        plt.close()

        print(f"📌 {task_name}: выбрано {optimal_k} признаков. ✅")

    # === Сводка по количеству признаков ===
    print("\n=== Сводка по количеству отобранных признаков ===")
    for task_name, mi_series in mi_all.items():
        k = (mi_series.cumsum() / mi_series.sum() < 0.95).sum() + 1
        print(f"{task_name}: {k} признаков")

    # === Печать top-25 признаков ===
    print("\n=== ТОП-25 признаков по MI для каждой задачи ===")
    for task_name, mi_series in mi_all.items():
        print(f"\n📌 {task_name} — top 25 признаков:")
        for i, (feat, score) in enumerate(mi_series.head(25).items(), 1):
            print(f"{i:2d}. {feat:<30} → MI = {score:.4f}")

    # === Общий рейтинг по средней MI ===
    mi_df = pd.DataFrame(mi_all)

    for col in forbidden_cols:
        assert col not in mi_df.index, f"🚨 Утечка! Признак {col} попал в итоговую таблицу MI."

    mi_df["MI_avg"] = mi_df.mean(axis=1)
    mi_ranked = mi_df.sort_values("MI_avg", ascending=False)

    mi_ranked.to_csv("data/eda_gen/features/mi_rank_all_tasks.csv")
    print("📁 Сохранён общий рейтинг признаков: data/eda_gen/features/mi_rank_all_tasks.csv")

    # === Визуализация топ-20 признаков ===
    top_features = mi_ranked["MI_avg"].head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="mako")
    plt.title("Top-20 признаков по средней взаимной информации (все задачи)")
    plt.xlabel("Средняя Mutual Information")
    plt.ylabel("Признаки")
    plt.tight_layout()
    plt.savefig("plots/eda_gen/feature_importance/overall_MI_ranking.png")
    plt.close()






    print("\n=== ЭТАП 6: Финальное удаление признаков по высокой корреляции (r > 0.95) с приоритетом MI ===")

    # === ЭТАП 6: Финальное удаление признаков по высокой корреляции (r > 0.95) с приоритетом MI ===


    # === Константы ===
    DATA_PATH = "data/eda_gen/data_clean_pruned.csv"
    FINAL_PATH = "data/eda_gen/data_final.csv"
    FEATURES_DIR = "data/eda_gen/features"
    PLOTS_DIR = "plots/eda_gen/feature_importance"

    # === Загрузка ===
    df = pd.read_csv(DATA_PATH)
    drop_cols = [
        "IC50_nM", "CC50_nM", "SI_corrected",
        "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI",
        "IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    print(f"🔢 Признаков до фильтрации: {X.shape[1]}")

    # === Расчёт MI по всем задачам ===
    mi_df = pd.DataFrame(index=X.columns)
    mi_df["reg_IC50"] = mutual_info_regression(X, df["log1p_IC50_nM"], random_state=42)
    mi_df["reg_CC50"] = mutual_info_regression(X, df["log1p_CC50_nM"], random_state=42)
    mi_df["reg_SI"] = mutual_info_regression(X, df["log1p_SI"], random_state=42)
    mi_df["clf_IC50"] = mutual_info_classif(X, df["IC50_gt_median"], random_state=42)
    mi_df["clf_CC50"] = mutual_info_classif(X, df["CC50_gt_median"], random_state=42)
    mi_df["clf_SI"] = mutual_info_classif(X, df["SI_gt_median"], random_state=42)
    mi_df["clf_SI_gt_8"] = mutual_info_classif(X, df["SI_gt_8"], random_state=42)
    mi_df["MI_avg"] = mi_df.mean(axis=1)
    mi_df.to_csv(f"{FEATURES_DIR}/mi_reg_avg.csv")
    print("✅ MI сохранена: mi_reg_avg.csv")

    # === Матрица корреляций ===
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Матрица корреляций (до удаления)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_matrix.png")
    plt.close()

    # === Удаление высоко коррелирующих признаков с приоритетом по MI ===
    high_corr_pairs = [
        (col, row, upper_tri.loc[row, col])
        for col in upper_tri.columns
        for row in upper_tri.index
        if pd.notnull(upper_tri.loc[row, col]) and upper_tri.loc[row, col] > 0.95
    ]
    print(f"🔗 Найдено пар с r > 0.95: {len(high_corr_pairs)}")

    features_to_drop_final = set()
    for a, b, r in high_corr_pairs:
        if a in features_to_drop_final or b in features_to_drop_final:
            continue
        if mi_df.loc[a, "MI_avg"] >= mi_df.loc[b, "MI_avg"]:
            features_to_drop_final.add(b)
        else:
            features_to_drop_final.add(a)
    print(f"🗑️ Удаляем: {len(features_to_drop_final)} признаков")

    with open(f"{FEATURES_DIR}/high_corr_removed_by_MI.txt", "w") as f:
        for feat in sorted(features_to_drop_final):
            f.write(f"{feat}\n")

    # === Обновление датасета ===
    df_final = df.drop(columns=features_to_drop_final, errors="ignore")
    df_final.to_csv(FINAL_PATH, index=False)
    print(f"💾 Сохранено: {FINAL_PATH} ({df_final.shape[1]} признаков)")

    # === Список удалённых признаков с обоснованием ===
    deleted_info = []
    for a, b, r in high_corr_pairs:
        if a in features_to_drop_final:
            kept, dropped = b, a
        elif b in features_to_drop_final:
            kept, dropped = a, b
        else:
            continue
        deleted_info.append({
            "dropped_feature": dropped,
            "kept_feature": kept,
            "correlation": r,
            "MI_dropped": mi_df.loc[dropped, "MI_avg"],
            "MI_kept": mi_df.loc[kept, "MI_avg"],
            "reason": f"r = {r:.3f}, {dropped} < {kept} по MI"
        })
    deleted_df = pd.DataFrame(deleted_info)
    deleted_df.to_csv(f"{FEATURES_DIR}/deleted_features_with_reasons.csv", index=False)

    # === Финальная корреляционная матрица ===
    corr_matrix_final = df_final.drop(columns=drop_cols, errors="ignore").corr().abs()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix_final, cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Матрица корреляций (после удаления)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_matrix_after.png")
    plt.close()

    # === Печать всех удалённых признаков ===
    print("\n📋 Удалённые признаки по MI при r > 0.95:")
    for feat in sorted(features_to_drop_final):
        print(f"• {feat}")






    # === ЭТАП 7: Масштабирование признаков ===

    # === Загрузка ===
    df = pd.read_csv("data/eda_gen/data_final.csv")

    # === Целевые переменные (не масштабируем) ===
    target_cols = [
        "log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI",
        "IC50_gt_median", "CC50_gt_median", "SI_gt_median", "SI_gt_8"
    ]

    # === Только признаки ===
    X = df.drop(columns=target_cols, errors="ignore")
    X_numeric = X.select_dtypes(include=[np.number])

    # === Разделение на бинарные и непрерывные признаки ===
    binary_features = [col for col in X_numeric.columns if set(df[col].dropna().unique()).issubset({0, 1})]
    continuous_features = [col for col in X_numeric.columns if col not in binary_features]

    print(f"🔢 Всего признаков: {X_numeric.shape[1]}")
    print(f"✅ Бинарных признаков: {len(binary_features)}")
    print(f"🔧 Масштабируем непрерывные признаки: {len(continuous_features)}")

    # === Масштабирование ===
    scaler = StandardScaler()
    X_scaled = X_numeric.copy()
    X_scaled[continuous_features] = scaler.fit_transform(X_scaled[continuous_features])

    # === Сохранение ===
    X_scaled.to_csv("data/eda_gen/scaled/X_scaled.csv", index=False)

    df_scaled = pd.concat([X_scaled, df[target_cols]], axis=1)
    df_scaled.to_csv("data/eda_gen/scaled/data_scaled.csv", index=False)

    joblib.dump(scaler, "data/eda_gen/scaled/scaler_reg.pkl")

    print("📁 Сохранено:")
    print("→ data/eda_gen/scaled/X_scaled.csv")
    print("→ data/eda_gen/scaled/data_scaled.csv")
    print("→ data/eda_gen/scaled/scaler_reg.pkl")


    desc = X_scaled.describe().T
    suspects = desc[desc["max"] > 500].index.tolist()
    print(f"⚠️ Подозрительные признаки по max > 500: {suspects}")







    # === ЭТАП 8: Проекции признаков (PCA, UMAP, LDA) по задачам ===


    # === 📦 Загрузка данных ===
    X_scaled = pd.read_csv("data/eda_gen/scaled/X_scaled.csv")
    df = pd.read_csv("data/eda_gen/data_final.csv")

    # === 🧭 Задачи и таргеты ===
    tasks = {
        "reg_log1p_IC50_nM": "log1p_IC50_nM",
        "reg_log1p_CC50_nM": "log1p_CC50_nM",
        "reg_log1p_SI": "log1p_SI",
        "clf_IC50_gt_median": "IC50_gt_median",
        "clf_CC50_gt_median": "CC50_gt_median",
        "clf_SI_gt_median": "SI_gt_median",
        "clf_SI_gt_8": "SI_gt_8"
    }

    # === Глобальная PCA на всех признаках ===
    pca_all = PCA(n_components=100, random_state=42)
    X_pca_all = pca_all.fit_transform(X_scaled)
    explained_var = pca_all.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_var)+1), explained_var, marker='o', label="Объяснённая дисперсия")
    plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='s', label="Накопленная дисперсия")
    plt.axhline(0.95, color='red', linestyle='--', label="95% дисперсии")
    plt.xlabel("Компонента")
    plt.ylabel("Доля дисперсии")
    plt.title("PCA: объяснённая и накопленная дисперсия")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/eda_gen/projections_variance/pca_explained_variance.png")
    plt.close()

    # === Генерация проекций по задачам ===
    for task, target_col in tasks.items():
        feat_path = f"data/eda_gen/features/{task}.txt"
        if not os.path.exists(feat_path):
            print(f"❌ Нет признаков для {task}")
            continue

        with open(feat_path) as f:
            features = [line.strip() for line in f if line.strip() in X_scaled.columns]

        if len(features) < 2:
            print(f"⚠️ Недостаточно признаков для {task}")
            continue

        X_task = X_scaled[features]
        y = df[target_col]
        is_clf = task.startswith("clf_")

        # === PCA ===
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_task)
        cmap = "tab10" if is_clf else "viridis"
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, alpha=0.7)
        plt.colorbar()
        plt.title(f"PCA: {task}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/projections_task/pca/{task}.png")
        plt.close()

        # === UMAP ===
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_task)
        cmap = "tab10" if is_clf else "plasma"
        plt.figure(figsize=(8, 6))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=cmap, alpha=0.7)
        plt.colorbar()
        plt.title(f"UMAP: {task}")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(f"plots/eda_gen/projections_task/umap/{task}.png")
        plt.close()

        # === LDA (только для классификации) ===
        if is_clf:
            y_array = y.values
            n_classes = len(np.unique(y_array))
            max_components = min(len(features), n_classes - 1)
            if max_components < 1:
                continue
            lda = LinearDiscriminantAnalysis(n_components=max_components)
            X_lda = lda.fit_transform(X_task, y_array)
            plt.figure(figsize=(8, 6))
            if max_components == 1:
                plt.scatter(X_lda[:, 0], [0]*len(X_lda), c=y_array, cmap="coolwarm", alpha=0.7)
                plt.yticks([])
            else:
                plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_array, cmap="coolwarm", alpha=0.7)
                plt.ylabel("LDA2")
            plt.xlabel("LDA1")
            plt.title(f"LDA: {task}")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"plots/eda_gen/projections_task/lda/{task}.png")
            plt.close()






    # === ЭТАП 9: Завершение EDA и логирование артефактов ===

    print("\n=== ЭТАП 9: Завершение пайплайна EDA ===")

    # === Загружаем финальные данные ===
    df_final = pd.read_csv("data/eda_gen/data_final.csv")
    X_scaled = pd.read_csv("data/eda_gen/scaled/X_scaled.csv")

    # === Проверка финальных размеров ===
    print(f"✅ Финальный датафрейм: {df_final.shape[0]} строк, {df_final.shape[1]} колонок")
    print(f"✅ Масштабированные признаки: {X_scaled.shape[0]} строк, {X_scaled.shape[1]} признаков")

    # === Список финальных признаков ===
    final_features = X_scaled.columns.tolist()
    with open("data/eda_gen/features/final_feature_list.txt", "w") as f:
        for feat in final_features:
            f.write(f"{feat}\n")
    print("📝 Финальный список признаков сохранён: data/eda_gen/features/final_feature_list.txt")

    # === Лог финальных файлов ===
    print("\n📦 Финальные файлы сохранены:")
    print(" • data/eda_gen/data_clean.csv — после логарифмирования и outlier-очистки")
    print(" • data/eda_gen/data_clean_pruned.csv — после Feature Engineering и удаления плохих признаков")
    print(" • data/eda_gen/data_final.csv — после корреляционной фильтрации")
    print(" • data/eda_gen/scaled/X_scaled.csv — масштабированные признаки")
    print(" • data/eda_gen/features/final_feature_list.txt — отобранные признаки")

    print("\n✅ Этап EDA завершён.")




if __name__ == "__main__":
    main()
