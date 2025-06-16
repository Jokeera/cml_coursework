# === eda_reg.py ===

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import joblib
import shap
import umap.umap_ as umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# === Настройки окружения ===
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Пути ===
INPUT_FILE = "data/Данные_для_курсовои_Классическое_МО.xlsx"
DATA_STAGE1 = "data/interim/reg_data_clean_1.csv"
DATA_STAGE2 = "data/interim/reg_data_clean_2.csv"
DATA_STAGE3 = "data/interim/reg_data_clean_3.csv"
DATA_STAGE4 = "data/interim/reg_data_clean_4.csv"
DATA_STAGE5 = "data/interim/reg_data_clean_5.csv"
DATA_STAGE6 = "data/data_final_reg.csv"
SCALED_CSV = "data/scaled/X_scaled_reg.csv"
SCALER_PKL = "data/scaled/scaler_reg.pkl"

os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/scaled", exist_ok=True)
os.makedirs("data/features", exist_ok=True)
os.makedirs("plots/eda", exist_ok=True)


def main():
    # === ЭТАП 0: Загрузка и очистка ===
    df = pd.read_excel(INPUT_FILE)
    logger.info(f"✅ Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(",", "")
    df = df.drop_duplicates()

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # === ЭТАП 1: Целевые переменные и логарифмирование ===
    df["IC50_nM"] = df["IC50_mM"] * 1e6
    df["CC50_nM"] = df["CC50_mM"] * 1e6
    df["SI_corrected"] = df["CC50_nM"] / df["IC50_nM"]
    df["SI_corrected"] = df["SI_corrected"].replace([np.inf, -np.inf], np.nan)

    for col in ["IC50_nM", "CC50_nM", "SI_corrected"]:
        df[f"log1p_{col}"] = np.log1p(df[col])

    df = df.dropna(subset=["log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI_corrected"])
    df = df.drop(columns=["IC50_mM", "CC50_mM", "SI", "IC50_nM", "CC50_nM", "SI_corrected"], errors="ignore")
    df.rename(columns={"log1p_SI_corrected": "log1p_SI"}, inplace=True)
    df.to_csv(DATA_STAGE1, index=False)

    # === ЭТАП 2: Feature Engineering ===
    if "MaxEStateIndex" in df.columns and "MinEStateIndex" in df.columns:
        df["EState_Delta"] = df["MaxEStateIndex"] - df["MinEStateIndex"]
    if "NumHAcceptors" in df.columns and "NumHDonors" in df.columns:
        df["HAcceptors_to_HDonors_Ratio"] = df["NumHAcceptors"] / (df["NumHDonors"] + 1e-6)
    if "MolLogP" in df.columns:
        df["MolLogP_sq"] = df["MolLogP"] ** 2
    if "MolWt" in df.columns and "TPSA" in df.columns:
        df["MolWt_x_TPSA"] = df["MolWt"] * df["TPSA"]
    df.to_csv(DATA_STAGE2, index=False)

    # === ЭТАП 3: Удаление плохих признаков ===
    targets = ["log1p_IC50_nM", "log1p_CC50_nM", "log1p_SI"]
    X = df.drop(columns=targets)

    nan_features = X.columns[X.isna().mean() > 0.3]
    constant_features = X.columns[X.nunique() <= 1]
    low_var_features = X.columns[X.std() < 0.01]
    bad_features = sorted(set(nan_features).union(constant_features).union(low_var_features))
    logger.info(f"Удаляем плохие признаки: {len(bad_features)} - {bad_features[:10]}{'...' if len(bad_features) > 10 else ''}")
    df = df.drop(columns=bad_features, errors="ignore")

    imp = SimpleImputer(strategy="median")
    df[df.columns] = imp.fit_transform(df)
    df.to_csv(DATA_STAGE3, index=False)

    # === ЭТАП 4: Отбор признаков по MI ===
    for target in targets:
        y = df[target]
        X = df.drop(columns=targets)
        mi = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        k = (mi_series.cumsum() / mi_series.sum() < 0.95).sum() + 1
        top_feats = mi_series.head(k)
        top_feats.to_csv(f"data/features/topMI_{target}.csv")
        with open(f"data/features/{target}.txt", "w") as f:
            for feat in top_feats.index:
                f.write(f"{feat}\n")
        logger.info(f"{target}: выбрано {k} признаков (MI > 0). Примеры: {list(top_feats.index[:5])}")
    df.to_csv(DATA_STAGE4, index=False)

    # === ЭТАП 5: Удаление коррелирующих признаков (r > 0.95) ===
    X = df.drop(columns=targets)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    features_to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    logger.info(f"Удалено коррелирующих признаков: {len(features_to_drop)} - {features_to_drop[:10]}{'...' if len(features_to_drop) > 10 else ''}")
    df = df.drop(columns=features_to_drop)
    df.to_csv(DATA_STAGE5, index=False)

    # === ЭТАП 6: Сохранение финального датасета ===
    df.to_csv(DATA_STAGE6, index=False)
    logger.info(f"✅ Финальный датасет сохранён: {DATA_STAGE6}")

    # === ЭТАП 7: Масштабирование признаков ===
    X = df.drop(columns=targets)
    binary_features = [col for col in X.columns if set(X[col].unique()).issubset({0, 1})]
    scale_features = [col for col in X.columns if col not in binary_features]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[scale_features]), columns=scale_features)
    X_scaled = pd.concat([X_scaled, X[binary_features].reset_index(drop=True)], axis=1)

    X_scaled.to_csv(SCALED_CSV, index=False)
    joblib.dump(scaler, SCALER_PKL)
    logger.info(f"✅ Масштабированные признаки сохранены: {SCALED_CSV}")

    # === ЭТАП 8: Графики распределений ===
    os.makedirs("plots/eda/distributions", exist_ok=True)
    for target in targets:
        if target in df.columns:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[target], kde=True, bins=50)
            plt.title(f"Распределение {target}")
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[target])
            plt.title(f"Boxplot: {target}")
            plt.tight_layout()
            plt.savefig(f"plots/eda/distributions/{target}.png")
            plt.close()

    # === ЭТАП 9: PCA и UMAP проекции ===
    os.makedirs("plots/eda/projections", exist_ok=True)
    X_numeric = X_scaled.select_dtypes(include=np.number)
    pca = PCA(n_components=2, random_state=42)
    umap_model = umap.UMAP(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(X_numeric)
    umap_proj = umap_model.fit_transform(X_numeric)

    for name, proj in zip(["PCA", "UMAP"], [pca_proj, umap_proj]):
        df_proj = pd.DataFrame(proj, columns=["Dim1", "Dim2"])
        df_proj["log1p_IC50_nM"] = df["log1p_IC50_nM"].values
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df_proj, x="Dim1", y="Dim2", hue="log1p_IC50_nM", palette="viridis", alpha=0.7)
        plt.title(f"{name} проекция (окрашено по log1p_IC50_nM)")
        plt.tight_layout()
        plt.savefig(f"plots/eda/projections/{name}_IC50_proj.png")
        plt.close()

    logger.info("🎯 EDA для регрессионных задач завершён.")


if __name__ == '__main__':
    main()
