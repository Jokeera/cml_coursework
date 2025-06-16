import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# === Настройка логгера ===
def setup_logging(log_level=logging.INFO):
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)

def get_logger(name: str):
    return logging.getLogger(name)

# === Константы ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA_PREPARED_PATH = os.path.join(DATA_DIR, "data_final.csv")     # для clf_* моделей
FINAL_DATA_CLASSIF = os.path.join(DATA_DIR, "data_final.csv")     # 
FINAL_DATA_REGRESS = os.path.join(DATA_DIR, "data_final_reg.csv") # используется в reg_*.py
X_SCALED_REG_PATH = os.path.join(DATA_DIR, "scaled", "X_scaled_reg.csv")  # используется в reg_*
SCALER_REG_PATH = os.path.join(DATA_DIR, "scaled", "scaler_reg.pkl")      # используется в reg_*


N_SPLITS_CV = 5
RANDOM_STATE = 42

# === Функции ===
def load_prepared_data(path: str = DATA_PREPARED_PATH) -> pd.DataFrame | None:
    logger = get_logger(__name__)
    try:
        df = pd.read_csv(path)
        logger.info(f"Данные успешно загружены из '{path}'. Размер: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Файл не найден: {path}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при чтении файла '{path}': {e}")
        return None

def load_scaled_regression_data():
    logger = get_logger(__name__)
    try:
        X = pd.read_csv(X_SCALED_REG_PATH)
        scaler = joblib.load(SCALER_REG_PATH)
        logger.info(f"Загружены X_scaled и scaler для регрессии: {X.shape}")
        return X, scaler
    except Exception as e:
        logger.error(f"Ошибка при загрузке масштабированных данных для регрессии: {e}")
        return None, None

def save_model_artifacts(artifacts: dict, task_name: str, model_type: str):
    logger = get_logger(__name__)
    try:
        task_model_dir = os.path.join(MODELS_DIR, model_type, task_name)
        os.makedirs(task_model_dir, exist_ok=True)
        logger.info(f"Папка для артефактов задачи '{task_name}': {task_model_dir}")

        for key, artifact in artifacts.items():
            file_path = os.path.join(task_model_dir, f"{task_name}_{key}.joblib")
            if key == 'threshold' and isinstance(artifact, (float, np.floating, int)):
                joblib.dump(artifact, file_path)
                logger.info(f"Значение '{key}' ({artifact:.4f}) сохранено в: {file_path}")
            elif key == 'features' and isinstance(artifact, list):
                joblib.dump(artifact, file_path)
                logger.info(f"Список признаков '{key}' сохранен в: {file_path}")
            elif hasattr(artifact, 'fit') or hasattr(artifact, 'predict') or hasattr(artifact, 'transform'):
                joblib.dump(artifact, file_path)
                logger.info(f"Артефакт '{key}' (тип: {type(artifact).__name__}) сохранен в: {file_path}")
            else:
                logger.warning(f"Артефакт '{key}' с типом {type(artifact).__name__} не является стандартным. Попытка сохранения...")
                try:
                    joblib.dump(artifact, file_path)
                    logger.info(f"Артефакт '{key}' (нестандартный тип: {type(artifact).__name__}) сохранен в: {file_path}")
                except Exception as e_joblib:
                    logger.error(f"Не удалось сохранить '{key}' через joblib: {e_joblib}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении артефактов для задачи '{task_name}': {e}")

def load_model_artifact(task_name: str, model_type: str, artifact_name: str):
    logger = get_logger(__name__)
    file_path = os.path.join(MODELS_DIR, model_type, task_name, f"{task_name}_{artifact_name}.joblib")
    try:
        artifact = joblib.load(file_path)
        logger.info(f"Артефакт '{artifact_name}' успешно загружен из: {file_path}")
        return artifact
    except FileNotFoundError:
        logger.warning(f"Файл артефакта не найден: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при загрузке артефакта '{artifact_name}' из '{file_path}': {e}")
        return None

def save_plot(fig, path: str, logger_msg: str = None):
    try:
        fig.tight_layout()
        fig.savefig(path)
        if logger_msg:
            get_logger(__name__).info(logger_msg)
    except Exception as e:
        get_logger(__name__).error(f"Ошибка при сохранении графика {path}: {e}")
    finally:
        plt.close(fig)

def plot_roc_curve(y_true, y_pred_proba, task_name: str, model_name: str = ""):
    logger = get_logger(__name__)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plot_task_dir = os.path.join(PLOTS_DIR, "classification", task_name)
    os.makedirs(plot_task_dir, exist_ok=True)
    file_path = os.path.join(plot_task_dir, f"{task_name}{'_' + model_name if model_name else ''}_roc_curve.png")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = f'ROC Curve: {task_name.replace("_", " ").title()}'
    if model_name:
        title += f' - {model_name}'
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    try:
        plt.savefig(file_path)
        logger.info(f"ROC-кривая сохранена: {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении ROC-кривой '{file_path}': {e}")
    plt.close()