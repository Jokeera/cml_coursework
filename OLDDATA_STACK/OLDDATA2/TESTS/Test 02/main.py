import logging
from utils import setup_logging, get_logger
import subprocess

# === Настройка логгера ===
setup_logging()
logger = get_logger(__name__)

logger.info("==========================")
logger.info("== Запуск всех задач ✅ ==")
logger.info("==========================")

# === EDA (предобработка) ===
eda_scripts = [
    "eda_general.py",
    "eda_clf.py",
    "eda_reg.py"
]

# === Задачи классификации ===
classification_scripts = [
    "clf_ic50_median.py",
    "clf_cc50_median.py",
    "clf_si_median.py",
    "clf_si_gt8.py",
]

# === Задачи регрессии ===
regression_scripts = [
    "reg_ic50.py",
    "reg_cc50.py",
    "reg_si.py",
]

# === Дополнительные проверки и визуализации ===
extra_scripts = [
    "sanity_check.py",
    "plot.py",
    "report_generator.py",
]

# === Общий список ===
all_scripts = eda_scripts + classification_scripts + regression_scripts + extra_scripts

# === Запуск ===
for script in all_scripts:
    logger.info(f"🚀 Запуск: {script}")
    try:
        subprocess.run(["python", script], check=True)
        logger.info(f"✅ Успешно: {script}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка при выполнении {script}: {e}")

logger.info("===========================")
logger.info("== Все задачи завершены ✅ ==")
logger.info("===========================")
