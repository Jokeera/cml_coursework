import logging
from utils import setup_logging, get_logger
import subprocess
import os
import sys


# === Настройка логгера ===
setup_logging()
logger = get_logger(__name__)


# === Установка зависимостей ===
requirements_file = "requirements.txt"
logger.info(f"Проверка и установка зависимостей из {requirements_file}...")

if os.path.exists(requirements_file):
    try:
        # Используем sys.executable, чтобы гарантированно использовать pip из текущего виртуального окружения
        command = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info("✅ Зависимости успешно установлены или уже присутствуют.")
        # Если были установлены пакеты, pip выведет информацию в stdout
        if result.stdout:
            logger.debug(f"Вывод pip:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка при установке зависимостей из {requirements_file}:")
        logger.error(e.stderr)
        logger.error("Пожалуйста, устраните ошибку установки и запустите скрипт снова.")
        sys.exit(1) # Прерываем выполнение, если зависимости не установлены
    except FileNotFoundError:
        logger.error("❌ Команда 'pip' не найдена. Убедитесь, что Python и pip установлены и доступны в PATH.")
        sys.exit(1)
else:
    logger.warning(f"⚠️ Файл {requirements_file} не найден. Пропускаю шаг установки зависимостей.")





logger.info("==========================")
logger.info("== Запуск всех задач ✅ ==")
logger.info("==========================")

# === EDA (предобработка) ===
eda_scripts = [
    "eda_general.py",
    # "eda_clf.py",
    # "eda_reg.py"
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
    "report_generator.py",
]

# === Общий список ===
all_scripts = eda_scripts + classification_scripts + regression_scripts + extra_scripts

# === Запуск ===
for script in all_scripts:
    script_path = os.path.join(os.getcwd(), script)  # Полный путь к скрипту
    logger.info(f"🚀 Запуск: {script_path}")
    
    if os.path.exists(script_path):
        try:
            subprocess.run(["python", script_path], check=True)
            logger.info(f"✅ Успешно: {script}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Ошибка при выполнении {script}: {e}")
    else:
        logger.error(f"❌ Скрипт не найден: {script_path}")

logger.info("===========================")
logger.info("== Все задачи завершены ✅ ==")
logger.info("===========================")
