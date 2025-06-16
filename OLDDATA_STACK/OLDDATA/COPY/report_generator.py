# report_generator.py

import os
import logging
import pandas as pd
from utils import setup_logging, get_logger, MODELS_DIR

from markdown import markdown
from weasyprint import HTML

# === Настройка логирования ===
setup_logging()
logger = get_logger(__name__)

logger.info("==========================")
logger.info("== Генерация отчета 📄 ==")
logger.info("==========================")

# === Конфигурация ===
report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)

txt_path = os.path.join(report_dir, "final_model_report.txt")
md_path = os.path.join(report_dir, "final_model_report.md")
pdf_path = os.path.join(report_dir, "final_model_report.pdf")

# === Описание задач ===
tasks = {
    "regression": [
        "reg_ic50", "reg_cc50", "reg_si"
    ],
    "classification": [
        "clf_ic50_median", "clf_cc50_median", "clf_si_median", "clf_si_gt8"
    ]
}

# === Сбор отчета ===
lines = []
lines.append("# Финальный отчет по всем моделям\n")
lines.append("## Задачи регрессии\n")

for task in tasks["regression"]:
    path = os.path.join(MODELS_DIR, "regression", task, f"{task}_metrics.joblib")
    if os.path.exists(path):
        metrics = pd.read_pickle(path)
        lines.append(f"### {task}")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v:.4f}")
        lines.append("")
    else:
        lines.append(f"### {task} — ❌ метрики не найдены\n")

lines.append("## Задачи классификации\n")

for task in tasks["classification"]:
    path = os.path.join(MODELS_DIR, "classification", task, f"{task}_metrics.joblib")
    if os.path.exists(path):
        metrics = pd.read_pickle(path)
        lines.append(f"### {task}")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v:.4f}")
        lines.append("")
    else:
        lines.append(f"### {task} — ❌ метрики не найдены\n")

# === Сохранение .txt и .md ===
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
with open(md_path, "w") as f:
    f.write("\n".join(lines))

# === Конвертация в PDF с помощью weasyprint ===
html_content = markdown("\n".join(lines))
HTML(string=html_content).write_pdf(pdf_path)
logger.info(f"📄 Отчёты сохранены в: {report_dir}")
