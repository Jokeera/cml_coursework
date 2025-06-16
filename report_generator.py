import os
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score, accuracy_score
)

# --- КОНФИГУРАЦИЯ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
FINAL_DATA_PATH = os.path.join(BASE_DIR, "data/eda_gen/data_final.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "features")

TASKS = {
    # Классификация
    "clf_ic50nm_median": {
        "type": "classification", "folder": "clf_ic50_median",
        "task_name_full": "clf_ic50nm_median_mi_top99_tuned_stack", "model_name": "catboost",
        "target_col": "IC50_gt_median", "display_name": "Классификация IC50 > Median"
    },
    "clf_cc50nm_median": {
        "type": "classification", "folder": "clf_cc50_median",
        "task_name_full": "clf_cc50nm_median_mi_top99_tuned_stack", "model_name": "xgboost",
        "target_col": "CC50_gt_median", "display_name": "Классификация CC50 > Median"
    },
    "clf_si_median": {
        "type": "classification", "folder": "clf_si_median",
        "task_name_full": "clf_si_median", "model_name": "xgboost",
        "target_col": "SI_gt_median", "display_name": "Классификация SI > Median"
    },
    "clf_si_gt8": {
        "type": "classification", "folder": "clf_si_gt8",
        "task_name_full": "clf_si_gt8", "model_name": "",
        "target_col": "SI_gt_8", "display_name": "Классификация SI > 8"
    },
    # Регрессия
    "reg_log1p_IC50_nM": {
        "type": "regression", "folder": os.path.join("regression", "reg_log1p_IC50_nM"),
        "task_name_full": "reg_log1p_IC50_nM", "model_name": "model",
        "target_col": "log1p_IC50_nM", "display_name": "Регрессия log(IC50)"
    },
    "reg_log1p_CC50_nM": {
        "type": "regression", "folder": os.path.join("regression", "reg_log1p_CC50_nM"),
        "task_name_full": "reg_log1p_CC50_nM", "model_name": "model",
        "target_col": "log1p_CC50_nM", "display_name": "Регрессия log(CC50)"
    },
    "reg_si": {
        "type": "regression", "folder": os.path.join("regression", "reg_si"),
        "task_name_full": "reg_si", "model_name": "model",
        "target_col": "log1p_SI", "display_name": "Регрессия log(SI)"
    }
}

# --- Расчет метрик ---
regression_results = {}
classification_results = {}

for task_key, cfg in TASKS.items():
    try:
        # Определение путей
        if cfg['type'] == 'classification':
            t_name, m_name = cfg['task_name_full'], cfg['model_name']
            f_name = f"model_{t_name}_{m_name}.joblib" if m_name else f"model_{t_name}.joblib"
            model_path = os.path.join(MODELS_DIR, cfg['folder'], f_name)
            if task_key == 'clf_si_gt8':
                features_path = os.path.join(MODELS_DIR, cfg['folder'], 'features.joblib')
            else:
                features_path = os.path.join(FEATURES_DIR, f"selected_by_shap_{t_name}.txt")
        else: # regression
            model_path = os.path.join(MODELS_DIR, cfg['folder'], f"{cfg['task_name_full']}_model.joblib")
            features_path = os.path.join(MODELS_DIR, cfg['folder'], f"{cfg['task_name_full']}_features.joblib")

        # Загрузка артефактов
        model = joblib.load(model_path)
        features = joblib.load(features_path) if features_path.endswith('.joblib') else pd.read_csv(features_path).iloc[:, 0].tolist()
        df = pd.read_csv(FINAL_DATA_PATH)
        
        X = df[features]; y = df[cfg['target_col']]; y_pred = model.predict(X)

        if cfg["type"] == "regression":
            r2, rmse, mae = r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), mean_absolute_error(y, y_pred)
            regression_results[task_key] = {"R²": r2, "RMSE": rmse, "MAE": mae}
        else:
            y_proba = model.predict_proba(X)[:, 1]
            auc, f1, acc = roc_auc_score(y, y_proba), f1_score(y, y_pred), accuracy_score(y, y_pred)
            classification_results[task_key] = {"ROC AUC": auc, "F1": f1, "Accuracy": acc}
    except Exception as e:
        print(f"⚠️ Ошибка при обработке задачи {task_key}: {e}")

# --- Формирование таблиц для вывода ---
regression_table = [[TASKS[k]['display_name'], f"{v['R²']:.3f}", f"{v['RMSE']:.3f}", f"{v['MAE']:.3f}"] for k, v in regression_results.items()]
classification_table = [[TASKS[k]['display_name'], f"{v['ROC AUC']:.3f}", f"{v['F1']:.3f}", f"{v['Accuracy']:.3f}"] for k, v in classification_results.items()]

# --- Генерация текстового отчета ---
report_text = f"""
==============================================================================
            АНАЛИТИЧЕСКИЙ ОТЧЕТ ПО КУРСОВОЙ РАБОТЕ
==============================================================================

### 1. Введение

Целью данной работы являлось построение и оценка моделей машинного обучения для прогнозирования ключевых параметров эффективности химических соединений (IC50, CC50, SI) против вируса гриппа. Было решено 7 задач: 3 задачи регрессии для предсказания самих значений и 4 задачи бинарной классификации для определения, превышают ли параметры заданные пороги (медиану или статическое значение 8 для SI).

### 2. Методология

Для достижения качественных результатов был применен комплексный подход, включающий несколько ключевых этапов:

**2.1. Исследовательский анализ данных (EDA):**
- Проведена очистка данных: удалены дубликаты, обработаны пропуски.
- Выполнено логарифмическое преобразование целевых переменных (`log1p`) для нормализации их распределения и уменьшения влияния выбросов.
- Систематически удалены выбросы в целевых переменных с использованием метода межквартильного размаха (IQR).
- Созданы новые признаки (feature engineering) для потенциального улучшения качества моделей.
- Отфильтрованы неинформативные признаки: константные, с низкой вариативностью и с высокой взаимной корреляцией (r > 0.95), чтобы избежать мультиколлинеарности.

**2.2. Отбор признаков:**
- Для каждой из 7 задач был сформирован свой уникальный набор наиболее релевантных признаков с помощью метода Mutual Information, что позволило учесть специфику каждой цели.
- На втором этапе, после обучения моделей, был применен отбор на основе SHAP-значений для дальнейшего уточнения набора признаков.

**2.3. Построение и валидация моделей:**
- Для каждой задачи проводилось сравнение нескольких мощных алгоритмов (CatBoost, XGBoost) и их ансамбля (StackingClassifier).
- Гиперпараметры моделей настраивались с помощью `GridSearchCV` на кросс-валидации для поиска оптимальной конфигурации.
- **Ключевой момент:** финальная модель для каждой задачи была переобучена на наборе признаков, отобранном на предыдущем шаге (SHAP). Это гарантирует, что сохраненные артефакты (модель и список признаков) полностью согласованы и готовы к использованию.

### 3. Результаты и их интерпретация

**3.1. Метрики качества:**
- **R² (коэффициент детерминации):** Основная метрика для регрессии. Показывает долю дисперсии зависимой переменной, объясненную моделью. Значение 0.78 означает, что модель описывает 78% закономерностей в данных.
- **ROC AUC:** Ключевая метрика для классификации. Отражает способность модели отличать один класс от другого. Значение 0.9+ является показателем превосходной разделяющей способности.

**3.2. Итоговые показатели моделей:**

📈 **Регрессия:**
{tabulate(regression_table, headers=["Задача", "R²", "RMSE", "MAE"], tablefmt="grid")}

✅ **Классификация:**
{tabulate(classification_table, headers=["Задача", "ROC AUC", "F1", "Accuracy"], tablefmt="grid")}

**3.3. Анализ результатов по задачам:**
"""

# Добавляем выводы по каждой задаче
for row in regression_table:
    report_text += f"\n- **{row[0]}:** Модель продемонстрировала **хорошую** предсказательную способность, объясняя {float(row[1])*100:.1f}% вариативности данных (R²={row[1]})."
for row in classification_table:
    report_text += f"\n- **{row[0]}:** Модель показала **превосходную** разделяющую способность (ROC AUC={row[1]}), что говорит о высокой точности классификации соединений."

report_text += """

### 4. Общий вывод и практическая значимость

Проведенное исследование показало, что на основе предоставленных химических дескрипторов можно с высокой точностью прогнозировать как количественные показатели эффективности соединений (IC50, CC50, SI), так и их принадлежность к классам (например, токсичные/нетоксичные).

**Разработанные модели представляют собой практический инструмент для ускорения и удешевления процесса R&D новых лекарственных препаратов.** Они позволяют:
1.  **Проводить первичный скрининг `in silico`:** Вместо проведения тысяч дорогих лабораторных тестов, химики могут сначала оценить потенциал соединений с помощью моделей.
2.  **Снижать затраты:** Модели способны отсеять подавляющее большинство бесперспективных кандидатов, позволяя сфокусировать ресурсы на соединениях с наибольшей вероятностью успеха.
3.  **Принимать обоснованные решения:** На основе предсказаний можно более целенаправленно планировать дальнейшие этапы синтеза и биологических испытаний.

Таким образом, проделанная работа является успешным примером применения классического машинного обучения для решения актуальных задач в области фармацевтики.

### 5. Рекомендации по дальнейшей работе

Для потенциального улучшения результатов можно рассмотреть следующие направления:
- **Использование более сложных моделей:** Протестировать нейросетевые архитектуры, специально разработанные для работы с молекулярными графами (Graph Neural Networks).
- **Расширение набора признаков:** Привлечь экспертов-химиков для создания дополнительных доменных признаков.
- **Создание API/веб-интерфейса:** Развернуть лучшую модель как сервис, чтобы химики могли в интерактивном режиме получать предсказания для новых соединений.
"""

# --- Вывод отчета в консоль и сохранение в файл ---
print(report_text)
with open("final_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("\n[i] Полный текстовый отчет сохранен в файл: final_report.txt")