"""Простые графики и таблицы для защиты FashionMNIST.

Запуск после обучения: python yeah_plots.py
Другой файл метрик: python yeah_plots.py --metrics path/to/metrics.json
Результаты: reports/yeah_plots/ (PNG, CSV и пояснения explanation.txt).

Файл независимый: не меняет обучение, модель и исходные отчёты.
Использует только matplotlib и стандартную библиотеку Python.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Работает и без открытого графического окна.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent


def save_plot(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_table(output, name, headers, rows):
    """CSV удобно открыть в Excel, PNG — вставить в презентацию."""
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 0.45 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#e7eef6")
            cell.set_text_props(weight="bold")
    save_plot(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=ROOT / "reports/fashion_mnist_metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/yeah_plots")
    args = parser.parse_args()
    if not args.metrics.is_file():
        parser.exit(1, f"Нет файла метрик: {args.metrics}\n"
                    "Сначала запустите python train.py или укажите --metrics.\n"
                    "Графики без реальных результатов не создаются.\n")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    history = metrics["history"]
    report = metrics["test"]
    if not history:
        parser.exit(1, "История обучения пуста: графики по эпохам построить нельзя.\n")
    classes = [name for name, values in report.items()
               if isinstance(values, dict) and name not in ("macro avg", "weighted avg")]
    if not classes:
        parser.exit(1, "В test нет метрик отдельных классов.\n")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    # 1. Accuracy: доля правильных ответов. Чем выше, тем лучше.
    # 2. Loss: ошибка, которую минимизирует обучение. Чем ниже, тем лучше.
    for metric, title, ylabel, scale in [
        ("accuracy", "Точность по эпохам", "Правильные ответы, %", 100),
        ("loss", "Ошибка по эпохам", "Loss", 1),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for split, label in [("train", "Обучение (train)"),
                             ("validation", "Проверка (validation)")]:
            ax.plot(epochs, [row[f"{split}_{metric}"] * scale for row in history],
                    marker="o", label=label)
        ax.set(title=title, xlabel="Эпоха", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        if metric == "accuracy":
            ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save_plot(fig, output, metric)

    # Recall: из всех настоящих изображений класса сколько распознано верно.
    recalls = [report[name]["recall"] * 100 for name in classes]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(classes, recalls, color="#3978af")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in recalls], padding=4)
    ax.invert_yaxis()
    ax.set(xlim=(0, 112), xlabel="Правильно распознано, % (recall)",
           title="Распознавание каждого класса на test", xticks=[0, 20, 40, 60, 80, 100])
    save_plot(fig, output, "class_recall")

    # train.py выбирает checkpoint по максимальной validation accuracy.
    best = max(history, key=lambda row: row["validation_accuracy"])
    total = sum(int(report[name]["support"]) for name in classes)
    save_table(output, "summary", ["Показатель", "Значение"], [
        ["Лучшая эпоха по validation", best["epoch"]],
        ["Точность validation, %", f'{best["validation_accuracy"] * 100:.2f}'],
        ["Точность test, %", f'{report["accuracy"] * 100:.2f}'],
        ["Изображений в test", total],
    ])
    save_table(output, "class_results", ["Класс", "Изображений в test", "Распознано верно, %"], [
        [name, int(report[name]["support"]), f'{report[name]["recall"] * 100:.1f}']
        for name in classes
    ])

    explanation = f"""Источник: {args.metrics.resolve()}

Как объяснить преподавателю:
1. accuracy.png: по горизонтали эпохи — проходы по обучающим данным.
   По вертикали процент правильных ответов. Выше — лучше.
   Train — данные для обучения. Validation — данные для выбора модели.
2. loss.png: величина ошибки, которую модель уменьшает при обучении.
   Ниже — лучше. Это не процент неправильных ответов.
   Рост validation loss при падении train loss может указывать на переобучение.
3. class_recall.png: для каждого настоящего класса показана доля верных ответов.
   Например, recall 90% означает: из 100 вещей этого класса верно распознаны 90.
   Это поясняющий пример, а не результат нашего эксперимента.
4. summary.png / summary.csv: лучшая эпоха, validation accuracy и test accuracy.
   Test — отдельные данные для итоговой оценки выбранной модели.
   Итоговая test accuracy — доля верных ответов среди всех test-изображений.
5. class_results.png / class_results.csv: число test-примеров и recall по классам.

Все числа взяты из указанного JSON. Нового обучения и оценки здесь нет.
Матрица ошибок не восстанавливается из сводных метрик: её сохраняет train.py.
"""
    (output / "explanation.txt").write_text(explanation, encoding="utf-8")
    print(f"Готово: {output.resolve()}")
    print("3 графика, 2 таблицы (PNG + CSV), explanation.txt с пояснениями.")


if __name__ == "__main__":
    main()
