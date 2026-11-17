"""Part D: эксперимент с тремя learning rate одной CNN.

Запуск из папки проекта: python part_d.py
Быстрая проверка: python part_d.py --epochs 1
Результаты: reports/part_d/. Нужны зависимости из requirements.txt.
"""

import argparse
import copy
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from app.inference import build_model
from train import make_loaders, run_epoch, seed_everything


def experiment(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path("reports/part_d")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    history = []
    print("Device:", device)

    for lr in [0.0001, 0.001, 0.01]:
        # Меняем только learning rate. Данные и начальные веса одинаковые.
        seed_everything(42)
        loaders, classes = make_loaders(Path("data"), batch_size=128)
        model = build_model(len(classes)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_accuracy = -1
        best_epoch = 0
        best_weights = None

        # Без scheduler и early stopping: одинаковое число эпох, постоянный LR.
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, loaders["train"], loss_function, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(model, loaders["val"], loss_function, device)
            history.append([lr, epoch, train_loss, val_loss,
                            train_accuracy * 100, val_accuracy * 100])
            print(f"LR={lr}, epoch={epoch}, validation accuracy={val_accuracy:.3f}")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())

        # Выбираем веса по validation, а test используем только для оценки.
        model.load_state_dict(best_weights)
        _, test_accuracy = run_epoch(model, loaders["test"], loss_function, device)
        results.append([lr, best_epoch, best_accuracy * 100, test_accuracy * 100])

    headers = ["Learning rate", "Best epoch", "Validation accuracy (%)", "Test accuracy (%)"]
    rows = [[lr, epoch, round(val, 2), round(test, 2)] for lr, epoch, val, test in results]
    for filename, columns, data in [
        ("comparison.csv", headers, rows),
        ("history.csv", ["Learning rate", "Epoch", "Train loss", "Validation loss",
                         "Train accuracy (%)", "Validation accuracy (%)"], history),
    ]:
        with (output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(data)

    print("\n" + " | ".join(headers))
    for row in rows:
        print(" | ".join(str(value) for value in row))

    # Два простых графика: сравниваем validation на каждой эпохе.
    for column, label, filename in [
        (5, "Validation accuracy (%)", "accuracy.png"),
        (3, "Validation loss", "loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for lr in [0.0001, 0.001, 0.01]:
            points = [row for row in history if row[0] == lr]
            ax.plot([row[1] for row in points], [row[column] for row in points],
                    marker="o", label=f"LR = {lr}")
        ax.set(xlabel="Epoch", ylabel=label, xticks=range(1, epochs + 1))
        ax.set_ylim(bottom=0)
        if column == 5:
            ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)

    # Победитель определяется по validation, а не по test.
    best = max(results, key=lambda row: row[2])
    explanation = f"""Part D: learning rate experiment
Device: {device}. Epochs per configuration: {epochs}. Seed: 42.
Train / validation / test: 8000 / 2000 / 10000 images.
Same CNN, initial weights, split, normalization, batch size and optimizer.
Only learning rate changes. No scheduler or early stopping.
Best LR by validation accuracy: {best[0]} ({best[2]:.2f}%).
One run per setting; this does not prove the result holds for every random seed.

Как объяснить преподавателю:
Мы сравнили три learning rate: 0.0001, 0.001 и 0.01.
Learning rate определяет размер шага при обновлении весов.
Маленький шаг может обучаться медленно. Большой может мешать сходимости.
Это возможные причины, а фактическое поведение видно на наших графиках.
Accuracy — процент правильных ответов: выше лучше.
Loss — ошибка обучения: ниже лучше, это не процент неправильных ответов.
Лучшую эпоху и настройку выбираем по validation. Test — итоговая оценка.
В таблице сравниваем результаты при одинаковом числе эпох.
Если различие небольшое, для уверенного вывода нужны повторные запуски.
"""
    (output / "explanation.txt").write_text(explanation, encoding="utf-8")
    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    experiment(args.epochs)
