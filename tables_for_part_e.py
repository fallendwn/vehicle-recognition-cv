import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
with (root / "reports/part_e/comparison.csv").open(encoding="utf-8-sig") as file:
    results = list(csv.DictReader(file))

folder = root / "res_of_parts/part_e"
folder.mkdir(parents=True, exist_ok=True)

rows = []
for row in results:
    rows.append([
        row["Model"],
        f'{int(row["Parameters"]):,}',
        f'{float(row["Test accuracy (%)"]):.2f}',
        f'{float(row["Train + validation (s)"]):.2f}',
    ])

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")
ax.set_title("Part E: CNN vs fully connected network", fontsize=16, weight="bold", pad=18)
table = ax.table(
    cellText=rows,
    colLabels=["Model", "Parameters", "Test accuracy (%)", "Train + validation (s)"],
    colWidths=[0.25, 0.20, 0.25, 0.30],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.2)
for (row, column), cell in table.get_celld().items():
    cell.set_edgecolor("#d5dde5")
    if row == 0:
        cell.set_facecolor("#243b53")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#f0f4f8" if row % 2 else "white")
fig.tight_layout()
fig.savefig(folder / "table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

models = [row["Model"] for row in results]
accuracy = [float(row["Test accuracy (%)"]) for row in results]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(models, accuracy, width=0.5, color=["#3978af", "#e59a3a"])
ax.bar_label(bars, fmt="%.2f%%", padding=5)
ax.set(title="Part E: test accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(folder / "comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(folder / "table.png")
print(folder / "comparison.png")
