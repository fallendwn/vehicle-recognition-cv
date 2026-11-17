import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


folder = Path(__file__).resolve().parent / "reports/part_d"
with (folder / "comparison.csv").open(encoding="utf-8-sig") as file:
    results = list(csv.DictReader(file))

rows = []
for row in results:
    rows.append([
        row["Learning rate"],
        row["Best epoch"],
        f'{float(row["Validation accuracy (%)"]):.2f}',
        f'{float(row["Test accuracy (%)"]):.2f}',
    ])

fig, ax = plt.subplots(figsize=(9, 2.4))
ax.axis("off")
ax.set_title("Part D: learning rate comparison", pad=15)
table = ax.table(
    cellText=rows,
    colLabels=["Learning rate", "Best epoch", "Validation (%)", "Test (%)"],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)
for (row, column), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(weight="bold")
fig.tight_layout()
fig.savefig(folder / "table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

x = list(range(len(results)))
validation = [float(row["Validation accuracy (%)"]) for row in results]
test = [float(row["Test accuracy (%)"]) for row in results]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars1 = ax.bar([i - 0.2 for i in x], validation, width=0.4, label="Validation")
bars2 = ax.bar([i + 0.2 for i in x], test, width=0.4, label="Test")
ax.bar_label(bars1, fmt="%.2f", padding=3)
ax.bar_label(bars2, fmt="%.2f", padding=3)
ax.set_xticks(x, [row["Learning rate"] for row in results])
ax.set(xlabel="Learning rate", ylabel="Accuracy (%)", ylim=(0, 100),
       title="Part D: accuracy comparison")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(folder / "comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(folder / "table.png")
print(folder / "comparison.png")
