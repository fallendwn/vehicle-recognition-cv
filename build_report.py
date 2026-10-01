from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "transport_classification_report.docx"
CLASSES = ["bicycle", "bus", "car", "motorcycle", "truck"]


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = margins.find(qn(f"w:{name}"))
        if element is None:
            element = OxmlElement(f"w:{name}")
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        set_cell_shading(cell, "DCEBE6")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.font.bold = False
            run.font.size = Pt(8.5)
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = value
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in cells[index].paragraphs[0].runs:
                run.font.size = Pt(8.5)
    for row in table.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for index, cell in enumerate(row.cells):
            set_cell_margins(cell)
            if widths:
                cell.width = widths[index]
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run(text)


def add_body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.line_spacing = 1.08


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])


def add_page_break(document: Document) -> None:
    document.add_page_break()


def fmt_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def main() -> None:
    manifest = json.loads(
        (ROOT / "dataset_augmented" / "manifest.json").read_text(encoding="utf-8")
    )
    base_metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    fine_metrics = json.loads(
        (REPORTS / "metrics_finetuned.json").read_text(encoding="utf-8")
    )

    train_counts = {name: manifest["classes"][name]["original"] for name in CLASSES}
    aug_counts = {name: manifest["classes"][name]["augmented"] for name in CLASSES}
    val_counts = {name: len(list((ROOT / "dataset" / "val" / name).glob("*"))) for name in CLASSES}
    test_counts = {name: len(list((ROOT / "dataset" / "test" / name).glob("*"))) for name in CLASSES}

    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(1.65)
    section.bottom_margin = Cm(1.45)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)

    title_style = styles["Title"]
    title_style.font.name = "Arial"
    title_style.font.size = Pt(22)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 0)

    for style_name, size in (("Heading 1", 14), ("Heading 2", 11)):
        style = styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(9)
        style.paragraph_format.space_after = Pt(5)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(155)
    title.paragraph_format.space_after = Pt(9)
    title.add_run("Transport Type Classification")

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(22)
    run = subtitle.add_run("Computer Vision Course Assignment 1")
    run.font.name = "Arial"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0, 0, 0)

    student = document.add_paragraph()
    student.alignment = WD_ALIGN_PARAGRAPH.CENTER
    student.paragraph_format.space_after = Pt(0)
    run = student.add_run("Student: Li Denis SE-2425")
    run.font.name = "Arial"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0, 0, 0)

    add_page_break(document)

    add_heading(document, "1 Objective and dataset")
    add_body(
        document,
        "This project classifies a photograph as bicycle, bus, car, motorcycle or truck. "
        "A MobileNetV3 Small network provides the classifier, and a FastAPI web page accepts "
        "an uploaded image and returns the predicted class with confidence scores.",
    )
    add_body(
        document,
        "The source dataset contains 873 readable images. The latest additions made the training "
        "classes uneven, especially bicycle and car. I kept validation and test separate and "
        "balanced only the data used to update the model. A SHA256 check found no exact file shared "
        "between different splits. One identical pair remains inside the car test folder, so the "
        "reported test score is an internal project result rather than a clean benchmark.",
    )

    dataset_rows = []
    for name in CLASSES:
        total = train_counts[name] + val_counts[name] + test_counts[name]
        dataset_rows.append(
            [
                name,
                str(total),
                str(train_counts[name]),
                str(aug_counts[name]),
                str(train_counts[name] + aug_counts[name]),
                str(val_counts[name]),
                str(test_counts[name]),
            ]
        )
    add_table(
        document,
        ["Class", "Original total", "Train", "Saved edits", "Effective train", "Val", "Test"],
        dataset_rows,
    )
    picture = document.add_picture(str(REPORTS / "dataset_balance.png"), width=Inches(6.6))
    picture._inline.graphic.graphicData.pic.nvPicPr.cNvPr.set("descr", "Balanced training image counts by class")
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_page_break(document)
    add_heading(document, "2 Image preparation and saved augmentation")
    add_body(
        document,
        "The new preparation script reads only dataset train. It leaves every source image unchanged "
        "and writes edited JPEG copies to dataset_augmented train. The manifest records the seed, "
        "target count, source path and output path for every generated file. Repeating the script with "
        "the same seed recreates the same set.",
    )
    add_table(
        document,
        ["Operation", "Configuration"],
        [
            ["Balance target", "154 effective training images per class"],
            ["Saved copies", "126 files in a separate augmented directory"],
            ["Crop and resize", "Random edge crop from 3% to 12%, restored to source size"],
            ["Geometry", "Horizontal reflection with probability 0.5 and rotation up to 10 degrees"],
            ["Colour", "Brightness and contrast 0.85 to 1.15, colour 0.90 to 1.10"],
            ["Network input", "Random 224 pixel crop for train, centre crop for validation and test"],
            ["Normalization", "ImageNet RGB mean and standard deviation"],
        ],
        widths=[Cm(4.2), Cm(12.5)],
    )
    add_body(
        document,
        "Saved copies are added through a second ImageFolder and combined with the original training "
        "set. Validation and test load only their original folders. During training, random crop, "
        "reflection, rotation and mild colour jitter are still applied in memory. This gives the model "
        "different views while keeping a visible, reproducible set of edited files for the assignment.",
    )

    add_heading(document, "3 Model")
    add_body(
        document,
        "MobileNetV3 Small starts with ImageNet weights. In the base run, the convolutional feature "
        "extractor is frozen and the final classifier is replaced for five transport classes. The "
        "classifier is trained with AdamW and cross entropy with label smoothing. A second run opens "
        "the last three feature blocks at smaller learning rates. It keeps the base checkpoint when "
        "validation accuracy does not improve.",
    )

    add_page_break(document)
    add_heading(document, "4 Training protocol")
    add_body(
        document,
        "The random seed is 42 and the run used the CPU. Each epoch sees 770 balanced training items. "
        "The validation set selects the checkpoint and the test set is evaluated after that choice. "
        "Early stopping limits unnecessary epochs.",
    )
    history = base_metrics["history"]
    selected_epoch = max(history, key=lambda row: row["validation_accuracy"])["epoch"]
    add_table(
        document,
        ["Training setting", "Value"],
        [
            ["Architecture", "MobileNetV3 Small with ImageNet initialization"],
            ["Seed and device", "42 and CPU"],
            ["Base optimizer", "AdamW with classifier learning rate 0.001"],
            ["Batch size and loss", "32 and cross entropy with 0.05 label smoothing"],
            ["Stopping rule", "Up to 15 epochs, patience 4 on validation accuracy"],
            ["Executed and selected epoch", f"{len(history)} executed, epoch {selected_epoch} selected"],
            ["Fine tuning", "Last 3 feature blocks, 4 executed epochs, no validation gain"],
        ],
        widths=[Cm(5.2), Cm(11.5)],
    )
    picture = document.add_picture(str(REPORTS / "training_curves.png"), width=Inches(6.65))
    picture._inline.graphic.graphicData.pic.nvPicPr.cNvPr.set("descr", "Training and validation curves")
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_body(
        document,
        "Validation accuracy reached its best value on epoch 5. Later training accuracy continued to "
        "rise while validation varied, so the saved model comes from the earlier checkpoint. Fine "
        "tuning did not beat that validation result and therefore retained the same weights.",
    )

    add_page_break(document)
    add_heading(document, "5 Results and error analysis")
    add_body(
        document,
        "The test set contains 119 images. Accuracy is the fraction of correct predictions. Precision "
        "measures how often a predicted class is correct, recall measures how many real examples are "
        "found, and F1 combines both. Macro values give each class the same weight.",
    )
    base_test = base_metrics["test"]
    final_test = fine_metrics["test"]
    add_table(
        document,
        ["Checkpoint", "Accuracy", "Macro precision", "Macro recall", "Macro F1"],
        [
            [
                "Base",
                fmt_percent(base_test["accuracy"]),
                fmt_percent(base_test["macro avg"]["precision"]),
                fmt_percent(base_test["macro avg"]["recall"]),
                fmt_percent(base_test["macro avg"]["f1-score"]),
            ],
            [
                "Fine tuned output",
                fmt_percent(final_test["accuracy"]),
                fmt_percent(final_test["macro avg"]["precision"]),
                fmt_percent(final_test["macro avg"]["recall"]),
                fmt_percent(final_test["macro avg"]["f1-score"]),
            ],
        ],
    )
    picture = document.add_picture(
        str(REPORTS / "confusion_matrix_finetuned.png"), width=Inches(4.45)
    )
    picture._inline.graphic.graphicData.pic.nvPicPr.cNvPr.set("descr", "Confusion matrix for the final checkpoint")
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    class_rows = []
    for name in CLASSES:
        row = final_test[name]
        class_rows.append(
            [
                name,
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1-score']:.3f}",
                str(int(row["support"])),
            ]
        )
    add_table(document, ["Class", "Precision", "Recall", "F1", "Test N"], class_rows)
    add_body(
        document,
        "The final accuracy is 82.35% and macro F1 is 82.15%. The remaining errors are spread across "
        "similar road vehicle classes. The result is useful for checking this project build, but it "
        "does not measure performance on a new camera, location or collection session.",
    )

    add_page_break(document)
    add_heading(document, "6 Web application and limitations")
    add_body(
        document,
        "The web application loads transport_classifier_finetuned.pt when it starts. The health route "
        "confirmed that all five classes loaded. A prediction request with a held out bus image returned "
        "bus with confidence 0.9972. Uploaded JPG, PNG and WebP files are converted to RGB, resized, "
        "centre cropped to 224 pixels and normalized before inference.",
    )
    add_body(
        document,
        "The classifier always chooses one of the five known classes. It has no unknown class and no "
        "object detector, so a photograph without a clear vehicle can still receive a confident label. "
        "The dataset is relatively small and includes images with different sizes and sources. The car "
        "test split also contains one exact duplicate pair. A stronger evaluation would remove that pair "
        "and use a new held out collection captured after all training choices are fixed.",
    )

    add_heading(document, "Reproducibility and deliverables")
    add_body(
        document,
        "Run prepare_augmented_dataset.py before training. Then run train.py and fine_tune.py from the "
        "project environment. Seed 42 controls the split independent saved augmentation. The project "
        "contains the augmented manifest, both checkpoints, JSON metrics, training plots and confusion "
        "matrices. The tested environment used Python 3.11, PyTorch 2.8.0 with CPU support and "
        "torchvision 0.23.0.",
    )

    for section in document.sections:
        footer = section.footer
        paragraph = footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        add_page_number(paragraph)

    all_paragraphs = list(document.paragraphs)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                all_paragraphs.extend(cell.paragraphs)
    for section in document.sections:
        all_paragraphs.extend(section.header.paragraphs)
        all_paragraphs.extend(section.footer.paragraphs)
    for paragraph in all_paragraphs:
        for run in paragraph.runs:
            run.font.color.rgb = RGBColor(0, 0, 0)
            run.font.italic = False

    core_properties = document.core_properties
    core_properties.title = "Transport Type Classification"
    core_properties.subject = "Computer Vision Course Assignment 1"
    core_properties.author = ""
    core_properties.last_modified_by = ""
    core_properties.comments = ""

    document.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
