from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "assignment2_dinov2_report.docx"
BLUE = "203864"
LIGHT_BLUE = "EAF0F8"
LIGHT_GRAY = "F4F5F7"
BORDER = "D9D9D9"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 100, start: int = 110, bottom: int = 100, end: int = 110) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "6")
        node.set(qn("w:color"), BORDER)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        set_cell_shading(cell, BLUE)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9)
    for row_index, row_data in enumerate(rows):
        cells = table.add_row().cells
        for column, value in enumerate(row_data):
            cell = cells[column]
            cell.text = value
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if row_index % 2:
                set_cell_shading(cell, LIGHT_BLUE)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if column == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            for run in paragraph.runs:
                run.font.size = Pt(8.6)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(70, 70, 70)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.add_run(item)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, text, end])


def build_pipeline_diagram(path: Path) -> None:
    canvas = Image.new("RGB", (1890, 430), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        small_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    items = [
        (35, 295, "Input image\n224 x 224"),
        (420, 330, "Frozen DINOv2\nViT-S/14"),
        (850, 330, "384-dimensional\nembedding"),
        (1280, 295, "Linear head\n5 classes"),
        (1620, 240, "Prediction\nexplanation"),
    ]
    for index, (x, width, label) in enumerate(items):
        box = (x, 105, x + width, 285)
        draw.rounded_rectangle(box, radius=12, fill="#EAF0F8", outline="#203864", width=4)
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=8, align="center")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.multiline_text(
            (x + (width - text_width) / 2, 105 + (180 - text_height) / 2),
            label,
            font=font,
            fill="#111111",
            spacing=8,
            align="center",
        )
        if index < len(items) - 1:
            start_x = x + width + 18
            end_x = items[index + 1][0] - 18
            draw.line((start_x, 195, end_x, 195), fill="#203864", width=4)
            draw.polygon(
                [(end_x, 195), (end_x - 18, 184), (end_x - 18, 206)],
                fill="#203864",
            )
    draw.text((460, 325), "pretrained weights remain fixed", font=small_font, fill="#555555")
    draw.text((1260, 325), "trained on the project dataset", font=small_font, fill="#555555")
    canvas.save(path, format="PNG", optimize=True)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(25, 25, 25)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for name, size in (("Title", 25), ("Heading 1", 16), ("Heading 2", 11.5)):
        style = styles[name]
        style.font.name = "Aptos Display" if name != "Normal" else "Aptos"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(8 if name == "Heading 2" else 4)
        style.paragraph_format.space_after = Pt(6)
    styles["Title"].paragraph_format.space_after = Pt(14)
    title_properties = styles["Title"]._element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)
    styles["List Bullet"].font.name = "Aptos"
    styles["List Bullet"].font.size = Pt(9.3)


def page_break(doc: Document) -> None:
    doc.add_page_break()


def build_report() -> None:
    metrics = json.loads((REPORTS / "dinov2_metrics.json").read_text(encoding="utf-8"))
    baseline = metrics["baseline"]
    no_aug = metrics["experiments"]["no_augmentation"]
    with_aug = metrics["experiments"]["with_augmentation"]
    selected = metrics["experiments"][metrics["selected_experiment"]]
    pipeline_path = REPORTS / "dinov2_pipeline.png"
    build_pipeline_diagram(pipeline_path)

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.65)
    section.bottom_margin = Cm(1.55)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)
    section.header_distance = Cm(0.7)
    section.footer_distance = Cm(0.65)
    configure_styles(doc)
    footer = section.footer.paragraphs[0]
    footer_run = footer.add_run("Computer Vision Assignment 2   |   ")
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(90, 90, 90)
    add_page_number(footer)

    # Page 1
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(46)
    title.add_run("Transport Classification with DINOv2")
    title_properties = title._p.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(24)
    run = subtitle.add_run("Computer Vision Assignment 2")
    run.bold = True
    run.font.size = Pt(14)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(30)
    meta.add_run("Astana IT University\nInstructor Nurym Kaiyrkhan\nNovember 2026")

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "This project upgrades a five-class vehicle classifier from a supervised MobileNetV3 "
        "baseline to a DINOv2 visual foundation model. A frozen DINOv2 ViT-S/14 backbone "
        "produces 384-dimensional embeddings, and a linear classification head is trained on "
        "the existing Open Images based dataset. The selected configuration reaches 91.60% "
        "test accuracy and 91.58% macro-F1, improving the Assignment 1 baseline by 9.24 and "
        "9.43 percentage points respectively. The web application reports top-class "
        "probabilities, CPU latency, a patch-relevance heatmap, and nearest training examples."
    )
    add_table(
        doc,
        ["Model", "Test accuracy", "Macro F1", "Mean latency"],
        [
            ["MobileNetV3 baseline", f"{baseline['accuracy']:.2%}", f"{baseline['macro avg']['f1-score']:.2%}", "Not remeasured"],
            ["DINOv2 selected model", f"{selected['test']['accuracy']:.2%}", f"{selected['test']['macro avg']['f1-score']:.2%}", f"{metrics['latency']['mean_ms']:.2f} ms CPU"],
        ],
        [2.35, 1.35, 1.25, 1.45],
    )
    doc.add_paragraph(
        "The main conclusion is that a frozen self-supervised representation substantially "
        "outperforms the earlier task-specific transfer-learning baseline on the same held-out test set."
    )

    # Page 2
    page_break(doc)
    doc.add_heading("1 Introduction", level=1)
    doc.add_paragraph(
        "Assignment 1 established an end-to-end vehicle recognition pipeline for bicycle, bus, "
        "car, motorcycle, and truck images. MobileNetV3 Small was pretrained on ImageNet, its "
        "feature extractor was initially frozen, and the classifier was trained on the project "
        "dataset. The resulting 82.35% test accuracy was useful, but the errors between visually "
        "related road vehicles indicated that a stronger general visual representation could help."
    )
    doc.add_paragraph(
        "Assignment 2 replaces the feature extractor with DINOv2, a self-supervised Vision "
        "Transformer trained to produce general-purpose image and patch features [1]. The goal is "
        "not only to increase accuracy, but also to make the system easier to inspect through "
        "spatial relevance and example-based explanations."
    )
    doc.add_heading("2 Dataset Update", level=1)
    dataset = metrics["dataset"]
    add_table(
        doc,
        ["Split", "Original images", "Saved augmentations", "Role"],
        [
            ["Train", str(dataset["train"]), str(dataset["saved_augmentations"]), "Fit linear head"],
            ["Validation", str(dataset["validation"]), "0", "Checkpoint selection"],
            ["Test", str(dataset["test"]), "0", "Final evaluation only"],
        ],
        [1.35, 1.35, 1.5, 2.3],
    )
    doc.add_paragraph(
        "The original dataset contains 873 cropped vehicle images derived from Open Images V7 "
        "[3]. The split is unchanged from Assignment 1: 644 training, 110 validation, and 119 test "
        "images. This preserves a direct baseline comparison. The training set also has 126 saved "
        "transformations created by horizontal flips, small rotations, crops, and color changes. "
        "Validation and test images are never augmented."
    )
    doc.add_heading("2 1 Data Quality and Disclosure", level=2)
    add_bullets(
        doc,
        [
            "No generated or synthetic images are used, so the synthetic share is 0%.",
            "Saved transformations remain linked to their source through dataset_augmented manifest.json.",
            "The same random seed 42 and unchanged test split are used in all reported experiments.",
            "Class-wise metrics are reported because overall accuracy can hide weak minority classes.",
        ],
    )

    # Page 3
    page_break(doc)
    doc.add_heading("3 DINOv2 Method", level=1)
    doc.add_paragraph(
        "DINOv2 learns visual representations without class labels by combining self-distillation, "
        "masked-image modeling, and regularization at large scale [1]. The official ViT-S/14 "
        "checkpoint has approximately 21 million parameters, uses 14 × 14 pixel patches, and emits "
        "a 384-dimensional class token [2]. The pretrained backbone is frozen in this project. "
        "Only a linear layer with five outputs is optimized, which reduces overfitting risk and "
        "makes training practical on a small dataset."
    )
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(pipeline_path), width=Inches(6.55))
    add_caption(doc, "Figure 1 DINOv2 classification and explanation pipeline")
    doc.add_heading("3 1 Feature Extraction and Classification", level=2)
    doc.add_paragraph(
        "Each image is resized, center-cropped to 224 × 224 pixels, normalized with ImageNet "
        "statistics, and passed through the frozen backbone. The normalized class-token embedding "
        "is given to a linear classifier. Cross-entropy with label smoothing 0.05 is minimized by "
        "AdamW. ReduceLROnPlateau halves the learning rate after validation accuracy stops improving, "
        "and early stopping prevents unnecessary optimization."
    )
    doc.add_heading("3 2 Explainability", level=2)
    doc.add_paragraph(
        "For spatial explanation, each normalized patch token is projected onto the learned weight "
        "vector of the predicted class. The resulting 16 × 16 relevance grid is interpolated to the "
        "display resolution and overlaid on the input. For example-based explanation, cosine "
        "similarity between the query class-token embedding and all original training embeddings "
        "returns the three nearest images. These two views answer complementary questions: where "
        "the classifier found evidence and which training examples have similar representations."
    )

    # Page 4
    page_break(doc)
    doc.add_heading("4 Experimental Design", level=1)
    doc.add_paragraph(
        "Two DINOv2 linear-probe experiments isolate the effect of the saved Assignment 1 "
        "augmentations. Both use the same pretrained backbone, validation split, test split, "
        "optimizer settings, and random seed. Model selection uses validation accuracy only."
    )
    add_table(
        doc,
        ["Setting", "No augmentation", "With augmentation"],
        [
            ["Training examples", "644", "770"],
            ["Saved transformed copies", "0", "126"],
            ["Initial learning rate", "0.003", "0.003"],
            ["Maximum epochs", "150", "150"],
            ["Early stopping patience", "20", "20"],
            ["Best validation accuracy", f"{no_aug['best_validation_accuracy']:.2%}", f"{with_aug['best_validation_accuracy']:.2%}"],
        ],
        [2.4, 1.9, 1.9],
    )
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(REPORTS / "dinov2_ablation_curves.png"), width=Inches(6.25))
    add_caption(doc, "Figure 2 Validation accuracy during the ablation experiments")
    doc.add_paragraph(
        "The augmented run reaches the highest validation accuracy, 94.55%, and is therefore the "
        "preselected deployment checkpoint. The non-augmented run reaches 92.73%. Test metrics are "
        "computed only after this selection rule is fixed."
    )
    doc.add_heading("4 1 Evaluation Measures", level=2)
    doc.add_paragraph(
        "Accuracy measures the proportion of correct predictions. Macro-F1 averages the F1 score "
        "of all five classes with equal weight. Precision, recall, and F1 are also reported per "
        "class. Latency is measured for 30 single-image forward passes after one warm-up pass on "
        "the available CPU environment."
    )

    # Page 5
    page_break(doc)
    doc.add_heading("5 Results", level=1)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(REPORTS / "model_comparison.png"), width=Inches(6.25))
    add_caption(doc, "Figure 3 MobileNetV3 baseline and DINOv2 test results")
    add_table(
        doc,
        ["Experiment", "Validation accuracy", "Test accuracy", "Macro F1"],
        [
            ["MobileNetV3 baseline", "79.09%", f"{baseline['accuracy']:.2%}", f"{baseline['macro avg']['f1-score']:.2%}"],
            ["DINOv2 no augmentation", f"{no_aug['best_validation_accuracy']:.2%}", f"{no_aug['test']['accuracy']:.2%}", f"{no_aug['test']['macro avg']['f1-score']:.2%}"],
            ["DINOv2 with augmentation", f"{with_aug['best_validation_accuracy']:.2%}", f"{with_aug['test']['accuracy']:.2%}", f"{with_aug['test']['macro avg']['f1-score']:.2%}"],
        ],
        [2.25, 1.45, 1.25, 1.15],
    )
    doc.add_paragraph(
        "The selected DINOv2 model improves test accuracy from 82.35% to 91.60% and macro-F1 "
        "from 82.15% to 91.58%. This corresponds to gains of 9.24 and 9.43 percentage points. "
        "The non-augmented ablation produces the highest test accuracy, 93.28%, even though its "
        "validation accuracy is lower. This disagreement suggests that some saved transformations "
        "shift the training distribution toward the validation set without improving the held-out "
        "test distribution. It does not justify selecting the non-augmented model after observing "
        "test labels; instead, it motivates revisiting augmentation strength in future work."
    )
    class_rows = []
    for class_name in dataset["classes"]:
        values = selected["test"][class_name]
        class_rows.append(
            [
                class_name,
                f"{values['precision']:.2%}",
                f"{values['recall']:.2%}",
                f"{values['f1-score']:.2%}",
                str(int(values["support"])),
            ]
        )
    add_table(doc, ["Class", "Precision", "Recall", "F1", "Support"], class_rows, [1.5, 1.2, 1.2, 1.1, 0.9])

    # Page 6
    page_break(doc)
    doc.add_heading("6 Explainability and Application", level=1)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(
        str(REPORTS / "dinov2_explainability_examples.png"), width=Inches(6.45)
    )
    add_caption(doc, "Figure 4 Example patch-relevance overlays from the deployed classifier")
    doc.add_paragraph(
        "The examples show that the model generally emphasizes the vehicle body and characteristic "
        "shape rather than a single background region. A heatmap is not a causal proof of the "
        "decision and can remain visually plausible when a prediction is wrong, so it must be "
        "interpreted together with confidence and retrieved neighbors."
    )
    doc.add_heading("6 1 Interactive Web Application", level=2)
    add_bullets(
        doc,
        [
            "Accepts JPG, PNG, and WebP uploads up to 10 MB and supports mobile camera capture.",
            "Returns the predicted class and ranked probabilities for all five classes.",
            "Displays measured inference latency and the execution device for each request.",
            "Shows a patch-relevance overlay and three nearest original training images.",
            "Exposes a health endpoint with model, device, class list, and loading status.",
        ],
    )
    doc.add_heading("6 2 Runtime Performance", level=2)
    doc.add_paragraph(
        f"On the available CPU, mean single-image latency is {metrics['latency']['mean_ms']:.2f} ms, "
        f"the median is {metrics['latency']['median_ms']:.2f} ms, and the 95th percentile is "
        f"{metrics['latency']['p95_ms']:.2f} ms over {metrics['latency']['samples']} measured images. "
        "The browser request also includes image decoding, heatmap construction, and retrieval, so "
        "end-to-end response time can be slightly higher."
    )

    # Page 7
    page_break(doc)
    doc.add_heading("7 Ethics and Limitations", level=1)
    add_bullets(
        doc,
        [
            "Dataset bias remains possible because Open Images reflects internet photography and may underrepresent local vehicles, weather, viewpoints, and camera quality.",
            "The system predicts only five known classes. Unknown objects are forced into one of them, so confidence must not be treated as an open-set guarantee.",
            "Nearest-neighbor examples may expose source images. A public deployment should retain attribution and confirm that redistribution follows the source licenses.",
            "No faces or personal identity attributes are processed intentionally. Incidental people or license plates may still occur in road images and should be removed for public release.",
            "No synthetic media is present. If generated images are added later, they must be labeled, counted, and separated during evaluation.",
        ],
    )
    doc.add_heading("8 Reproducibility", level=1)
    doc.add_paragraph(
        "Install requirements.txt, prepare the dataset folders, and run python train_dinov2.py. "
        "The script downloads the official DINOv2 ViT-S/14 checkpoint through PyTorch Hub, extracts "
        "features, trains both ablation heads, writes metrics and figures, and saves the selected "
        "linear head plus retrieval index. Run uvicorn app.main:app --reload and open "
        "http://127.0.0.1:8000. The repository records the seed, split sizes, hyperparameters, "
        "histories, class-wise metrics, and latency protocol."
    )
    doc.add_heading("9 Conclusion", level=1)
    doc.add_paragraph(
        "DINOv2 provides a clear improvement over MobileNetV3 on the existing vehicle dataset while "
        "requiring only a small trained head. The selected model gains more than nine percentage "
        "points in both accuracy and macro-F1 and remains fast enough for an interactive CPU demo. "
        "The ablation also shows that augmentation should be validated carefully rather than assumed "
        "to help. Future work should add open-set rejection, collect local adverse-condition images, "
        "and evaluate calibration before practical use."
    )
    doc.add_heading("References", level=1)
    references = [
        "[1] M. Oquab et al. DINOv2 Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research 2024. https://arxiv.org/abs/2304.07193",
        "[2] Meta AI Research. DINOv2 official repository and model card. https://github.com/facebookresearch/dinov2",
        "[3] A. Kuznetsova et al. The Open Images Dataset V4. International Journal of Computer Vision 2020. https://storage.googleapis.com/openimages/web/index.html",
        "[4] A. Howard et al. Searching for MobileNetV3. IEEE ICCV 2019. https://arxiv.org/abs/1905.02244",
    ]
    for reference in references:
        paragraph = doc.add_paragraph(reference)
        paragraph.paragraph_format.left_indent = Cm(0.45)
        paragraph.paragraph_format.first_line_indent = Cm(-0.45)
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(8.3)

    doc.core_properties.title = "Transport Classification with DINOv2"
    doc.core_properties.subject = "Computer Vision Assignment 2"
    doc.core_properties.keywords = "DINOv2, vehicle classification, explainability, computer vision"
    doc.save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    build_report()
