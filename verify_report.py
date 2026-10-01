from pathlib import Path

from docx import Document


path = Path("reports/transport_classification_report.docx")
document = Document(path)
text = "\n".join(paragraph.text for paragraph in document.paragraphs)
text += "\n" + "\n".join(
    cell.text
    for table in document.tables
    for row in table.rows
    for cell in row.cells
)
page_breaks = sum(
    len(run._r.xpath('.//w:br[@w:type="page"]'))
    for paragraph in document.paragraphs
    for run in paragraph.runs
)
headings = [
    paragraph.text
    for paragraph in document.paragraphs
    if paragraph.style.name.startswith("Heading")
]

print(
    {
        "paragraphs": len(document.paragraphs),
        "tables": len(document.tables),
        "inline_shapes": len(document.inline_shapes),
        "sections": len(document.sections),
        "page_breaks": page_breaks,
        "double_hyphen": text.count("--"),
        "em_dash": text.count("—"),
        "en_dash": text.count("–"),
        "replacement": text.count("�"),
        "headings": headings,
        "characters": len(text),
    }
)

assert "--" not in text
assert "—" not in text
assert "–" not in text
assert "�" not in text
assert page_breaks == 5
assert len(document.inline_shapes) == 3
assert len(document.tables) == 5
assert path.stat().st_size > 100_000
assert "References" not in headings
assert "Transport Type Classification | Computer Vision Course Assignment" not in text
assert "Student: Li Denis SE-2425" in text
