"""Convert a markdown file to PDF using markdown + xhtml2pdf."""

import sys
import markdown
from xhtml2pdf import pisa

CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm 2cm 2.2cm;
}
body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 { font-size: 18pt; color: #111; margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13pt; color: #222; margin-top: 18pt; border-bottom: 1px solid #aaa; padding-bottom: 2px; }
h3 { font-size: 11pt; color: #333; margin-top: 12pt; }
h4 { font-size: 10pt; color: #444; }
pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 8px 10px;
    font-size: 8.5pt;
    font-family: Courier, monospace;
    overflow: hidden;
    word-wrap: break-word;
}
code {
    font-family: Courier, monospace;
    font-size: 8.5pt;
    background: #f4f4f4;
    padding: 1px 3px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 9pt;
}
th {
    background: #2c3e50;
    color: white;
    padding: 5px 8px;
    text-align: left;
    font-weight: bold;
}
td {
    padding: 4px 8px;
    border: 1px solid #ccc;
    vertical-align: top;
}
tr:nth-child(even) td { background: #f8f8f8; }
blockquote {
    border-left: 3px solid #888;
    margin: 8pt 0;
    padding: 2pt 10pt;
    color: #555;
    background: #fafafa;
}
a { color: #2980b9; }
hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
"""


def convert(md_path: str, pdf_path: str) -> None:
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "nl2br"],
    )

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<style>{CSS}</style>
</head><body>
{html_body}
</body></html>"""

    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)

    if result.err:
        print(f"ERROR: {result.err}")
        sys.exit(1)
    else:
        print(f"PDF written to: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
