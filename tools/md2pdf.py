#!/usr/bin/env python3
"""Render a Markdown deliverable to PDF: markdown -> HTML (document stylesheet) -> headless Chrome.

    .venv/bin/python tools/md2pdf.py docs/ETX_Technical_Design.md docs/ETX_Technical_Design.pdf
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "DejaVu Sans", "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 178mm; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 6pt; line-height: 1.25; }
h2 { font-size: 14.5pt; margin: 22pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #999; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt; }
p { margin: 5pt 0; text-align: left; }
ul, ol { margin: 4pt 0 6pt; padding-left: 18pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: auto; }
th, td { border: 1px solid #999; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eee; font-weight: bold; }
tr { page-break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 8.8pt; }
pre { background: #f4f4f4; border: 1px solid #ccc; padding: 6pt 8pt; font-size: 8.5pt; line-height: 1.35; white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #999; margin: 12pt 0; }
blockquote { margin: 6pt 0; padding-left: 10pt; border-left: 3px solid #bbb; color: #333; }
.meta { color: #333; margin-bottom: 10pt; }
a { color: #0645ad; text-decoration: none; }
"""


def main(src: str, dst: str) -> int:
    text = Path(src).read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists", "smarty"])
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), Path(src).stem)
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{html}</body></html>"
    html_path = Path(dst).with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={Path(dst).resolve()}",
           "--virtual-time-budget=5000", f"file://{html_path.resolve()}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {dst} ({Path(dst).stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
