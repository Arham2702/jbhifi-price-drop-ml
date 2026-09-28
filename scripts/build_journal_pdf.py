"""Render report/journal.md to HTML and PDF (headless Chrome + MathJax).

Usage: .venv/bin/python scripts/build_journal_pdf.py --name "Firstname Lastname" --id 12345678
"""

import argparse
import html
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; max-width: 190mm; margin: auto; color: #111; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 3px; margin-top: 22px; }
h2 { font-size: 13.5pt; margin-top: 18px; } h3 { font-size: 11.5pt; margin-top: 14px; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; text-align: left; }
th { background: #eee; }
code { font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; padding: 8px; font-size: 8.5pt; overflow-x: auto; }
img { max-width: 100%; display: block; margin: 8px auto; }
.title { text-align: center; margin-bottom: 18px; } .title h1 { border: none; font-size: 20pt; margin-bottom: 4px; }
.title p { margin: 2px; }
@page { size: A4; margin: 14mm; }
"""

MATHJAX = """
<script>window.MathJax = {tex: {inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="STUDENTNAME")
    ap.add_argument("--id", default="STUDENTID")
    args = ap.parse_args()

    text = (REPORT / "journal.md").read_text()
    front, body = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S).groups()
    meta = dict(line.split(": ", 1) for line in front.splitlines() if ": " in line)
    meta = {k: v.strip().strip('"') for k, v in meta.items()}
    author = f"{args.name} — {args.id}"

    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    maths = []

    def stash(m):
        s = m.group(0)
        maths.append(rf"\[{s[2:-2]}\]" if s.startswith("$$") else rf"\({s[1:-1]}\)")
        return f"@@MATH{len(maths) - 1}@@"

    # a "$" followed by a digit or space is currency, not the start of maths
    body = re.sub(r"\$\$.+?\$\$|\$(?![\d\s])[^$\n]+?\$", stash, body, flags=re.S)
    html_body = markdown.markdown(body, extensions=["tables", "fenced_code"])
    html_body = re.sub(r"@@MATH(\d+)@@", lambda m: html.escape(maths[int(m.group(1))]), html_body)

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{meta['title']}</title>
<style>{CSS}</style>{MATHJAX}</head><body>
<div class="title"><h1>{meta['title']}</h1><p>{meta.get('subtitle', '')}</p><p><b>{author}</b></p><p>{meta.get('date', '')}</p></div>
{html_body}</body></html>"""
    out_html = REPORT / "journal.html"
    out_html.write_text(page)

    pdf = REPORT / f"{args.name.replace(' ', '')}_{args.id}_2026_UTS_ML_Journal.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={pdf}", out_html.as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {out_html.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
