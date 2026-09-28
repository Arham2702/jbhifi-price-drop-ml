"""Convert notebooks/ADAA_A2_price_drop.py (percent-format cells) into a .ipynb."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "ADAA_A2_price_drop.py"
DST = ROOT / "notebooks" / "ADAA_A2_price_drop.ipynb"


def main() -> None:
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code" and text:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in SRC.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": []}
    nbformat.write(nb, DST)
    print(f"wrote {DST.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
