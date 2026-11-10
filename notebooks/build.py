import sys, nbformat as nbf
src, dst = sys.argv[1], sys.argv[2]
cells, kind, buf = [], None, []
def flush():
    if kind is not None:
        text = "\n".join(buf).strip("\n")
        cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
for line in open(src, encoding="utf-8").read().splitlines():
    if line.startswith("#%%"):
        flush(); kind = "md" if line.strip() == "#%% md" else "code"; buf = []
    else:
        buf.append(line)
flush()
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "accelerator": "GPU", "colab": {"provenance": []}})
nbf.write(nb, dst)
print(len(cells), "cells")
