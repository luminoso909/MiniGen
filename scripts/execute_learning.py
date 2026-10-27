"""Execute a learning notebook with this interpreter and a project-local kernel spec."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def export_preview(notebook, source: Path, destination: Path):
    """Keep local Notebook links valid when the HTML lives in a different directory."""
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)

    def relocate(match):
        value = match.group(1)
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
            return match.group(0)
        target = source.parent / unquote(parsed.path)
        relative = os.path.relpath(target.resolve(), destination.parent)
        href = urlunsplit(("", "", quote(relative), parsed.query, parsed.fragment))
        return f'href="{href}"'

    html = re.sub(r'href="([^"]+)"', relocate, html)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--html-output", type=Path, default=None)
    args = parser.parse_args()
    path = args.notebook.resolve()
    cache = ROOT / ".cache/jupyter"
    kernel_dir = cache / "kernels/minigen-local"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "MiniGen local interpreter",
                "language": "python",
            }
        )
    )
    old_path = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = str(cache) + (os.pathsep + old_path if old_path else "")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(cache / "runtime"))
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    notebook = nbformat.read(path, as_version=4)

    def persist():
        temporary = path.with_suffix(".tmp")
        nbformat.write(notebook, temporary)
        temporary.replace(path)

    def before_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing code cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    def after_cell(**kwargs):
        persist()

    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="minigen-local",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=before_cell,
        on_cell_executed=after_cell,
    )
    try:
        client.execute()
    finally:
        notebook.metadata.kernelspec = {
            "display_name": "MiniGen (project Python)",
            "language": "python",
            "name": "python3",
        }
        persist()
    nbformat.validate(notebook)
    html_path = (
        args.html_output.resolve()
        if args.html_output is not None
        else ROOT / "artifacts/runs/s1_week1/learning_preview.html"
    )
    export_preview(notebook, path, html_path)
    print(json.dumps({"notebook": str(path), "html": str(html_path), "executed": True}), flush=True)


if __name__ == "__main__":
    main()
