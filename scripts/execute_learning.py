"""用项目解释器显式执行 Notebook，并保留输出和独立预览。"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def export_preview(notebook: nbformat.NotebookNode, source: Path, destination: Path) -> None:
    """导出 Notebook 为 HTML，并把相对链接改为相对 HTML 输出目录。

    1 输入：
        notebook：内存中的 NotebookNode，保留代码执行输出。
        source：原 ipynb 路径，作为本地引用的原始基准。
        destination：HTML 目标路径，父目录按需创建。

    2 输出：
        无返回值；写入 HTML 文件，不执行 Notebook。

    3 报错（Raises）：
        OSError：目录或文件不可写；nbconvert 导出异常原样透传。
    """
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)

    def relocate(match: re.Match[str]) -> str:
        """为一个 href 匹配项保留远程链接或重定位本地相对路径。

        1 输入：
            match：正则匹配，group(1) 为原链接；闭包 source/destination 提供目录基准。

        2 输出：
            替换用的 href 字符串，保留原查询参数与片段。

        3 报错（Raises）：
            ValueError：URL 解析失败时透传；不在此检查目标文件存在性。
        """
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


def main() -> None:
    """按显式命令执行学习 Notebook，并持久化真实输出及独立 HTML。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；写入本地内核配置、Notebook 与 HTML，打印执行位置；本轮注释任务不调用它。

    3 报错（Raises）：
        SystemExit：命令行参数错误；内核、单元执行、Notebook 校验与文件异常透传，finally 保留进度。
    """
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

    def persist() -> None:
        """把闭包中的 Notebook 原子写回当前目标文件。

        1 输入：
            无显式参数；使用对象状态、闭包或命令行上下文。

        2 输出：
            无返回值；先写 .tmp 再替换 ipynb。

        3 报错（Raises）：
            OSError 或 nbformat 序列化错误原样透传。
        """
        temporary = path.with_suffix(".tmp")
        nbformat.write(notebook, temporary)
        temporary.replace(path)

    def before_cell(cell: nbformat.NotebookNode, cell_index: int, **kwargs: Any) -> None:
        """在代码单元执行前打印位置，便于追踪长实验进度。

        1 输入：
            cell：本次待执行单元。
            cell_index：Notebook 中的零起始单元位置。
            kwargs：nbclient 提供的其他回调上下文，接口接收但本函数不使用。

        2 输出：
            无返回值；仅对代码单元打印首行。

        3 报错（Raises）：
            IndexError：空代码单元访问首行时可能透传；当前不为注释新增保护分支。
        """
        if cell.cell_type == "code":
            print(f"Executing code cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    def after_cell(**kwargs: Any) -> None:
        """在单元执行回调中保存 Notebook 当前输出。

        1 输入：
            kwargs：nbclient 附加上下文，本回调通过闭包 notebook 获取状态。

        2 输出：
            无返回值；调用 persist 保留执行进度。

        3 报错（Raises）：
            文件写入或序列化异常由 persist 原样透传。
        """
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
