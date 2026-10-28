"""检查第一方 Python 的类型与中文接口说明；不导入模型，不执行 Notebook。"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_files(root: Path) -> list[Path]:
    """枚举维护中的源码、脚本和测试，排除运行产物。

    1 输入：
        root：项目根目录，只扫描 src、scripts、tests 中的 .py。
    2 输出：
        排序后的 Python 路径列表，包括本检查脚本自身。
    3 报错（Raises）：
        OSError：文件系统访问失败时透传；不调用 Git 或导入可选依赖。
    """
    return sorted(
        p for folder in ("src", "scripts", "tests") for p in (root / folder).rglob("*.py")
    )


def inspect_source(source: str, name: str) -> tuple[list[str], dict[str, int]]:
    """按 AST 检查函数参数、返回类型和定义文档的三个必需部分。

    1 输入：
        source：完整 Python 源码，不执行其中的代码。
        name：报告中显示的相对文件名。
    2 输出：
        问题字符串列表和定义/函数/类数量；没有问题时列表为空。
    3 报错（Raises）：
        SyntaxError：Python 源码无法解析时透传，不能当作检查通过。
    """
    tree = ast.parse(source, filename=name)
    issues: list[str] = []
    functions = classes = 0
    if not ast.get_docstring(tree):
        issues.append(f"{name}:1: 缺少模块用途说明")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        location = f"{name}:{node.lineno}:{node.name}"
        doc = ast.get_docstring(node) or ""
        if not doc.strip() or not doc.splitlines()[0].strip():
            issues.append(f"{location}: 缺少总体简介")
        for section in ("1 输入：", "2 输出：", "3 报错（Raises）："):
            if section not in doc:
                issues.append(f"{location}: 缺少 {section}")
        if isinstance(node, ast.ClassDef):
            classes += 1
            continue
        functions += 1
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            arguments.append(node.args.vararg)
        if node.args.kwarg:
            arguments.append(node.args.kwarg)
        for arg in arguments:
            if arg.arg not in {"self", "cls"} and arg.annotation is None:
                issues.append(f"{location}: 参数 {arg.arg} 缺少类型")
        if node.returns is None:
            issues.append(f"{location}: 缺少返回类型")
    return issues, {"functions": functions, "classes": classes}


def main() -> int:
    """检查整个项目并输出机器可读结果，供本地和 CI 使用。

    1 输入：
        无命令行参数；根目录由脚本位置确定。
    2 输出：
        退出码 0 表示结构检查通过，1 表示存在问题；控制台打印 JSON 摘要。
    3 报错（Raises）：
        OSError、UnicodeError 或 SyntaxError：文件或源码有问题时不捕获，CI 会失败。
    """
    files = project_files(ROOT)
    issues: list[str] = []
    functions = classes = 0
    for path in files:
        found, counts = inspect_source(path.read_text(), path.relative_to(ROOT).as_posix())
        issues.extend(found)
        functions += counts["functions"]
        classes += counts["classes"]
    print(
        json.dumps(
            {
                "files": len(files),
                "functions": functions,
                "classes": classes,
                "issues": issues,
                "scope": "类型/文档结构；不证明数学或语义正确",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
