"""检查 Python 类型、简写形状与条件式接口文档；不执行待检查代码。"""

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("Args:", "Raises:", "Results:", "副作用:")
OLD_SECTIONS = ("1 输入：", "2 输出：", "3 报错（Raises）：")
LONG_AXES = {
    "batch",
    "channels",
    "in_channels",
    "out_channels",
    "height",
    "width",
    "hidden",
    "tokens",
    "patch_features",
    "latent",
    "latent_channels",
    "latent_height",
    "latent_width",
    "validation_batch",
    "dataset_size",
    "state_bytes",
    "pixels",
}


def project_files(
    root: Path,
) -> list[Path]:
    """枚举维护中的 Python 源码、脚本和测试。

    Args:
        root: 项目根目录；只扫描 src、scripts、tests。

    Raises:
        OSError: 文件系统访问失败时透传。

    Results:
        list[Path]: 排序后的 .py 路径，包括本检查脚本，不包含 Notebook 或运行产物。
    """
    return sorted(
        p for folder in ("src", "scripts", "tests") for p in (root / folder).rglob("*.py")
    )


def _own_nodes(
    statements: list[ast.stmt],
) -> Iterator[ast.AST]:
    """遍历本函数的控制流程，避免把嵌套函数的返回值算入外层。

    Args:
        statements: 当前定义的语句列表。

    Results:
        Iterator[ast.AST]: 逐个产生本层节点，跳过嵌套函数、类和 lambda 的内部。
    """
    stack: list[ast.AST] = list(reversed(statements))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _sections(
    doc: str,
) -> tuple[dict[str, str], list[str]]:
    """拆分接口文档，检查段落顺序、重复、空段和过时标题。

    Args:
        doc: ast.get_docstring 得到的文档文字。

    Results:
        tuple[dict[str, str], list[str]]: 各段内容和结构问题，不推断下层调用语义。
    """
    parts: dict[str, list[str]] = {}
    issues: list[str] = []
    active: str | None = None
    order: list[int] = []
    for line in doc.splitlines():
        text = line.strip()
        if text in OLD_SECTIONS:
            issues.append(f"旧标题 {text} 需迁移到新格式")
        if text in SECTIONS:
            if text in parts:
                issues.append(f"重复段落 {text}")
            parts.setdefault(text, [])
            active = text
            order.append(SECTIONS.index(text))
        elif active:
            parts[active].append(line)
    if order != sorted(order):
        issues.append("段落顺序应为 Args、Raises、Results、副作用（只保留适用项）")
    result = {key: "\n".join(lines).strip() for key, lines in parts.items()}
    for title, content in result.items():
        if not content:
            issues.append(f"空段落 {title} 应省略")
        elif re.match(r"^(?:无[；。\s]|无显式|无返回值|无输出|没有|None\s*:)", content):
            issues.append(f"{title} 使用了不存在项目的占位说明，应省略整段")
    return result, issues


def inspect_source(
    source: str,
    name: str,
) -> tuple[list[str], dict[str, int]]:
    """检查类型、适用文档段落和 jaxtyping 维度简称。

    Args:
        source: 完整 Python 源码，不执行其中的代码。
        name: 用于定位问题的文件名。

    Raises:
        SyntaxError: 源码无法解析时透传。

    Results:
        tuple[list[str], dict[str, int]]: 定位问题列表与 functions/classes 数量。
            问题列表: 检查直接参数、返回、raise/assert 和文档结构；不穷尽下层异常或副作用。
            计数: 只统计显式函数、方法和类，不把测试字符串里的示例当作实际定义。
    """
    tree = ast.parse(source, filename=name)
    issues: list[str] = []
    functions = classes = 0
    if not ast.get_docstring(tree):
        issues.append(f"{name}:1: 缺少模块用途说明")
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in {"Float", "Int", "Bool", "Shaped", "UInt8", "Int64", "Float32"}:
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) > 1:
                    shape = node.slice.elts[1]
                    if isinstance(shape, ast.Constant) and isinstance(shape.value, str):
                        axes = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", shape.value))
                        if axes & LONG_AXES:
                            issues.append(
                                f"{name}:{node.lineno}: 维度需用简称 {sorted(axes & LONG_AXES)}"
                            )
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        location = f"{name}:{node.lineno}:{node.name}"
        doc = ast.get_docstring(node) or ""
        if not doc.strip() or doc.splitlines()[0].strip() in SECTIONS:
            issues.append(f"{location}: 缺少总体简介")
        sections, found = _sections(doc)
        issues.extend(f"{location}: {item}" for item in found)
        if isinstance(node, ast.ClassDef):
            classes += 1
            continue
        functions += 1
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [node.args.vararg] if node.args.vararg else []
        arguments += [node.args.kwarg] if node.args.kwarg else []
        explicit = [arg for arg in arguments if arg.arg not in {"self", "cls"}]
        if explicit and "Args:" not in sections:
            issues.append(f"{location}: 缺少 Args:")
        if not explicit and "Args:" in sections:
            issues.append(f"{location}: 无显式输入，应省略 Args:")
        for arg in explicit:
            if arg.annotation is None:
                issues.append(f"{location}: 参数 {arg.arg} 缺少类型")
            if "Args:" in sections and not re.search(
                rf"(?m)^\s*\*{{0,2}}{re.escape(arg.arg)}\s*:", sections["Args:"]
            ):
                issues.append(f"{location}: Args 缺少参数 {arg.arg}")
        if node.returns is None:
            issues.append(f"{location}: 缺少返回类型")
        own = list(_own_nodes(node.body))
        has_result = any(
            (
                isinstance(n, ast.Return)
                and n.value is not None
                and not (isinstance(n.value, ast.Constant) and n.value.value is None)
            )
            or isinstance(n, (ast.Yield, ast.YieldFrom))
            for n in own
        )
        declaration = any(
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and n.value.value is Ellipsis
            for n in node.body
        )
        if has_result and "Results:" not in sections:
            issues.append(f"{location}: 缺少 Results:")
        if not has_result and not declaration and "Results:" in sections:
            issues.append(f"{location}: 无返回值，应省略 Results:")
        if any(isinstance(n, (ast.Raise, ast.Assert)) for n in own) and "Raises:" not in sections:
            issues.append(f"{location}: 缺少 Raises:")
    return issues, {"functions": functions, "classes": classes}


def main() -> int:
    """检查整个项目，并输出本地与 CI 共用的结果。

    Raises:
        OSError: 读取源码失败时透传。
        UnicodeError: 源码编码无法读取时透传。
        SyntaxError: 源码无效时透传，不当作通过。

    Results:
        int: 通过返回 0；存在结构问题返回 1。

    副作用:
        向 stdout 打印 JSON 检查摘要。
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
                "scope": "类型/条件式文档结构；异常与副作用仍需人工核对",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
