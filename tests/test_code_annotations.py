"""验证条件式接口文档规则，不执行测试字符串中的代码。"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def checker() -> ModuleType:
    """装载 AST 检查模块，供测试显式调用。

    Raises:
        ImportError: 检查器装载失败时透传。
        OSError: 检查器文件不可读时透传。

    Results:
        ModuleType: 含 inspect_source 和 project_files 的检查模块。
    """
    path = Path(__file__).resolve().parents[1] / "scripts/check_code_annotations.py"
    spec = importlib.util.spec_from_file_location("annotation_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_nested_argument_return_and_sections_are_reported(
    checker: ModuleType,
) -> None:
    """验证嵌套函数、关键字参数、可变参数与总体简介缺漏。

    Args:
        checker: AST 检查模块 fixture。

    Raises:
        AssertionError: 缺漏没有被定位，或定义计数不符。
    """
    source = '''"""测试输入模块。"""
class Missing:
    def outer(self, value, *, option, **kwargs):
        def inner(extra):
            return extra
        return inner(value)
'''
    issues, counts = checker.inspect_source(source, "sample.py")
    assert counts == {"functions": 2, "classes": 1}
    assert any("value 缺少类型" in issue for issue in issues)
    assert any("option 缺少类型" in issue for issue in issues)
    assert any("kwargs 缺少类型" in issue for issue in issues)
    assert any("inner" in issue and "返回类型" in issue for issue in issues)
    assert any("Missing" in issue and "总体简介" in issue for issue in issues)
    assert not any("self 缺少类型" in issue for issue in issues)


def test_compliant_source_and_project_scope(
    checker: ModuleType,
    tmp_path: Path,
) -> None:
    """验证纯函数可省略异常和副作用段，且扫描排除运行目录。

    Args:
        checker: AST 检查模块。
        tmp_path: pytest 隔离临时目录。

    Raises:
        AssertionError: 纯函数误报或扫描范围错误。
        OSError: 写入临时测试文件失败时透传。

    副作用:
        在 tmp_path 内创建目录和测试文件。
    """
    source = '''"""规范源码示例。"""
def identity(value: int) -> int:
    """返回输入整数。

    Args:
        value: 示例整数。
    Results:
        int: 原样返回 value。
    """
    return value
'''
    assert checker.inspect_source(source, "valid.py")[0] == []
    for folder in ("src", "scripts", "tests", ".venv", "artifacts", "data"):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "example.py").write_text(source)
        (directory / "ignored.ipynb").write_text("{}")
    paths = checker.project_files(tmp_path)
    assert len(paths) == 3
    assert {p.parent.name for p in paths} == {"src", "scripts", "tests"}


def test_invalid_syntax_never_passes(
    checker: ModuleType,
) -> None:
    """验证无效语法直接令检查失败。

    Args:
        checker: AST 检查模块 fixture。

    Raises:
        AssertionError: 无效源码未抛出 SyntaxError。
    """
    with pytest.raises(SyntaxError):
        checker.inspect_source("def incomplete(", "invalid.py")


@pytest.mark.parametrize("missing", [None, "Args:", "Raises:", "Results:"])
def test_modern_doc_sections(
    checker: ModuleType,
    missing: str | None,
) -> None:
    """验证只对实际输入、返回及直接异常要求相应段落。

    Args:
        checker: AST 检查模块。
        missing: 要删除的必需段落；None 表示文档完整。

    Raises:
        AssertionError: 完整文档误报，或缺项没有被准确报告。
    """
    source = '''"""新格式示例。"""
def checked(value: int) -> int:
    """返回非负输入整数。

    Args:
        value: 待检查整数。
    Raises:
        ValueError: value 为负数。
    Results:
        int: 原样返回 value。
    """
    if value < 0:
        raise ValueError("negative")
    return value
'''
    if missing is not None:
        source = source.replace(f"    {missing}\n", "")
    issues, _ = checker.inspect_source(source, "modern.py")
    if missing is None:
        assert issues == []
    else:
        assert len(issues) == 1
        assert f"缺少 {missing}" in issues[0]


@pytest.mark.parametrize("section", [None, "Args:", "Raises:", "Results:", "副作用:"])
def test_absent_parts_are_omitted_not_filled_with_none(
    checker: ModuleType,
    section: str | None,
) -> None:
    """验证无输入/输出/异常/副作用时可只写总体简介，不能填“无”。

    Args:
        checker: AST 检查模块。
        section: 需要人为加入的冗余标题；None 表示不加入。

    Raises:
        AssertionError: 最简文档被误报，或“无”占位没有被拒绝。
    """
    extra = f"\n\n    {section}\n        无。" if section else ""
    source = (
        '"""模块说明。"""\ndef noop() -> None:\n'
        '    """只保留简介。' + extra + '\n    """\n    pass\n'
    )
    issues, _ = checker.inspect_source(source, "optional.py")
    assert not issues if section is None else any("应省略" in issue for issue in issues)


def test_side_effect_only_method_and_none_return(
    checker: ModuleType,
) -> None:
    """验证只有副作用的方法不需要空 Args、Raises 或 Results。

    Args:
        checker: AST 检查模块。

    Raises:
        AssertionError: 合法的仅副作用文档被误报。
    """
    source = '''"""示例模块。"""
class Box:
    """保存可清理的对象。"""
    def reset(self) -> None:
        """清空已有条目。

        副作用:
            原地清空 self.items。
        """
        self.items.clear()
'''
    assert checker.inspect_source(source, "state.py")[0] == []


def test_tensor_axes_use_short_names(
    checker: ModuleType,
) -> None:
    """验证 Tensor 参数和返回值中的长维度名都能被发现。

    Args:
        checker: AST 检查模块。

    Raises:
        AssertionError: 长维度未被发现，或合法简称误报。
    """
    source = '''"""形状示例。"""
def same(
    x: Float[Tensor, "batch channels height width"],
) -> Float[Tensor, "batch channels height width"]:
    """原样返回浮点图像。

    Args:
        x: 输入图像。
    Results:
        Tensor: 相同图像。
    """
    return x
'''
    issues, _ = checker.inspect_source(source, "shape.py")
    assert len([issue for issue in issues if "维度需用简称" in issue]) == 2
    source = source.replace("batch channels height width", "b c h w")
    assert checker.inspect_source(source, "shape.py")[0] == []


def test_sections_order_empty_sections_and_nested_returns(
    checker: ModuleType,
) -> None:
    """验证段落顺序、空段落及嵌套函数返回值的归属。

    Args:
        checker: AST 检查模块。

    Raises:
        AssertionError: 乱序/空段未被发现，或把内层返回值算作外层输出。
    """
    source = '''"""示例模块。"""
def outer() -> None:
    """定义内部常量函数。"""
    def inner() -> int:
        """返回固定整数。

        Results:
            int: 固定值。
        """
        return 1
'''
    assert checker.inspect_source(source, "nested.py")[0] == []
    source = source.replace("定义内部常量函数。", "定义内部常量函数。\n\n    Results:\n")
    issues, _ = checker.inspect_source(source, "empty.py")
    assert any("空段落" in issue for issue in issues)
    unordered = "简介。\n\nResults:\n    int: 结果。\n\nArgs:\n    value: 参数。"
    assert any("段落顺序" in issue for issue in checker._sections(unordered)[1])
