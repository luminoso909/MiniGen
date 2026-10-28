"""验证注释结构检查器能发现缺漏，且不扫描运行产物或执行待检查代码。"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def checker() -> ModuleType:
    """装载独立 AST 检查脚本，供结构检查测试调用。

    1 输入：
        无参数；从项目 scripts 目录解析脚本。
    2 输出：
        含 inspect_source 和 project_files 的模块，不触发其 main。
    3 报错（Raises）：
        ImportError/OSError：脚本装载失败时透传。
    """
    path = Path(__file__).resolve().parents[1] / "scripts/check_code_annotations.py"
    spec = importlib.util.spec_from_file_location("annotation_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_nested_argument_return_and_sections_are_reported(checker: ModuleType) -> None:
    """验证嵌套函数、可变参数与类文档缺失不会被漏检。

    1 输入：
        checker：独立 AST 检查模块 fixture。
    2 输出：
        无返回值；断言各类缺漏都产生定位信息。
    3 报错（Raises）：
        AssertionError：缺漏未被报告，或定义计数不符。
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
    assert any("Missing" in issue and "3 报错" in issue for issue in issues)
    assert not any("self 缺少类型" in issue for issue in issues)


def test_compliant_source_and_project_scope(checker: ModuleType, tmp_path: Path) -> None:
    """验证完整文档可通过，且扫描范围排除 Notebook 与生成目录。

    1 输入：
        checker：AST 检查模块；tmp_path：pytest 隔离临时目录。
    2 输出：
        无返回值；断言规范源码无问题且只枚举第一方 Python 目录。
    3 报错（Raises）：
        AssertionError：合法源码误报或排除目录被扫描；文件错误透传。
    """
    source = '''"""规范源码示例。"""
def identity(value: int) -> int:
    """返回输入整数。

    1 输入：
        value：用于示例的整数。
    2 输出：
        同一个整数。
    3 报错（Raises）：
        无显式 raise。
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


def test_invalid_syntax_never_passes(checker: ModuleType) -> None:
    """验证语法错误直接令检查失败，不伪装成没有结构问题。

    1 输入：
        checker：独立 AST 检查模块 fixture。
    2 输出：
        无返回值；只接受 SyntaxError 作为无效源码的结果。
    3 报错（Raises）：
        AssertionError：无效语法未抛出 SyntaxError。
    """
    with pytest.raises(SyntaxError):
        checker.inspect_source("def incomplete(", "invalid.py")
