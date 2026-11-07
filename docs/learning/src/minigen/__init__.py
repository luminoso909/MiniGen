"""历史课件专用导入入口，由显式的学习 PYTHONPATH 激活。"""

from importlib.metadata import version
from pathlib import Path

# 先解析教学模块，教学入口也可调用仓库中的共用工程组件。
__path__.append(str(Path(__file__).resolve().parents[4] / "src" / "minigen"))
__version__ = version("minigen")
