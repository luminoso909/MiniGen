"""为显式执行的历史检查脚本启用课件源码路径。"""

import sys
from pathlib import Path


def enable_learning_imports() -> None:
    """使当前独立检查进程能解析历史课件中的 minigen 模块。

    副作用:
        将仓库下的课件源码目录加到当前进程的 sys.path。
    """
    learning_src = str(Path(__file__).resolve().parents[1] / "docs" / "learning" / "src")
    sys.path.insert(0, learning_src)
