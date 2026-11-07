"""在历史回归中显式启用教学导入路径，包括回归启动的子进程。"""

import os
import sys
from pathlib import Path


def pytest_sessionstart() -> None:
    """让历史测试及其子进程解析原有课件模块名。

    副作用:
        更新当前 pytest 进程的导入路径和传给子进程的 PYTHONPATH。
    """
    root = Path(__file__).resolve().parents[1]
    learning_src = str(root / "docs" / "learning" / "src")
    sys.path.insert(0, learning_src)
    # 历史回归还通过 importlib 按文件路径加载独立脚本。
    sys.path.insert(0, str(root / "scripts"))
    old_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = learning_src + (os.pathsep + old_path if old_path else "")
