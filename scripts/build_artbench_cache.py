"""兼容旧 P2 命令；正式实现由 minigen cache 共用。"""

from pathlib import Path

from minigen.cache_command import main

if __name__ == "__main__":
    main(root=Path(__file__).resolve().parents[1])
