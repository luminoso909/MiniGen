"""Explicit boundary for algorithms owned by the learner."""


class CoreImplementationRequired(NotImplementedError):
    """Raised instead of silently delegating to a third-party implementation."""


def pending(task: str) -> None:
    raise CoreImplementationRequired(f"{task}: 请先完成 learning/README.md 中的核心学习任务")
