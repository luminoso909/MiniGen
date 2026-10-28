"""个人算法学习边界；缺失核心时明确报错而不自动替代。"""

from typing import NoReturn


class CoreImplementationRequired(NotImplementedError):
    """标识个人算法核心尚未完成的异常类型。

    1 输入：
        args：继承异常类的消息参数，说明当前缺失的学习任务。

    2 输出：
        异常实例，由 pending 主动抛出，阻止静默替代个人实现。

    3 报错（Raises）：
        类自身无显式 raise；调用 pending 时会抛出本类型。
    """


def pending(task: str) -> NoReturn:
    """明确终止尚未完成的个人核心调用。

    1 输入：
        task：任务名称，写入错误消息以定位对应学习入口。

    2 输出：
        不返回；始终抛出 CoreImplementationRequired。

    3 报错（Raises）：
        CoreImplementationRequired：每次调用均抛出，提示先完成学习任务。
    """
    raise CoreImplementationRequired(f"{task}: 请先完成 learning/README.md 中的核心学习任务")
