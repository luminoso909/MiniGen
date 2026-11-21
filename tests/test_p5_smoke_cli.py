"""验证 P5 冒烟入口只使用调用方提供的本地题库。"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from run_p5_smoke import DEFAULT_BENCHMARK, main


@pytest.mark.parametrize("missing", ["benchmark", "lock"])
def test_missing_benchmark_input_stops_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    """本地题库或内容锁缺失时，在新建输出之前提示调用方。

    Args:
        tmp_path: 隔离的评测项目目录。
        monkeypatch: 写入测试 CLI 参数。
        capsys: 捕获标准错误。
        missing: 指定缺少题库还是内容锁。

    Raises:
        AssertionError: 未报告缺失路径或仍启动评测。

    副作用:
        写入测试文件并临时替换命令行参数和子进程执行器。
    """
    output = tmp_path / "smoke"
    benchmark = tmp_path.resolve() / DEFAULT_BENCHMARK
    missing_path = benchmark
    if missing == "lock":
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text("{}\n", encoding="utf-8")
        missing_path = benchmark.with_name("lock.json")
    argv = ["run_p5_smoke.py", "--root", str(tmp_path), "--output", "smoke"]
    monkeypatch.setattr(sys, "argv", argv)
    with patch("run_p5_smoke.subprocess.run") as runner, pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert str(missing_path) in capsys.readouterr().err
    assert not output.exists()
    runner.assert_not_called()


@pytest.mark.parametrize("path_kind", ["default", "relative", "absolute"])
def test_smoke_forwards_local_benchmark_to_all_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_kind: str,
) -> None:
    """检查生成、参考评分和性能剖析使用同一份本地题库。

    Args:
        tmp_path: 隔离的父目录和合成题库位置。
        monkeypatch: 切换工作目录并写入测试 CLI 参数。
        path_kind: 默认位置、相对项目根的路径或绝对路径。

    Raises:
        AssertionError: 路径、子进程的工作目录或退出状态不正确。

    副作用:
        写入合成题库和测试回执，临时替换子进程执行器。
    """
    root = (tmp_path / "project").resolve()
    argv = ["run_p5_smoke.py", "--root", "project", "--output", "smoke"]
    match path_kind:
        case "default":
            benchmark = root / DEFAULT_BENCHMARK
        case "relative":
            benchmark = root / "local" / "benchmark.json"
            argv.extend(["--benchmark", str(benchmark.relative_to(root))])
        case _:
            benchmark = tmp_path / "outside" / "benchmark.json"
            argv.extend(["--benchmark", str(benchmark)])
    root.mkdir()
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text("{}\n", encoding="utf-8")
    benchmark.with_name("lock.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)

    completed = subprocess.CompletedProcess([], 0)
    with (
        patch("run_p5_smoke.subprocess.run", return_value=completed) as runner,
        pytest.raises(SystemExit) as result,
    ):
        main()

    assert result.value.code == 0
    assert runner.call_count == 10
    forwarded = []
    for call in runner.call_args_list:
        assert call.kwargs["cwd"] == root
        command = call.args[0]
        for option in ("--benchmark", "--reference"):
            if option in command:
                forwarded.append(command[command.index(option) + 1])
    assert forwarded == [str(benchmark)] * 7
