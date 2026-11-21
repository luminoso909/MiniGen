"""从本地六组完成回执、JSON 日志和 PNG 生成训练曲线及 48 张模型网格。"""

import argparse
import hashlib
import html
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ARMS = ("b1_30m_p2", "b2_30m_p2", "b1_50m_p2", "b2_50m_p2", "b1_50m_p1", "b2_50m_p1")
RUNS = Path("artifacts/runs/anime_faces_256/six_30k_v1")
OUTPUT = Path("artifacts/exports/anime-faces-b1/six_30k_v1")
COLORS = ("#0b7285", "#1971c2", "#2b8a3e", "#a61e4d", "#e67700", "#6741d9")


def read_json(
    path: Path,
) -> dict[str, Any]:
    """读取已有的本地回执，不执行模型相关代码。

    Args:
        path: 需要读取的 JSON 路径。

    Raises:
        OSError/ValueError: 文件读取或 JSON 解析失败。

    Results:
        dict: 异构实验元数据。
    """
    return json.loads(path.read_text())


def digest(
    path: Path,
) -> str:
    """计算图片与训练证据文件的内容摘要。

    Args:
        path: 本地已有文件。

    Raises:
        OSError: 文件不可读。

    Results:
        str: SHA-256 十六进制字符串。
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def aggregate(
    metrics: list[dict[str, Any]],
    train_samples: int,
) -> list[dict[str, Any]]:
    """按连续一千次更新聚合日志，并保留窗口末的验证与学习率。

    Args:
        metrics: 一个已完成分支的逐步训练日志。
        train_samples: 训练集实际独立图片数量，用于计算等价 epoch。

    Results:
        list: 每千步的均值、样本暴露次数、等价轮数和验证 MSE。
    """
    periods = []
    samples_seen = 0
    for first in range(0, len(metrics), 1000):
        window = metrics[first : first + 1000]
        samples_seen += sum(row["samples"] for row in window)
        periods.append(
            {
                "step": window[-1]["step"],
                "first_step": window[0]["step"],
                "loss_mean": fmean(row["loss"] for row in window),
                "validation_flow_mse": window[-1]["val_loss"],
                "lr_last": window[-1]["lr"],
                "lr_mean": fmean(row["lr"] for row in window),
                "grad_norm_mean": fmean(row["grad_norm"] for row in window),
                "step_seconds_mean": fmean(row["step_seconds"] for row in window),
                "data_seconds_mean": fmean(row["data_seconds"] for row in window),
                "elapsed_hours": window[-1]["elapsed_seconds"] / 3600,
                "samples_seen": samples_seen,
                "equivalent_epochs": samples_seen / train_samples,
            }
        )
    return periods


def load_runs(
    root: Path,
    output: Path,
) -> dict[str, dict[str, Any]]:
    """核对本地完成回执并汇总原始日志，不读取权重或数据缓存。

    Args:
        root: 已回传训练小文件的项目根目录。
        output: 含六组生成回执和 PNG 的评测目录。

    Raises:
        OSError/ValueError: 完成身份、日志覆盖或验收回执不符。

    Results:
        dict: 六组的架构、配方、实际结果和千步聚合记录。
    """
    acceptance = read_json(root / RUNS / "acceptance.json")
    if acceptance["status"] != "verified" or set(acceptance["arms"]) != set(ARMS):
        raise ValueError("Training report requires the completed six-arm acceptance")
    arms = {}
    for arm in ARMS:
        folder = root / RUNS / arm
        run, result = read_json(folder / "run.json"), read_json(folder / "result.json")
        generation = read_json(output / "generation" / arm / "receipt.json")
        metrics_path = folder / "metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if (
            result != acceptance["arms"][arm]["result"]
            or result["status"] != "completed"
            or result["step"] != 30000
            or result["run_id"] != run["run_id"]
            or result["run_id"] != generation["run_id"]
            or generation["image_count"] != 128
            or len(generation["records"]) != 128
            or [row["step"] for row in metrics] != list(range(1, 30001))
        ):
            raise ValueError(f"Incomplete or unpaired six-arm report input: {arm}")
        arms[arm] = {
            "config": run["config"],
            "result": result,
            "periods": aggregate(metrics, result["train_samples"]),
            "loss_windows": acceptance["arms"][arm]["loss_windows"],
            "snapshots": acceptance["arms"][arm]["snapshots"],
            "data_identity": result["data_identity"],
            "source_sha256": result["source_sha256"],
            "metrics_sha256": digest(metrics_path),
            "generation": {
                "run_id": generation["run_id"],
                "checkpoint_sha256": generation["checkpoint_sha256"],
                "seed": generation["seed"],
                "weights": generation["weights"],
                "precision": generation["precision"],
                "image_count": generation["image_count"],
            },
        }
    return arms


def draw_curves(
    output: Path,
    arms: dict[str, dict[str, Any]],
) -> list[str]:
    """绘制独立 PNG 曲线，避免用训练 loss 代替视觉质量。

    Args:
        output: 报告及曲线所在目录。
        arms: 六组已校验的千步聚合数据。

    Raises:
        OSError: 图片无法保存。

    Results:
        list: 相对报告目录的曲线 PNG 路径。

    副作用:
        使用 Agg 后端在 curves 目录写五张曲线图。
    """
    directory = output / "curves"
    directory.mkdir(exist_ok=True)
    fields = (
        ("loss_mean", "Training Flow MSE (1000-update mean)", False),
        ("validation_flow_mse", "Fixed validation Flow MSE (EMA)", False),
        ("lr_last", "Learning rate (window endpoint)", True),
        ("step_seconds_mean", "Seconds/update (1000-update mean, shared GPU)", False),
        ("grad_norm_mean", "Gradient norm before clipping (1000-update mean)", False),
    )
    paths = []
    for field, title, logarithmic in fields:
        figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
        for arm, color in zip(ARMS, COLORS, strict=True):
            periods = arms[arm]["periods"]
            axis.plot(
                [row["step"] for row in periods],
                [row[field] for row in periods],
                label=arm,
                color=color,
                linewidth=1.7,
            )
        axis.set(title=title, xlabel="Optimizer updates")
        if logarithmic:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(ncols=2, fontsize=9)
        path = directory / (field + ".png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path.relative_to(output)))
    return paths


def make_grids(
    output: Path,
) -> list[dict[str, Any]]:
    """从 768 张已生成 PNG 拼出 48 张模型内 4×4 原尺寸网格。

    Args:
        output: 六组生成回执、图片及报告目录。

    Raises:
        OSError/ValueError: PNG 哈希、尺寸或每个设置的十六图覆盖不符。

    Results:
        list: 每张网格的模型、CFG、积分器和相对路径。

    副作用:
        在 model_grids 目录写入 48 张带序号的 PNG。
    """
    directory = output / "model_grids"
    directory.mkdir(exist_ok=True)
    grids = []
    heading, font = ImageFont.load_default(size=18), ImageFont.load_default(size=14)
    for arm in ARMS:
        source = output / "generation" / arm
        receipt = read_json(source / "receipt.json")
        groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
        for record in receipt["records"]:
            groups[record["guidance"], record["method"], record["steps"]].append(record)
        expected = {(g, m, s) for g in (0, 1, 3, 5) for m, s in (("euler", 40), ("heun", 20))}
        if set(groups) != expected:
            raise ValueError(f"Sampling condition coverage changed: {arm}")
        for (guidance, method, steps), records in sorted(groups.items()):
            records.sort(key=lambda record: record["index"])
            if [record["index"] for record in records] != list(range(16)):
                raise ValueError(f"Sampling grid must contain indices 0–15: {arm}")
            condition = f"cfg{guidance}_{method}{steps}"
            canvas = Image.new("RGB", (1064, 1208), "#f5f3ec")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (10, 10), f"{arm} | EMA 30k | {condition} | seed 5001", "#243544", font=heading
            )
            for index, record in enumerate(records):
                path = source / record["image_path"]
                if digest(path) != record["image_sha256"]:
                    raise ValueError(f"Generation PNG hash differs: {path}")
                x, y = 10 + (index % 4) * 263, 46 + (index // 4) * 288
                draw.text((x, y), f"{index + 1:02d}", "#243544", font=font)
                with Image.open(path) as image:
                    if image.size != (256, 256):
                        raise ValueError(f"Expected native 256px image: {path}")
                    canvas.paste(image.convert("RGB"), (x, y + 20))
            path = directory / f"{arm}_{condition}.png"
            canvas.save(path)
            grids.append(
                {
                    "arm": arm,
                    "guidance": guidance,
                    "method": method,
                    "steps": steps,
                    "path": str(path.relative_to(output)),
                }
            )
    return grids


def table(
    headings: list[str],
    rows: list[list[object]],
    attributes: list[str] | None = None,
) -> str:
    """将纯文本行编码成可横向滚动的 HTML 表格。

    Args:
        headings: 各列的显示名称。
        rows: 不含 HTML 的单元格内容。
        attributes: 每行内部生成的筛选属性；默认留空。

    Results:
        str: 转义文本后的完整表格片段。
    """
    header = "".join("<th>" + html.escape(value) + "</th>" for value in headings)
    body = "".join(
        "<tr "
        + (attributes[index] if attributes else "")
        + ">"
        + "".join("<td>" + html.escape(str(cell)) + "</td>" for cell in row)
        + "</tr>"
        for index, row in enumerate(rows)
    )
    return (
        '<div class="scroll"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def write_report(
    root: Path,
    output: Path,
    arms: dict[str, dict[str, Any]],
    curves: list[str],
    grids: list[dict[str, Any]],
) -> None:
    """拼接模型、配方、暴露次数、日志和网格的浏览报告。

    Args:
        root: 已回传日志与验收文件的本地项目目录。
        output: 报告发布目录。
        arms: 已验证的六组元数据及千步统计。
        curves: 已生成曲线的相对路径。
        grids: 已生成四乘四网格的元数据。

    Raises:
        OSError/ValueError: 元数据不可序列化或报告不可写。

    副作用:
        写 training_summary.json 与 training_report.html。
    """
    architecture, recipes, exposures, periods = [], [], [], []
    attributes = []
    for arm, record in arms.items():
        config, result = record["config"], record["result"]
        model = config["model"]
        depth = (
            str(model["depth"])
            if "depth" in model
            else f"double {model['double_depth']} + single {model['single_depth']}"
        )
        architecture.append(
            [
                arm,
                f"{result['trainable_parameters']:,}",
                model["architecture"].upper(),
                depth,
                model["hidden_size"],
                model["num_heads"],
                model["hidden_size"] // model["num_heads"],
                model["patch_size"],
                (model["latent_size"] // model["patch_size"]) ** 2,
                f"{model['in_channels']}×{model['latent_size']}×{model['latent_size']}",
                model["text_dim"],
                model["mlp_ratio"],
                model["position_encoding"],
            ]
        )
        recipes.append(
            [
                arm,
                config["seed"],
                f"{config['batch_size']}×{config['grad_accum']}="
                f"{config['batch_size'] * config['grad_accum']}",
                f"{config['learning_rate']:.1e}",
                config["warmup_steps"],
                config["schedule_steps"],
                config["min_lr_ratio"],
                config["weight_decay"],
                config["grad_clip"],
                config["condition_dropout"],
                config["ema_decay"],
                config["precision"],
                "uniform"
                if config["time_mid_probability"] is None
                else str(config["time_mid_probability"]),
            ]
        )
        last = record["periods"][-1]
        exposures.append(
            [
                arm,
                result["train_samples"],
                result["validation_samples"],
                result["step"],
                f"{last['samples_seen']:,}",
                f"{last['equivalent_epochs']:.3f}",
                f"{result['elapsed_seconds'] / 3600:.2f}",
                f"{record['loss_windows']['first_100']:.4f} → "
                f"{record['loss_windows']['last_100']:.4f}",
                f"{last['validation_flow_mse']:.6f}",
            ]
        )
        for row in record["periods"]:
            periods.append(
                [
                    arm,
                    row["step"],
                    f"{row['loss_mean']:.6f}",
                    f"{row['validation_flow_mse']:.6f}",
                    f"{row['lr_last']:.3e}",
                    f"{row['grad_norm_mean']:.4f}",
                    f"{row['step_seconds_mean']:.3f}",
                    f"{row['data_seconds_mean']:.5f}",
                    f"{row['equivalent_epochs']:.3f}",
                    f"{row['elapsed_hours']:.2f}",
                ]
            )
            attributes.append(f'data-arm="{arm}"')
    model_table = table(
        [
            "组",
            "可训练参数",
            "架构",
            "层数",
            "宽度",
            "头数",
            "每头维度",
            "latent patch",
            "图像 tokens",
            "latent C×H×W",
            "文本维度",
            "MLP 倍率",
            "位置编码",
        ],
        architecture,
    )
    recipe_table = table(
        [
            "组",
            "seed",
            "微批×累积",
            "峰值 LR",
            "warmup",
            "cosine 总步数",
            "最低 LR 比例",
            "weight decay",
            "grad clip",
            "CFG dropout",
            "EMA",
            "精度",
            "时间采样",
        ],
        recipes,
    )
    exposure_table = table(
        [
            "组",
            "训练图",
            "日志验证图",
            "更新数",
            "累计样本暴露",
            "等价 epoch",
            "累计运行小时",
            "训练 loss 首/末100步",
            "最终验证 MSE",
        ],
        exposures,
    )
    period_table = table(
        [
            "组",
            "窗口末步",
            "千步 loss 均值",
            "EMA 验证 MSE",
            "末步 LR",
            "梯度范数均值",
            "秒/更新",
            "数据秒/更新",
            "等价 epoch",
            "累计小时",
        ],
        periods,
        attributes,
    )
    options = '<option value="all">全部模型</option>' + "".join(
        "<option>" + arm + "</option>" for arm in ARMS
    )
    grid_html = "".join(
        f'<figure data-arm="{grid["arm"]}" data-cfg="{grid["guidance"]}" '
        f'data-method="{grid["method"]}">'
        + f"<figcaption>{grid['arm']} · CFG {grid['guidance']} · "
        f"{grid['method']} {grid['steps']}</figcaption>"
        + f'<a href="{grid["path"]}"><img loading="lazy" src="{grid["path"]}"></a></figure>'
        for grid in grids
    )
    figures = "".join(
        '<a href="' + path + '"><img class="curve" src="' + path + '"></a>' for path in curves
    )
    label_records = read_json(output / "generation" / ARMS[0] / "receipt.json")["records"][:16]
    label_table = table(
        ["图片编号", "原始固定验证标签"],
        [[f"{row['index'] + 1:02d}", row["caption"]] for row in label_records],
    )
    source = next(iter(arms.values()))["data_identity"]["source"]
    evidence = html.escape(
        json.dumps(
            {
                "source_sha256": next(iter(arms.values()))["source_sha256"],
                "caption_policy": source["caption_policy"],
                "max_content_tokens": source["max_content_tokens"],
                "image_preprocessing": source["image_preprocessing"],
                "manifest_sha256": source["manifest_sha256"],
                "split_receipt_sha256": source["split_receipt_sha256"],
                "vae_revision": source["vae"]["revision"],
                "vae_scaling_factor": source["vae"]["scaling_factor"],
                "vae_encoding": source["vae"]["inference"],
                "clip_revision": source["clip"]["revision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    relative_acceptance = html.escape(os.path.relpath(root / RUNS / "acceptance.json", output))
    page = HTML.replace("__ARCHITECTURE__", model_table).replace("__RECIPES__", recipe_table)
    for placeholder, value in {
        "__EXPOSURES__": exposure_table,
        "__PERIODS__": period_table,
        "__OPTIONS__": options,
        "__GRIDS__": grid_html,
        "__CURVES__": figures,
        "__EVIDENCE__": evidence,
        "__ACCEPTANCE__": relative_acceptance,
        "__LABELS__": label_table,
    }.items():
        page = page.replace(placeholder, value)
    (output / "training_report.html").write_text(page)
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "instance_cost": None,
                "scope": "Six completed 30k runs; final EMA; 16 development labels, seed 5001",
                "arms": arms,
                "curves": curves,
                "model_grids": grids,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    """运行纯本地的训练报告与网格整理，不调用训练和采样。

    Raises:
        OSError/ValueError: 本地完成证据或生成图不完整。

    副作用:
        读取 JSON/PNG，写入 HTML、JSON、曲线和模型网格。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    arms = load_runs(root, output)
    curves = draw_curves(output, arms)
    grids = make_grids(output)
    write_report(root, output, arms, curves, grids)
    print(output / "training_report.html")


HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>MiniGen 六组 30k 训练报告</title>
<style>
body{margin:24px auto;padding:0 20px;max-width:1500px;background:#f4f2eb;color:#243544;
font:15px/1.6 system-ui,sans-serif}h1,h2{line-height:1.3}a{color:#096b7d}
.scroll{overflow:auto;margin:18px 0}table{border-collapse:collapse;background:#fff}
th,td{padding:7px 10px;border:1px solid #cdd1d2;white-space:nowrap}th{background:#e8eeed}
img{max-width:100%;height:auto}.curve{max-width:740px}figure{margin:24px 0;background:#fff;
padding:10px;border-radius:8px}figcaption{font-weight:600}select{font:inherit;padding:6px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:16px}
</style><h1>MiniGen 六组 30k 训练报告</h1>
<p><a href="comparison.html">锁分后的六模型对照与评分</a> ·
<a href="review/blind_review.html">匿名评分页面</a> ·
<a href="training_summary.json">完整训练汇总 JSON</a> ·
<a href="__ACCEPTANCE__">训练验收回执</a></p>
<p>六组随机初始化、训练 seed 42。最终 30k EMA 使用同一批 16 条开发标签和 seed 5001 初噪，
比较 CFG 0/1/3/5 与 Euler40/Heun20。积分器每张图均为 40 次速度场求值；CFG 3/5 的双分支
实际主干调用数为 80，CFG 0/1 为 40。本页完整展示 768 张生成图，未增加训练或采样。</p>
<p>开发图与训练误差用于诊断；单个初噪种子、反复使用的 16 条标签不能替代独立正式 Benchmark。
Flow MSE 下降不代表五官、虹膜和画质已经提高，视觉结论以先锁分后解盲的评分为准。</p>
<h2>架构与参数</h2>__ARCHITECTURE__
<p>B1 为单流图文注意力；B2 先双流联合注意力，再单流。patch 指 32×32 VAE 潜空间上的边长，
patch 1 的图像 token 数为 1024，patch 2 为 256；文本 token 另计。
冻结 VAE/CLIP 不计入可训练参数。</p>
<h2>训练配方</h2>__RECIPES__
<p>使用 Flow Matching 速度目标、AdamW。500 步线性 warmup 后按累计 30k 余弦衰减，
最低 LR 为峰值的 0.1。每 1000 步验证并更新滚动 last，编号快照保留在
3000/5000/7000/10000/13000/15000/17000/20000/23000/25000/27000/30000。</p>
<h2>训练暴露与实际耗时</h2>__EXPOSURES__
<p>等价 epoch = 日志累计 samples / 独立训练图数量，包含 epoch 尾部小批的实际样本数。
日志验证图为固定子集，生成展示用其中 16 条标签。累计运行小时读取真实训练回执，
六组共享 GPU 并发，不能将各组小时相加作为实例总时长；未取得账单，实例实际费用未知。</p>
<h2>训练曲线</h2>__CURVES__
<p>每点为连续 1000 次更新的训练均值；验证值为该窗口末的固定配对 EMA Flow MSE。
学习率取窗口最后一步，warmup 的前 500 步可查原始 metrics 日志。
秒/更新含数据访问与同步，受同卡并发影响，不能当作独立模型的硬件吞吐基准。</p>
<h2>每 1000 步记录</h2><select id="arm">__OPTIONS__</select>__PERIODS__
<h2>48 张模型内网格</h2>
<p>每张网格按原始标签顺序放置 16 图；图片编号 01–16 在全部模型与采样设置中一致。
点击网格查看原尺寸。CFG 0 为无条件生成，不评价文字遵循。</p>
<details><summary>展开 16 条固定标签</summary>__LABELS__</details>
<label>CFG <select id="cfg"><option value="all">全部</option><option>0</option>
<option>1</option><option>3</option><option>5</option></select></label>
<label>积分器 <select id="method"><option value="all">全部</option>
<option>euler</option><option>heun</option></select></label>__GRIDS__
<h2>数据与冻结组件身份</h2><pre>__EVIDENCE__</pre>
<p>分支的完整数据身份、配置、checkpoint 摘要与曲线路径保存在 training_summary.json。
报告不读取 checkpoint、数据集或表示缓存。</p>
<script>
function filter(){
const arm=document.getElementById('arm').value,cfg=document.getElementById('cfg').value;
const method=document.getElementById('method').value;
for(const row of document.querySelectorAll('[data-arm]')){
  row.hidden=arm!=='all'&&row.dataset.arm!==arm;
  if(row.tagName==='FIGURE')row.hidden=row.hidden
    ||cfg!=='all'&&row.dataset.cfg!==cfg||method!=='all'&&row.dataset.method!==method;
}}
for(const id of ['arm','cfg','method'])document.getElementById(id).onchange=filter;
</script></html>"""


if __name__ == "__main__":
    main()
