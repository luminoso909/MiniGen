"""核验 B2 patch1 三个里程碑的同噪样图，并生成离线逐图对照报告。"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

type JsonDict = dict[str, Any]
type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_p1_milestones_10k_20k_30k")
STEPS = (10000, 20000, 30000)
COMMON = ("run_id", "model_config", "source_sha256", "data_identity")


def read_json(
    path: Path,
) -> JsonDict:
    """读取现有采样回执，不加载模型权重。

    Args:
        path: 需要读取的 JSON 文件。

    Raises:
        OSError/ValueError: 文件不可读取或 JSON 无效。

    Results:
        JsonDict: 采样身份与图像记录。
    """
    return json.loads(path.read_text())


def digest(
    path: Path,
) -> str:
    """计算待回传产物的完整字节摘要。

    Args:
        path: 已存在的产物文件。

    Raises:
        OSError: 文件不可读取。

    Results:
        str: SHA-256 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(
    path: Path,
    payload: JsonDict,
) -> None:
    """保存本次图像核验与报告清单。

    Args:
        path: 输出 JSON 文件。
        payload: 可序列化的报告对象。

    Raises:
        OSError/ValueError: 文件无法写入或对象含非法 JSON 数值。

    副作用:
        写入或更新本次输出目录中的 JSON 产物。
    """
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_inputs(
    output: Path,
) -> tuple[dict[int, JsonDict], dict[str, str]]:
    """核对三份回执、固定标签与噪声及全部 48 张原图。

    Args:
        output: 包含三个 generation/step_XXXXXX 目录的本次输出根目录。

    Raises:
        OSError/ValueError: 回执、图片、身份、标签或初噪不一致。

    Results:
        tuple: 按里程碑索引的回执，与所有已验证输入的相对路径及摘要。
    """
    receipts = {
        step: read_json(output / "generation" / f"step_{step:06d}" / "receipt.json")
        for step in STEPS
    }
    first = receipts[STEPS[0]]
    shared_rows = [
        (row["index"], row["sample_id"], row["caption"], row["noise_sha256"])
        for row in first["records"]
    ]
    files = {}
    for step, receipt in receipts.items():
        expected = {
            "schema": "minigen.anime_b2_milestones.generation.v1",
            "status": "completed",
            "arm": "b2_50m_p1",
            "checkpoint_step": step,
            "seed": 5001,
            "guidance": 3,
            "method": "euler",
            "steps": 40,
            "precision": "fp32",
            "sampling_batch_size": 4,
        }
        rows = receipt["records"]
        paired = [
            (row["index"], row["sample_id"], row["caption"], row["noise_sha256"]) for row in rows
        ]
        if (
            any(receipt[key] != value for key, value in expected.items())
            or any(receipt[key] != first[key] for key in COMMON)
            or [row["index"] for row in rows] != list(range(16))
            or paired != shared_rows
        ):
            raise ValueError(f"Milestone sampling identity or fixed noise differs: {step}")
        directory = output / "generation" / f"step_{step:06d}"
        receipt_path = directory / "receipt.json"
        files[str(receipt_path.relative_to(output))] = digest(receipt_path)
        for row in rows:
            path = directory / row["image_path"]
            actual = digest(path)
            if actual != row["image_sha256"]:
                raise ValueError(f"Generated image digest differs: {path}")
            with Image.open(path) as picture:
                picture.load()
                if picture.mode != "RGB" or picture.size != (256, 256):
                    raise ValueError(f"Expected a native RGB 256 image: {path}")
            files[str(path.relative_to(output))] = actual
    if len({row["checkpoint_sha256"] for row in receipts.values()}) != len(STEPS):
        raise ValueError("The three milestones must use distinct checkpoint identities")
    return receipts, files


def caption_lines(
    text: str,
    font: Font,
    width: int,
) -> list[str]:
    """按字体实际宽度折行，保留完整标签文本。

    Args:
        text: 要显示的英文标签。
        font: 拼图采用的字体。
        width: 每行最大像素宽度。

    Results:
        list[str]: 不超过画布宽度的连续文本行。
    """
    lines, current = [], ""
    for character in text:
        if current and font.getlength(current + character) > width:
            lines.append(current)
            current = character
        else:
            current += character
    if current:
        lines.append(current)
    return lines


def draw_comparisons(
    output: Path,
    receipts: dict[int, JsonDict],
) -> list[Path]:
    """生成四张四行三列原尺寸对照图，标题与标签按实际高度排版。

    Args:
        output: 本次输出根目录。
        receipts: 已通过核验的三个采样回执。

    Raises:
        OSError: 原图无法读取或拼图无法写入。

    Results:
        list[Path]: 四张已保存的比较图路径。

    副作用:
        创建 figures 目录并写入四张 PNG。
    """
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=14)
    title_font = ImageFont.load_default(size=16)
    width, gutter, heading = 816, 12, 76
    paths = []
    rows = receipts[STEPS[0]]["records"]
    for first in range(0, 16, 4):
        labels = [
            caption_lines(f"{i + 1:02d}. {rows[i]['caption']}", font, width - gutter * 2)
            for i in range(first, first + 4)
        ]
        heights = [256 + 12 + len(lines) * 18 + 20 for lines in labels]
        canvas = Image.new("RGB", (width, heading + sum(heights)), "#f7f5ef")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (gutter, 10),
            "B2 50M patch1 | EMA | seed5001 | CFG3 | Euler40",
            font=title_font,
            fill="#17323c",
        )
        for column, step in enumerate(STEPS):
            draw.text(
                (gutter + column * 268, 44), f"{step // 1000}k", font=title_font, fill="#17323c"
            )
        y = heading
        for offset, lines in enumerate(labels):
            index = first + offset
            for column, step in enumerate(STEPS):
                row = receipts[step]["records"][index]
                path = output / "generation" / f"step_{step:06d}" / row["image_path"]
                with Image.open(path) as picture:
                    canvas.paste(picture, (gutter + column * 268, y))
            for line_index, line in enumerate(lines):
                draw.text((gutter, y + 264 + line_index * 18), line, font=font, fill="#17323c")
            y += heights[offset]
        path = figures / f"steps_{first + 1:02d}-{first + 4:02d}.png"
        canvas.save(path)
        paths.append(path)
    return paths


def write_html(
    output: Path,
    receipts: dict[int, JsonDict],
    figures: list[Path],
) -> Path:
    """生成含三步训练量、四张拼图和十六题逐图对照的离线页面。

    Args:
        output: 生成报告的输出目录。
        receipts: 已核验的里程碑回执。
        figures: 四张原尺寸比较图。

    Raises:
        OSError: HTML 无法写入。

    Results:
        Path: 可离线打开的 index.html。

    副作用:
        写入 index.html，不修改图像或模型。
    """
    table_rows = []
    for step, receipt in receipts.items():
        meta = receipt["metadata"]
        table_rows.append(
            f"<tr><th>{step:,}</th><td>{meta['validation_flow_mse']:.6f}</td>"
            f"<td>{meta['equivalent_epochs']:.3f}</td><td>{meta['samples_seen']:,}</td>"
            f"<td><code>{receipt['checkpoint_sha256']}</code></td></tr>"
        )
    figure_sections = "".join(
        f'<figure><img src="{path.relative_to(output)}" alt="{path.stem}" loading="lazy">'
        f'<figcaption><a href="{path.relative_to(output)}">原尺寸 {path.stem}</a>'
        "</figcaption></figure>"
        for path in figures
    )
    questions = []
    for index, row in enumerate(receipts[STEPS[0]]["records"]):
        images = "".join(
            f"<figure><figcaption>{step // 1000}k</figcaption><img "
            f'src="generation/step_{step:06d}/'
            f'{escape(receipts[step]["records"][index]["image_path"], quote=True)}" '
            f'alt="第{index + 1}题 {step // 1000}k" loading="lazy"></figure>'
            for step in STEPS
        )
        questions.append(
            f'<article id="q{index + 1:02d}"><h3>{index + 1:02d}. '
            f'{escape(row["caption"])}</h3><div class="three">{images}</div>'
            f"<details><summary>配对身份</summary><code>{escape(row['sample_id'])}</code>"
            f"<br>初噪 SHA256：<code>{row['noise_sha256']}</code></details></article>"
        )
    first = receipts[STEPS[0]]
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B2 patch1 10k/20k/30k 同噪对照</title>
<style>body{{font:16px/1.6 system-ui;background:#f7f5ef;color:#17323c;max-width:1280px;
margin:24px auto;padding:0 18px}}h1{{line-height:1.25}}table{{border-collapse:collapse;width:100%;
background:white}}td,th{{border:1px solid #c7d3d5;padding:8px;text-align:left}}
code{{overflow-wrap:anywhere}}
.scroll{{overflow:auto}}figure{{margin:12px 0}}figure>img{{max-width:100%;height:auto}}
.three{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.three figure img{{width:100%;max-width:256px}}article{{background:white;border:1px solid #cad5d7;
padding:14px;margin:18px 0}}a{{color:#176a78}}
@media(max-width:620px){{.three{{gap:5px}}}}</style></head>
<body><h1>B2 50M patch1 · 10k / 20k / 30k 同噪对照</h1>
<p>同一训练分支、原16条开发标签、seed5001、EMA、CFG3、Euler40、FP32。
10k和20k仅新增采样，30k复用既有已验收样图。本轮不训练。</p>
<p>图像并排用于观察五官、虹膜和布局随训练步数的变化。
以下 Flow MSE 是固定验证误差，不等于图像质量；本页不自动给出视觉评分。
这16条是复用开发标签，不是新的独立 Benchmark。</p>
<h2>三个里程碑</h2><div class="scroll"><table><thead><tr><th>步数</th><th>固定验证 Flow MSE</th>
<th>等价 epoch</th><th>累计样本暴露</th><th>检查点 SHA256</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div>
<details><summary>共同模型和源码身份</summary><p>run_id：<code>{first["run_id"]}</code></p>
<p>源码 SHA256：<code>{first["source_sha256"]}</code></p>
<pre>{escape(json.dumps(first["model_config"], ensure_ascii=False, indent=2))}</pre></details>
<h2>四组完整对照</h2>{figure_sections}<h2>16题逐图浏览</h2>{"".join(questions)}
<p><a href="acceptance.json">图像及配对核验</a> ·
<a href="report_manifest.json">文件摘要清单</a></p>
</body></html>"""
    path = output / "index.html"
    path.write_text(page)
    return path


def main() -> None:
    """核验产物后保存拼图、HTML、验收回执和回传文件清单。

    Raises:
        OSError/ValueError: 输入身份、图像或报告文件不符合约定。

    副作用:
        读取三步已有样图，在同一输出目录写入比较图及报告；不读取模型权重。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.root.resolve() / args.output
    receipts, inputs = verify_inputs(output)
    figures = draw_comparisons(output, receipts)
    html = write_html(output, receipts, figures)
    files = inputs | {str(path.relative_to(output)): digest(path) for path in [*figures, html]}
    acceptance = {
        "schema": "minigen.anime_b2_milestones.acceptance.v1",
        "status": "verified",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "image_count": 48,
        "same_prompts_and_noise": True,
        "training_updates": 0,
        "source": {key: receipts[STEPS[0]][key] for key in COMMON},
        "milestones": {
            str(step): {
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "metadata": receipt["metadata"],
                "receipt_sha256": inputs[f"generation/step_{step:06d}/receipt.json"],
            }
            for step, receipt in receipts.items()
        },
        "input_files": inputs,
        "figures": {str(path.relative_to(output)): digest(path) for path in figures},
        "html_sha256": digest(html),
    }
    path = output / "acceptance.json"
    write_json(path, acceptance)
    files[str(path.relative_to(output))] = digest(path)
    write_json(
        output / "report_manifest.json",
        {"schema": "minigen.anime_b2_milestones.report_manifest.v1", "files": files},
    )
    print(f"Verified 48 images with identical labels/noise; wrote {len(figures)} grids: {html}")


if __name__ == "__main__":
    main()
