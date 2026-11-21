"""生成动漫头像 Flow 时间采样消融的配对预览面板。"""

import argparse
import json
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont

from minigen._typing import JsonDict

ARMS = ("uniform", "mid60")
MODES = ("correct", "eye_control")
SAMPLING_FIELDS = (
    "schema",
    "architecture",
    "model_config",
    "objective",
    "checkpoint_step",
    "data_identity",
    "weights",
    "precision",
    "method",
    "steps",
    "guidance",
    "seed",
    "noise_sha256",
    "plan_sha256",
    "condition_mode",
)
COMMON_RECORD_FIELDS = ("index", "sample_id", "condition_sample_id", "caption", "noise_sha256")
EYE_RECORD_FIELDS = ("variant", "pair_index", "eye_from", "eye_to")


def paired_reports(
    root: Path,
    mode: str,
) -> tuple[JsonDict, JsonDict]:
    """读取两个实验臂的采样回执并验证设置与逐图条件一一对应。

    Args:
        root: 含 uniform 和 mid60 预览的评测根目录。
        mode: correct 为 16 条验证样本，eye_control 为 12 条眼色提示词。

    Raises:
        OSError: 采样回执无法读取。
        ValueError: JSON、采样范围、训练身份或配对设置不符合约定。
        KeyError: 回执缺少约定字段。

    Results:
        tuple[JsonDict, JsonDict]: Uniform、Mid60 顺序的标准采样回执。
    """
    uniform, mid60 = (
        json.loads((root / arm / mode / "samples.json").read_text(encoding="utf-8")) for arm in ARMS
    )
    if any(uniform[key] != mid60[key] for key in SAMPLING_FIELDS):
        raise ValueError(f"Sampling settings differ: {mode}")
    if (
        uniform["schema"] != "minigen.samples.v1"
        or uniform["condition_mode"] != mode
        or uniform["checkpoint_sha256"] == mid60["checkpoint_sha256"]
    ):
        raise ValueError(f"Expected two distinct checkpoints and a matching condition mode: {mode}")
    count = 16 if mode == "correct" else 12
    if any(
        report["split"] != "validation" or len(report["records"]) != count
        for report in (uniform, mid60)
    ):
        raise ValueError(f"Unexpected split or sample count: {mode}")
    fields = COMMON_RECORD_FIELDS + (EYE_RECORD_FIELDS if mode == "eye_control" else ())
    records = [tuple(record[key] for key in fields) for record in uniform["records"]]
    other_records = [tuple(record[key] for key in fields) for record in mid60["records"]]
    if records != other_records or len(set(records)) != count:
        raise ValueError(f"Sample identities or prompts differ: {mode}")
    if any(record[1] != record[2] for record in records):
        raise ValueError("Each validation image must retain its own fixed condition")
    for report in (uniform, mid60):
        if len({row["image_path"] for row in report["records"]}) != count:
            raise ValueError(f"Sample image paths are repeated: {mode}")
    return uniform, mid60


def eye_replacement(
    before: str,
    after: str,
) -> tuple[str, str]:
    """核对相邻提示词只替换了一个眼色标签，并返回前后的标签。

    Args:
        before: 以逗号分隔的基础英文标签。
        after: 仅替换眼色后的同格式英文标签。

    Raises:
        ValueError: 两组标签数量不等或存在眼色以外的更改。

    Results:
        tuple[str, str]: 旧眼色标签和新眼色标签。
    """
    first = [tag.strip().casefold() for tag in before.split(",")]
    second = [tag.strip().casefold() for tag in after.split(",")]
    if len(first) != len(second):
        raise ValueError(f"The eye-control prompt changes tag count: {before!r}, {after!r}")
    changed = [(a, b) for a, b in zip(first, second, strict=True) if a != b]
    if len(changed) != 1 or any(not tag.endswith(" eyes") for tag in changed[0]):
        raise ValueError(f"Expected exactly one changed eye-color tag: {before!r}, {after!r}")
    return changed[0]


def verify_eye_pairs(
    validation: JsonDict,
    control: JsonDict,
) -> None:
    """确认干预组每对使用同一初噪，且只替换固定验证提示词中的眼色。

    Args:
        validation: 正确条件的 16 图采样回执。
        control: 12 图眼色干预回执，按 base、swap 交错。

    Raises:
        ValueError: 配对出自不同样本、初噪不同或标签替换不吻合。
        KeyError: 样本记录未提供必要的身份或标签字段。
    """
    references = {record["index"]: record for record in validation["records"]}
    records = control["records"]
    for index, (base, swap) in enumerate(zip(records[::2], records[1::2], strict=True)):
        reference = references[base["index"]]
        if (
            (base["variant"], swap["variant"]) != ("base", "swap")
            or (base["pair_index"], swap["pair_index"]) != (index, index)
            or base["index"] != swap["index"]
            or (base["sample_id"], swap["sample_id"]) != (reference["sample_id"],) * 2
            or base["caption"] != reference["caption"]
            or base["noise_sha256"] != reference["noise_sha256"]
            or base["noise_sha256"] != swap["noise_sha256"]
        ):
            raise ValueError(f"Eye-control pair {index} does not match its validation record")
        change = eye_replacement(base["caption"], swap["caption"])
        if any((record["eye_from"], record["eye_to"]) != change for record in (base, swap)):
            raise ValueError(f"Eye-control pair {index} has inconsistent eye-color metadata")


def render_panel(
    output: Path,
    root: Path,
    mode: str,
    reports: tuple[JsonDict, JsonDict],
    indices: range,
) -> None:
    """绘制按标签或眼色替换配对的两臂样图。

    Args:
        output: 面板 PNG 目标路径。
        root: 含各实验臂输出的评测根目录。
        mode: correct 绘制两个臂的固定条件，eye_control 绘制换色前后四列。
        reports: Uniform、Mid60 顺序的同模式采样回执。
        indices: 要绘制的验证样本或眼色配对的零基索引区间。

    Raises:
        OSError: 图像无法读取或输出不可写。
        ValueError: 图像不是 256×256，或眼色提示词的配对不符。
        KeyError: 采样记录缺少图片路径或提示词。

    副作用:
        写入一张逐样本的对照 PNG。
    """
    eye_control = mode == "eye_control"
    settings = reports[0]
    headers = (
        ("Uniform · base", "Uniform · swapped", "Mid60 · base", "Mid60 · swapped")
        if eye_control
        else ("Uniform", "Mid60")
    )
    tile, gap, pad, top, caption_width = 206, 10, 22, 110, 345
    caption_x = pad + len(headers) * (tile + gap) + 9
    stride = tile + 14
    canvas = Image.new(
        "RGB", (caption_x + caption_width + pad, top + len(indices) * stride + 8), "#f7f6f0"
    )
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=24)
    label = ImageFont.load_default(size=16)
    body = ImageFont.load_default(size=14)
    ink, muted = "#1d3038", "#405662"
    heading = "Eye-color attribute control" if eye_control else "Fixed validation tags"
    noise = "Same noise within each pair" if eye_control else "Same noise across arms"
    draw.text(
        (pad, 15), f"{heading}  |  {indices.start + 1:02d}-{indices.stop:02d}", font=title, fill=ink
    )
    draw.text(
        (pad, 52),
        f"Step {settings['checkpoint_step']:,}  /  Seed {settings['seed']}  /  "
        f"{settings['method'].upper()} {settings['steps']}  /  CFG {settings['guidance']:g}  /  "
        f"{settings['weights'].upper()}  /  {noise}",
        font=body,
        fill=muted,
    )
    for column, heading_text in enumerate(headers):
        x = pad + column * (tile + gap)
        color = "#dce8eb" if column < len(headers) // 2 else "#e7e7d7"
        draw.rounded_rectangle((x, 80, x + tile - 1, 103), radius=4, fill=color)
        draw.text((x + tile / 2, 91), heading_text, font=label, fill=ink, anchor="mm")
    draw.text((caption_x, 82), "Base tags" if eye_control else "Shared tags", font=label, fill=ink)

    for offset, index in enumerate(indices):
        y = top + offset * stride
        record_indices = (2 * index, 2 * index + 1) if eye_control else (index,)
        column = 0
        for arm, report in zip(ARMS, reports, strict=True):
            for record_index in record_indices:
                file = root / arm / mode / report["records"][record_index]["image_path"]
                with Image.open(file) as source:
                    if source.size != (256, 256):
                        raise ValueError(f"Expected 256x256: {file}")
                    image = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(image, (pad + column * (tile + gap), y))
                column += 1
        record = settings["records"][record_indices[0]]
        caption = record["caption"]
        if eye_control:
            after = settings["records"][record_indices[1]]["caption"]
            before_eye, after_eye = eye_replacement(caption, after)
            id_text = f"{index + 1:02d}  /  {record['sample_id']}  /  {before_eye} -> {after_eye}"
        else:
            id_text = f"{index + 1:02d}  /  {record['sample_id']}"
        draw.text((caption_x, y + 3), id_text, font=label, fill=ink)
        wrapped = "\n".join(wrap(caption, width=44, max_lines=9, placeholder=" ..."))
        draw.multiline_text((caption_x, y + 29), wrapped, font=body, fill=muted, spacing=3)
        draw.line((pad, y + tile + 7, canvas.width - pad, y + tile + 7), fill="#d7dfdc")
    canvas.save(output)


def main() -> None:
    """校验两臂的固定条件与眼色提示词并生成三张面板。

    Raises:
        OSError: 采样回执、图像或输出目录无法访问。
        ValueError: 采样条件、实验臂身份或图片不符合约定。
        KeyError: 采样记录缺少约定字段。

    副作用:
        创建输出目录并保存三张 PNG，打印路径。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {mode: paired_reports(args.root, mode) for mode in MODES}
    cross_mode_fields = tuple(key for key in SAMPLING_FIELDS if key != "condition_mode")
    for correct, eye_control in zip(reports["correct"], reports["eye_control"], strict=True):
        if any(
            correct[key] != eye_control[key] for key in (*cross_mode_fields, "checkpoint_sha256")
        ):
            raise ValueError("Fixed and eye-control previews use different checkpoints or settings")
    verify_eye_pairs(reports["correct"][0], reports["eye_control"][0])
    output = args.output or args.root / "panels"
    output.mkdir(parents=True, exist_ok=True)
    panels = (
        ("fixed_01-08.png", "correct", range(8)),
        ("fixed_09-16.png", "correct", range(8, 16)),
        ("eye_control.png", "eye_control", range(6)),
    )
    for name, mode, indices in panels:
        target = output / name
        render_panel(target, args.root, mode, reports[mode], indices)
        print(target)


if __name__ == "__main__":
    main()
