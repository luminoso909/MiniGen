"""保守清洗 ArtBench 的英文视觉描述，记录被移除的信息与 CLIP 长度处理。"""

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from minigen.autoencoders.sdxl import hash_file

CAPTION_CLEANUP_VERSION = "artbench-caption-v1"
_SENTENCE_END = re.compile(r"(?<!\b[A-Z]\.)(?<!\bSt\.)(?<!\bDr\.)(?<=[.!?])\s+")
_TITLE = r"""(?:"[^"\n]+"|'[^'\n]+')"""
_YEAR = r"(?:1[0-9]{3}|20[0-9]{2})(?:\s*(?:–|-|to)\s*(?:1[0-9]{3}|20[0-9]{2}))?"
_DATE = rf"(?:(?:in|around|circa)\s+{_YEAR}|(?:in|during)\s+the\s+\d{{1,2}}(?:st|nd|rd|th) century)"
_CREDIT = r"(?:painted|created|drawn|made|illustrated|signed|executed)"
_NAME_WORD = r"[A-ZÀ-ÖØ-Þ][\w’'\-]*"
_NAME = rf"{_NAME_WORD}(?:\s+(?:(?:de|del|van|von|da|di|la|le)\s+)?{_NAME_WORD}){{0,5}}"
_OPENING = re.compile(r"^The (?:image|picture) (?:shows|depicts|features)\s+", re.I)
_VISUAL_VERB = r"(?:shows|depicts|features|contains|illustrates)"
_HISTORY = re.compile(
    r"^(?:It|This (?:painting|artwork|work)|The (?:painting|artwork|work))\s+"
    rf"(?:(?:is|was) (?:{_CREDIT}\s+(?:by|in|around|during|circa)\b|"
    r"(?:titled|called|known as|housed|exhibited|published)\b)|"
    r"(?:belongs to|is part of) (?:the|a) collection\b|dates back to\b)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class CaptionCleanup:
    """保存清洗结果及实际输入 CLIP 的长度受限版本。"""

    text: str
    budget_text: str
    status: Literal["ok", "empty", "over_limit"]
    actions: tuple[str, ...]
    token_count: int | None
    budget_token_count: int | None


def _artist_pattern(
    artist: str | None,
) -> str:
    """将作者 slug 转成带词边界的姓名匹配式。

    Args:
        artist: 原始作者元数据，可以缺失。

    Results:
        str: 可插入正则的作者名，仅在明确署名位置使用。
    """
    words = re.findall(r"[^\W_]+", artist or "")
    if not words or " ".join(words).casefold() in {"unknown", "anonymous", "n a"}:
        return _NAME
    exact = r"[\s'’\-]+".join(re.escape(word) for word in words)
    return rf"(?:(?i:{exact})(?![\w’'\-])|{_NAME})"


def _visual_sentences(
    caption: str,
    artist: str | None,
) -> tuple[list[str], list[str]]:
    """移除可明确识别的艺术史元数据，保留视觉描述及修改原因。

    Args:
        caption: 原始英文描述。
        artist: 上游作者元数据，用于定位显式署名。

    Results:
        tuple: 视觉句子列表和实际采取的处理动作。
    """
    actions: list[str] = []
    normalized = " ".join(unicodedata.normalize("NFC", caption).split())
    if normalized != caption:
        actions.append("normalize_spacing")
    name = _artist_pattern(artist)
    suffix = r"(?=\s*[,;.!?]|\s+(?i:and|with|featuring|showing|depicting)\b|$)"
    # 署名只在显式创作动词后处理，画中人物或文字出现的姓名与年份保留。
    credit = re.compile(rf"(?:,\s*|\s+)(?i:{_CREDIT}\s+by\s+){name}(?:\s+(?i:{_DATE}))?{suffix}")
    date = re.compile(rf"(?:,\s*|\s+)(?i:{_CREDIT}\s+{_DATE}){suffix}")
    title = re.compile(rf"(?:,\s*|\s+)(?i:titled)\s+{_TITLE}(?=\W|$)")
    sent_hist = re.compile(rf"\b(?:and|but)\s+({_VISUAL_VERB}\b.*)$", re.I)
    kept = []
    seen = set()
    for part in _SENTENCE_END.split(normalized):
        sentence = part.strip()
        if _HISTORY.match(sentence):
            # 元数据句尾如有独立的视觉动词，提取该句而不是删除画面事实。
            visual_clause = sent_hist.search(sentence)
            sentence = f"The painting {visual_clause[1]}" if visual_clause else ""
            actions.append("remove_history")
        if not sentence:
            continue
        for pattern, action in (
            (credit, "remove_artist_credit"),
            (date, "remove_creation_date"),
            (title, "remove_work_title"),
            (_OPENING, "remove_image_opening"),
        ):
            sentence, changes = pattern.subn("", sentence)
            if changes:
                actions.append(action)
        sentence = sentence.strip(" ,;")
        if not sentence:
            continue
        key = sentence.casefold().rstrip(".!?")
        if key in seen:
            actions.append("remove_duplicate_sentence")
            continue
        seen.add(key)
        kept.append(sentence[0].upper() + sentence[1:])
    return kept, actions


def _fit_caption(
    sentences: list[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> tuple[str, int, str]:
    """按完整句、从句边界和词边界依次裁短过长的视觉描述。

    Args:
        sentences: 非空视觉句子列表，首句优先描述主画面。
        count_tokens: 不加 BOS/EOS 的 CLIP 计数函数。
        max_tokens: 允许的内容 token 数。

    Results:
        tuple: 长度受限的描述、实际 token 数、已使用的裁短方式。
    """
    candidates = (
        (
            "fit_sentence_budget",
            (" ".join(sentences[:n]) for n in range(len(sentences) - 1, 0, -1)),
        ),
        (
            "fit_clause_budget",
            (
                sentences[0][: marker.start()].rstrip()
                for marker in reversed(list(re.finditer(r"[,;:](?=\s)", sentences[0])))
            ),
        ),
        (
            "fit_word_budget",
            (
                sentences[0][: marker.start()].rstrip(" ,;:")
                for marker in reversed(list(re.finditer(r"\s+", sentences[0])))
            ),
        ),
    )
    for action, texts in candidates:
        for text in texts:
            if text and (size := count_tokens(text)) <= max_tokens:
                return text, size, action
    return "", 0, "unresolved_token_budget"


def clean_caption(
    caption: str,
    artist: str | None = None,
    *,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = 75,
) -> CaptionCleanup:
    """从 Florence 草稿中保留画面描述，并显式记录超长裁短。

    Args:
        caption: 原始英文 caption；清洗规则无法判断描述是否符合图像。
        artist: 可选的上游作者标识，只用于识别明确创作署名。
        count_tokens: 可选 CLIP 内容 token 计数；缺省时不执行长度裁短。
        max_tokens: 不包含首尾特殊 token 的预算，必须大于零。

    Raises:
        ValueError: max_tokens 小于一。

    Results:
        CaptionCleanup: 完整清洗文本、预算文本、状态、处理原因和两种长度。
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    sentences, actions = _visual_sentences(caption, artist)
    text = " ".join(sentences)
    count = count_tokens(text) if count_tokens is not None else None
    if not text:
        return CaptionCleanup(text, text, "empty", tuple(dict.fromkeys(actions)), count, count)
    if count_tokens is not None and count is not None and count > max_tokens:
        bounded, bounded_count, action = _fit_caption(sentences, count_tokens, max_tokens)
        return CaptionCleanup(
            text,
            bounded,
            "over_limit",
            tuple(dict.fromkeys([*actions, action])),
            count,
            bounded_count,
        )
    return CaptionCleanup(text, text, "ok", tuple(dict.fromkeys(actions)), count, count)


def load_clip_token_counter(
    root: Path,
    lock_path: Path,
) -> Callable[[str], int]:
    """仅加载并核验本地锁定的 CLIP-B/32 tokenizer。

    Args:
        root: 项目根目录。
        lock_path: 固定 CLIP 组件的 JSON 锁文件路径。

    Raises:
        ImportError: 当前环境缺少 transformers。
        OSError: 本地词表、配置或锁文件不可读。
        ValueError: tokenizer 资产的字节数或 SHA-256 不匹配。

    Results:
        Callable: 输入英文描述，返回不加 BOS/EOS 且未截断的真实 token 数。
    """
    from transformers import CLIPTokenizer

    identity = json.loads(lock_path.read_text())
    directory = root / identity["local_path"]
    tokenizer_files = (
        "config.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    for name in tokenizer_files:
        path = directory / name
        file_info = identity["files"][name]
        if path.stat().st_size != file_info["bytes"] or hash_file(path) != file_info["sha256"]:
            raise ValueError(f"CLIP tokenizer file does not match lock: {name}")
    tokenizer = CLIPTokenizer.from_pretrained(str(directory), local_files_only=True)

    def count_tokens(
        text: str,
    ) -> int:
        """获得单条文本的完整 CLIP 内容 token 长度。

        Args:
            text: 待计数的英文描述。

        Results:
            int: 关闭 BOS/EOS 注入和截断时的 token 数。
        """
        return len(
            tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        )

    return count_tokens
