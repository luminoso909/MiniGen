"""覆盖 ArtBench 描述清洗中可能改变视觉条件的分支。"""

import pytest

from minigen.data.artbench_caption import clean_caption


def _word_count(
    text: str,
) -> int:
    """在单元测试中模拟只数内容 token 的计数器。

    Args:
        text: 由空格分隔词的测试字符串。

    Results:
        int: 词数；真实数据由锁定 CLIP tokenizer 计数。
    """
    return len(text.split())


def test_caption_preserves_visual_facts_during_cleanup() -> None:
    """检查署名与日期被删除，画内年份、颜色、数量和位置仍保留。

    Raises:
        AssertionError: 清洗后丢失视觉事实或未处理明确的艺术史信息。
    """
    result = clean_caption(
        "The image shows an oil painting of two red boats, painted by Claude Monet in 1889, "
        'to the left of a blue door marked "1889". The painting is titled "Two Boats". '
        "The painting was created in 1888. The sky is gray. The sky is gray.",
        "claude-monet",
    )
    assert result.text == (
        'An oil painting of two red boats, to the left of a blue door marked "1889". '
        "The sky is gray."
    )
    assert set(result.actions) == {
        "remove_artist_credit",
        "remove_image_opening",
        "remove_history",
        "remove_duplicate_sentence",
    }


def test_caption_preserves_subject_names_and_mixed_history() -> None:
    """验证画中姓名未被全局替换，并从混合句中保留明确视觉从句。

    Raises:
        AssertionError: 姓名或视觉从句被删除。
    """
    result = clean_caption(
        "It was painted by Marie Laurencin and shows two women in front of a tree. "
        "The woman on the right is Marie Laurencin.",
        "marie-laurencin",
    )
    assert result.text == (
        "The painting shows two women in front of a tree. "
        "The woman on the right is Marie Laurencin."
    )


@pytest.mark.parametrize(
    ("caption", "max_tokens", "expected", "action"),
    [
        (
            "Two red birds perch nearby. A second sentence is too long.",
            5,
            "Two red birds perch nearby.",
            "fit_sentence_budget",
        ),
        (
            "A blue house, with many trees behind it.",
            4,
            "A blue house",
            "fit_clause_budget",
        ),
        (
            "Two purple flowers grow under the tree.",
            4,
            "Two purple flowers grow",
            "fit_word_budget",
        ),
    ],
)
def test_caption_budget_keeps_original_cleaned_text(
    caption: str,
    max_tokens: int,
    expected: str,
    action: str,
) -> None:
    """检查裁短模式、实际长度和完整清洗文本均可追溯。

    Args:
        caption: 测试输入描述。
        max_tokens: 模拟的内容词数预算。
        expected: 应得到的受限描述。
        action: 应记录的裁短方式。

    Raises:
        AssertionError: 未记录裁短、失去原文或长度越界。
    """
    result = clean_caption(caption, count_tokens=_word_count, max_tokens=max_tokens)
    assert result.status == "over_limit"
    assert result.text == caption
    assert result.budget_text == expected
    assert result.budget_token_count == _word_count(expected)
    assert result.budget_token_count <= max_tokens
    assert action in result.actions


def test_metadata_only_caption_requires_explicit_resolution() -> None:
    """检查纯艺术史描述被标为需处理，不编造画面条件。

    Raises:
        AssertionError: 元数据被错误作为视觉条件保留或添加了虚构描述。
    """
    result = clean_caption("The painting was created by Claude Monet in 1889.", "claude-monet")
    assert result.status == "empty"
    assert result.budget_text == ""
    assert "remove_history" in result.actions
