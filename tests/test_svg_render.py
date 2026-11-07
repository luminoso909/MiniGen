"""验证 P1 静态 SVG 的资源拒绝、尺寸规范与透明合成。"""

import io
from importlib.util import find_spec

import pytest
from PIL import Image

from minigen.data.svg_render import SvgRenderError, render_svg, validate_svg

WIDE = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="-10 0 100 50">'
    b'<rect x="-10" width="100" height="50" fill="#ff0000"/></svg>'
)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            b'<!DOCTYPE svg [<!ENTITY load SYSTEM "file:///tmp/example">]>'
            b"<svg viewBox='0 0 1 1'>&load;</svg>",
            "unsafe_feature",
        ),
        (b'<svg width="1" height="1"><script>alert(1)</script></svg>', "unsafe_feature"),
        (
            b'<svg width="1" height="1"><image href="data:image/png;base64,AA=="/></svg>',
            "unsafe_feature",
        ),
        (b'<svg width="1" height="1"><style>@import "x";</style></svg>', "unsafe_feature"),
        (
            b'<svg width="1" height="1"><rect style="fill:url(file:///tmp/x)"/></svg>',
            "unsafe_feature",
        ),
        (
            b'<svg width="1" height="1"><rect style="fill:\\75rl(http://example)"/></svg>',
            "unsafe_feature",
        ),
        (
            b'<svg width="1" height="1"><rect style="fill:ur&#108;(http://example)"/></svg>',
            "unsafe_feature",
        ),
        (b'<svg width="1" height="1"><use href="#self" id="self"/></svg>', "unsafe_feature"),
        (b'<svg width="1" height="1" onload="x"/>', "unsafe_feature"),
        (b'<svg width="1" height="1" xml:base="https://example.invalid"/>', "unsafe_feature"),
        (b'<svg viewBox="0 0 0 100"/>', "bad_geometry"),
        (b'<svg viewBox="0 0 1 1"><path d="M 1e90 0"/></svg>', "bad_geometry"),
        (b'<svg width="1" height="1">' + b"<g>" * 65 + b"</g>" * 65 + b"</svg>", "too_complex"),
        (b'<svg width="1" height="1">' + b" " * (1024**2) + b"</svg>", "too_complex"),
    ],
)
def test_rejects_svg_outside_safe_subset(
    payload: bytes,
    code: str,
) -> None:
    """危险、不受限或无效输入在调用渲染器之前即被拒绝。

    Args:
        payload: SVG 攻击或边界示例。
        code: 预期拒绝类别。

    Raises:
        AssertionError: SVG 未被拒绝或分类错误。
    """
    with pytest.raises(SvgRenderError) as caught:
        validate_svg(payload)
    assert caught.value.code == code


def test_accepts_editor_metadata_and_local_gradient() -> None:
    """保留静态绘图并去除编辑器元数据，接受文档内渐变引用。

    Raises:
        AssertionError: 有效画布、局部资源或文本识别与预期不符。
    """
    data = b"""
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:custom="urn:editor" viewBox="-10 0 100 50">
      <metadata><custom:title>Editor metadata</custom:title></metadata>
      <defs><linearGradient id="paint"><stop stop-color="blue"/></linearGradient></defs>
      <rect width="99" height="49" style="fill:url(#paint);stroke:#ffffff;stroke-width:1"/>
      <text x="10" y="20">sample</text>
    </svg>
    """
    checked = validate_svg(data)
    assert checked.source_size == (100, 50)
    assert checked.view_box == (-10, 0, 100, 50)
    assert checked.contains_text
    assert b"Editor metadata" not in checked.xml
    assert b"url(#paint)" in checked.xml


@pytest.mark.skipif(find_spec("cairosvg") is None, reason="需要 SVG 可选渲染依赖")
def test_renders_centered_rgb_with_intrinsic_aspect() -> None:
    """验证真实子进程能按原比例渲染并正确填白。

    Raises:
        AssertionError: 画布、边界、透明标记或颜色错误。
        SvgRenderError: 本机后端不可用或渲染失败。
    """
    rendered = render_svg(WIDE)
    assert rendered.view_box == (-10, 0, 100, 50)
    assert rendered.source_size == (100, 50)
    assert rendered.content_box == (0, 64, 256, 192)
    assert not rendered.had_transparency
    assert rendered.renderer.startswith("CairoSVG ")
    with Image.open(io.BytesIO(rendered.png)) as image:
        assert image.mode == "RGB"
        assert image.size == (256, 256)
        assert image.getpixel((128, 0)) == (255, 255, 255)
        assert image.getpixel((128, 128)) == (255, 0, 0)


@pytest.mark.skipif(find_spec("cairosvg") is None, reason="需要 SVG 可选渲染依赖")
def test_renders_fragment_gradient_and_alpha() -> None:
    """验证内部渐变不触发外部抓取，透明位置在最终图像填白。

    Raises:
        AssertionError: 渐变、透明合成或内容框错误。
        SvgRenderError: 本机后端不可用或渲染失败。
    """
    svg = b"""
    <svg viewBox="0 0 100 100">
      <defs><linearGradient id="paint">
        <stop offset="0" stop-color="blue"/><stop offset="1" stop-color="red"/>
      </linearGradient></defs>
      <rect x="25" y="25" width="50" height="50" fill="url(#paint)" opacity="0.5"/>
    </svg>
    """
    rendered = render_svg(svg)
    assert rendered.had_transparency
    assert rendered.content_box == (64, 64, 192, 192)
    with Image.open(io.BytesIO(rendered.png)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((80, 100)) != image.getpixel((180, 100))
