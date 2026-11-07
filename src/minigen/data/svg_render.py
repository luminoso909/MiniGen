"""受限静态 SVG 校验和栅格化。"""

import io
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path

from PIL import Image

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
INKSCAPE_NAMEDVIEW = "{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}namedview"
MAX_SVG_BYTES = 1024 * 1024
MAX_ELEMENTS = 6000
MAX_DEPTH = 64
MAX_GEOMETRY = 100_000
MIN_DIMENSION = 1e-6
MAX_NUMBERS = 60_000
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
NUMBER_RE = re.compile(NUMBER)
LENGTH_RE = re.compile(rf"^({NUMBER})(px|pt|pc|in|cm|mm|Q|%)?$")
LOCAL_ID = r"[A-Za-z0-9_.:-]+"
LOCAL_HREF_RE = re.compile(rf"^#({LOCAL_ID})$")
LOCAL_URL_RE = re.compile(rf"""url\(\s*['"]?#({LOCAL_ID})['"]?\s*\)""", re.IGNORECASE)
URL_CALL_RE = re.compile(r"\burl\s*\(", re.IGNORECASE)
SAFE_TAGS = frozenset(
    "svg g defs rect circle ellipse line polyline polygon path "
    "linearGradient radialGradient stop clipPath text tspan".split()
)
EDITOR_TAGS = frozenset(
    {
        "metadata",
        "title",
        "desc",
        *(f"{{{SVG_NAMESPACE}}}{tag}" for tag in ("metadata", "title", "desc")),
    }
)
GRADIENT_TAGS = frozenset({"linearGradient", "radialGradient"})
NUMERIC_ATTRIBUTES = frozenset(
    "x y x1 x2 y1 y2 cx cy r rx ry fx fy fr width height d points transform "
    "gradientTransform dx dy rotate textLength stroke-width stroke-dasharray stroke-dashoffset "
    "stroke-miterlimit font-size".split()
)
UNSUPPORTED_PROPERTIES = frozenset(
    {"filter", "mask", "marker", "marker-start", "marker-mid", "marker-end", "cursor"}
)
PIXELS_PER_UNIT = {
    "": 1.0,
    "px": 1.0,
    "in": 96.0,
    "cm": 96 / 2.54,
    "mm": 96 / 25.4,
    "Q": 96 / 101.6,
    "pt": 96 / 72,
    "pc": 16.0,
}


class SvgRenderError(ValueError):
    """记录可归类的 SVG 拒绝原因。"""

    def __init__(
        self,
        code: str,
        detail: str,
    ) -> None:
        """记录稳定错误码和简短细节。

        Args:
            code: 用于数据审计分组的错误码。
            detail: 面向排查的简短说明。
        """
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class ValidatedSvg:
    """记录已验证的渲染输入、原始画布信息与文本特征。"""

    xml: bytes
    view_box: tuple[float, float, float, float]
    source_size: tuple[float, float]
    contains_text: bool


@dataclass(frozen=True)
class RenderedSvg:
    """记录正方形 RGB PNG、源画布、内容框与后端信息。"""

    png: bytes
    view_box: tuple[float, float, float, float]
    source_size: tuple[float, float]
    content_box: tuple[int, int, int, int]
    had_transparency: bool
    contains_text: bool
    renderer: str


def _local_name(
    tag: str,
) -> str:
    """解析空命名空间或 SVG 命名空间中的标签名。

    Args:
        tag: ElementTree 标签字符串。

    Raises:
        SvgRenderError: 标签来自其他命名空间。

    Results:
        str: 本地 SVG 标签名。
    """
    if tag.startswith("{"):
        namespace, local = tag[1:].split("}", 1)
        if namespace != SVG_NAMESPACE:
            raise SvgRenderError("unsafe_feature", "foreign SVG content")
        return local
    return tag


def _pixel_length(
    value: str | None,
) -> float | None:
    """将绝对长度转换为 CSS 像素，相对长度由 viewBox 决定。

    Args:
        value: 原始宽高属性，可缺省。

    Raises:
        SvgRenderError: 可解析的绝对尺寸超出安全范围。

    Results:
        float | None: 以 96 DPI 换算的绝对尺寸，或无可用绝对尺寸。
    """
    match = LENGTH_RE.fullmatch(value.strip()) if value is not None else None
    if match is None or match[2] == "%":
        return None
    pixels = float(match[1]) * PIXELS_PER_UNIT[match[2] or ""]
    if not (MIN_DIMENSION <= pixels <= MAX_GEOMETRY):
        raise SvgRenderError("bad_geometry", "invalid canvas dimension")
    return pixels


def _canvas(
    root: ET.Element,
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    """解析原始 viewBox 和可用的画布宽高。

    Args:
        root: SVG 根元素。

    Raises:
        SvgRenderError: viewBox 非法、尺寸不足或超出安全范围。

    Results:
        tuple: 原始或推得的 viewBox，以及用于保比例栅格化的源宽高。
    """
    width = _pixel_length(root.get("width"))
    height = _pixel_length(root.get("height"))
    view_box = root.get("viewBox")
    if view_box is None:
        if width is None or height is None:
            raise SvgRenderError("bad_geometry", "absolute dimensions or viewBox required")
        return (0.0, 0.0, width, height), (width, height)
    parts = re.split(r"[\s,]+", view_box.strip())
    if len(parts) != 4 or any(not NUMBER_RE.fullmatch(part) for part in parts):
        raise SvgRenderError("bad_geometry", "viewBox must contain four numbers")
    x, y, vw, vh = map(float, parts)
    if not (abs(x) <= MAX_GEOMETRY and abs(y) <= MAX_GEOMETRY):
        raise SvgRenderError("bad_geometry", "viewBox origin outside bounds")
    if not (MIN_DIMENSION <= vw <= MAX_GEOMETRY and MIN_DIMENSION <= vh <= MAX_GEOMETRY):
        raise SvgRenderError("bad_geometry", "viewBox dimensions outside bounds")
    if width is not None and height is not None:
        return (x, y, vw, vh), (width, height)
    return (x, y, vw, vh), (vw, vh)


def _check_style(
    value: str,
) -> None:
    """校验内联 CSS，禁止绕过资源检查或生成大中间表面。

    Args:
        value: 已由 XML 解码的 style 属性值。

    Raises:
        SvgRenderError: 内联 CSS 出现规则、注释、转义或不支持的活动属性。
    """
    if any(mark in value for mark in ("@", "\\", "/*", "{", "}")):
        raise SvgRenderError("unsafe_feature", "dynamic or escaped CSS")
    for declaration in value.split(";"):
        if not declaration.strip():
            continue
        name, separator, item = declaration.partition(":")
        if not separator or (
            name.strip().lower() in UNSUPPORTED_PROPERTIES and item.strip() != "none"
        ):
            raise SvgRenderError("unsafe_feature", "unsupported CSS declaration")


def _attribute_references(
    node: ET.Element,
    tag: str,
) -> tuple[list[str], int]:
    """筛除编辑器属性，校验属性，并收集本地图元引用和复杂度。

    Args:
        node: 当前已知静态 SVG 元素。
        tag: 已通过白名单的本地标签名。

    Raises:
        SvgRenderError: SVG 试图触发资源、执行事件，或几何量过大。

    Results:
        tuple: 待校验的本地引用 ID 和几何数字计数。

    副作用:
        从当前元素删除不参与渲染的外部命名空间属性。
    """
    references = []
    numeric = 0
    for key, value in list(node.attrib.items()):
        name = key.rsplit("}", 1)[-1]
        lower = name.lower()
        if lower.startswith("on") or lower in {"src", "base", "data"}:
            raise SvgRenderError("unsafe_feature", f"active attribute {name}")
        if lower == "href":
            match = LOCAL_HREF_RE.fullmatch(value.strip())
            if key not in {"href", XLINK_HREF} or tag not in GRADIENT_TAGS or match is None:
                raise SvgRenderError("unsafe_feature", "unsupported or external href")
            references.append(match[1])
            continue
        if key.startswith("{") and key != XML_SPACE:
            del node.attrib[key]
            continue
        if key == "style":
            _check_style(value)
        if lower in UNSUPPORTED_PROPERTIES and value.strip() != "none":
            raise SvgRenderError("unsafe_feature", f"unsupported rendering property {name}")
        clean_value = value
        if URL_CALL_RE.search(value):
            for match in LOCAL_URL_RE.finditer(value):
                references.append(match[1])
            clean_value = LOCAL_URL_RE.sub("", value)
            if URL_CALL_RE.search(clean_value):
                raise SvgRenderError("unsafe_feature", "external or ambiguous URL reference")
        if key == "style":
            # CSS 十六进制颜色不应当作大坐标。
            properties = [declaration.partition(":") for declaration in clean_value.split(";")]
            numeric_values = [
                item for name, _, item in properties if name.strip().lower() in NUMERIC_ATTRIBUTES
            ]
        else:
            numeric_values = [clean_value] if key in NUMERIC_ATTRIBUTES else []
        for item in numeric_values:
            for match in NUMBER_RE.finditer(item):
                numeric += 1
                if abs(float(match[0])) > MAX_GEOMETRY:
                    raise SvgRenderError("bad_geometry", f"large geometry value in {key}")
    return references, numeric


def validate_svg(
    svg: bytes,
) -> ValidatedSvg:
    """校验有限的静态 SVG，并重序列化可交给渲染器的 XML。

    Args:
        svg: UTF-8 SVG 原始字节，最多 1 MiB。

    Raises:
        SvgRenderError: XML、画布、资源引用、复杂度或静态子集检查失败。

    Results:
        ValidatedSvg: 可渲染 XML、源画布和文本标记。
    """
    if not svg or len(svg) > MAX_SVG_BYTES:
        raise SvgRenderError("too_complex", "SVG byte limit")
    try:
        source = svg.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SvgRenderError("malformed_xml", "UTF-8 required") from error
    if "\x00" in source or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", source, re.IGNORECASE):
        raise SvgRenderError("unsafe_feature", "XML DTD, entity declaration or NUL")
    try:
        root = ET.fromstring(source)
    except (ET.ParseError, ValueError) as error:
        raise SvgRenderError("malformed_xml", "invalid SVG XML") from error
    if _local_name(root.tag) != "svg":
        raise SvgRenderError("malformed_xml", "SVG root required")
    view_box, source_size = _canvas(root)
    count = numeric = 0
    contains_text = False
    ids: set[str] = set()
    references: list[str] = []
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_ELEMENTS or depth > MAX_DEPTH:
            raise SvgRenderError("too_complex", "SVG element or nesting limit")
        tag = _local_name(node.tag)
        if tag not in SAFE_TAGS or (tag == "svg" and depth != 1):
            raise SvgRenderError("unsafe_feature", f"unsupported SVG tag {tag}")
        if tag in {"text", "tspan"}:
            contains_text = True
        node_id = node.get("id")
        if node_id:
            if node_id in ids:
                raise SvgRenderError("unsafe_feature", "duplicate element ID")
            ids.add(node_id)
        found, added = _attribute_references(node, tag)
        references.extend(found)
        numeric += added
        if numeric > MAX_NUMBERS:
            raise SvgRenderError("too_complex", "SVG geometry token limit")
        for child in reversed(list(node)):
            if child.tag in EDITOR_TAGS or child.tag == INKSCAPE_NAMEDVIEW:
                node.remove(child)
            else:
                stack.append((child, depth + 1))
    if not set(references) <= ids:
        raise SvgRenderError("unsafe_feature", "unresolved fragment reference")
    root.set("viewBox", " ".join(map(str, view_box)))
    return ValidatedSvg(ET.tostring(root, encoding="utf-8"), view_box, source_size, contains_text)


def _deny_fetch(
    url: str,
    resource_type: str,
) -> bytes:
    """拒绝渲染库尝试加载任何外部资源。

    Args:
        url: 渲染库传入的 URL。
        resource_type: 请求的资源种类。

    Raises:
        SvgRenderError: 任何外部抓取尝试。
    """
    del url, resource_type
    raise SvgRenderError("unsafe_feature", "renderer attempted a resource fetch")


def _worker(
    width: int,
    height: int,
    cpu_limit: int,
) -> None:
    """在一次性进程中施加资源限制并输出固定尺寸透明 PNG。

    Args:
        width: Cairo 输出宽。
        height: Cairo 输出高。
        cpu_limit: 进程 CPU 时间软硬限制（秒）。

    Raises:
        ImportError/OSError: 资源控制或本机渲染依赖不可用。
        SvgRenderError: 发生资源访问。

    副作用:
        读取 stdin、限制当前 worker 资源、向 stdout 写 PNG。
    """
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    if sys.platform.startswith("linux"):
        memory_limit = 768 * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    from cairosvg.surface import PNGSurface

    png = PNGSurface.convert(
        bytestring=sys.stdin.buffer.read(2 * MAX_SVG_BYTES),
        output_width=width,
        output_height=height,
        unsafe=False,
        url_fetcher=_deny_fetch,
    )
    sys.stdout.buffer.write(png)


def _worker_env() -> dict[str, str] | None:
    """让 macOS worker 查找已存在的 Homebrew Cairo，保持当前进程环境。

    Results:
        dict[str, str] | None: worker 专用动态库路径或继承当前环境。
    """
    if sys.platform != "darwin" or "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        return None
    cairo_dirs = [
        path
        for path in ("/opt/homebrew/opt/cairo/lib", "/usr/local/opt/cairo/lib")
        if (Path(path) / "libcairo.2.dylib").is_file()
    ]
    if not cairo_dirs:
        return None
    return {
        **os.environ,
        "DYLD_FALLBACK_LIBRARY_PATH": os.pathsep.join([*cairo_dirs, "/usr/local/lib", "/usr/lib"]),
    }


def render_svg(
    svg: bytes,
    *,
    size: int = 256,
    timeout_seconds: float = 6.0,
) -> RenderedSvg:
    """在子进程栅格化受限 SVG，保比例居中到白色 RGB 画布。

    Args:
        svg: UTF-8 SVG 原始字节。
        size: 正方形画布边长，取值 1 到 256。
        timeout_seconds: 单张渲染的墙钟超时，取值大于 0 且至多 30 秒。

    Raises:
        SvgRenderError: 校验、渲染、资源或图像尺寸失败，或图像无内容。

    Results:
        RenderedSvg: RGB PNG、源画布、原透明性、像素内容框与 CairoSVG 版本。

    副作用:
        启动一次性受限子进程，读取本机渲染依赖。无网络或输出路径写入。
    """
    if not (1 <= size <= 256 and 0 < timeout_seconds <= 30):
        raise SvgRenderError("bad_geometry", "size or timeout outside audit bounds")
    checked = validate_svg(svg)
    if util.find_spec("cairosvg") is None:
        raise SvgRenderError("missing_renderer", "install the CairoSVG data extra and system Cairo")
    source_width, source_height = checked.source_size
    scale = size / max(source_width, source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    code = (
        "from minigen.data.svg_render import _worker; import sys; _worker(*map(int, sys.argv[1:]))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(width), str(height), str(math.ceil(timeout_seconds))],
            input=checked.xml,
            capture_output=True,
            check=True,
            env=_worker_env(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise SvgRenderError("render_timeout", "SVG wall-clock limit reached") from error
    except subprocess.CalledProcessError as error:
        tail = error.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = tail[-1][:180] if tail else f"exit status {error.returncode}"
        raise SvgRenderError("render_failed", f"CairoSVG worker: {detail}") from error
    except OSError as error:
        raise SvgRenderError("render_failed", "could not start CairoSVG worker") from error
    try:
        with Image.open(io.BytesIO(completed.stdout), formats=["PNG"]) as decoded:
            if decoded.size != (width, height):
                raise SvgRenderError("render_failed", "unexpected renderer output dimensions")
            rgba = decoded.convert("RGBA")
    except OSError as error:
        raise SvgRenderError("render_failed", "invalid renderer PNG") from error
    alpha = rgba.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        raise SvgRenderError("empty_render", "no opaque or partially opaque pixels")
    offset_x, offset_y = (size - width) // 2, (size - height) // 2
    image = Image.new("RGB", (size, size), "white")
    image.paste(rgba, (offset_x, offset_y), alpha)
    png = io.BytesIO()
    image.save(png, format="PNG")
    left, top, right, bottom = bounds
    return RenderedSvg(
        png.getvalue(),
        checked.view_box,
        checked.source_size,
        (left + offset_x, top + offset_y, right + offset_x, bottom + offset_y),
        alpha.getextrema()[0] < 255,
        checked.contains_text,
        f"CairoSVG {metadata.version('CairoSVG')}",
    )
