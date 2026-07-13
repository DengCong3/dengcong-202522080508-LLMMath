"""鲁棒的程序化 SVG 奖励/校验工具，用于 Logo 生成任务。

设计目标
------------
- 作为训练代理指标：廉价、确定性，并能对 Gemma 3 270M 等小模型经 LoRA 微调后
  的典型失效模式保持敏感。
- 避免过度拟合表面标记：奖励应反映 SVG 是否结构合法、视觉上是否合理，以及对提示词
  是否有响应。
- 可在实验报告中解释：每个子分数都应有清晰的设计理由。

评分维度
------------------
1. 干净提取
   - 是否恰好存在一个顶层 <svg>？
   - 是否存在 markdown/code fences，以及 SVG 前后是否有多余文本。
2. 结构有效性
   - 是否正确闭合。
   - 是否包含 viewBox 与 xmlns。
3. 完整性/无退化
   - SVG 文本长度是否落在合理区间。
   - 是否存在截断痕迹（如未闭合标签、重复片段）。
4. 提示词覆盖
   - 提示词中的关键词是否在 SVG 的标签、属性或文本内容中出现。
5. 图元多样性
   - 是否使用了多种矢量图元。
6. 配色合理性
   - 使用的颜色是否基本合法且数量可控。
7. 坐标合理性
   - 图元是否大致落在预期的画布中心区域内。
8. 平滑度/规则性
   - 标签开闭是否平衡，是否存在明显畸形标记。

输出
-------
- 一个取值范围在 [0,1] 的标量奖励。
- 用于分析和消融实验的详细子分数。
"""

from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# 常量与默认阈值定义
# ---------------------------------------------------------------------------

# 默认最短字符数，低于此长度视为退化输出
_DEFAULT_MIN_CHARS = 120
# 默认最长字符数，高于此长度可能包含重复或截断
_DEFAULT_MAX_CHARS = 4000
# 默认最大颜色数量，防止出现混乱的霓虹色板
_DEFAULT_MAX_COLORS = 12
# 默认最大坐标绝对值，超出该值视为明显越界
_DEFAULT_MAX_COORD = 260
# 默认画布中心区域框，格式为 (xmin, ymin, xmax, ymax)
_DEFAULT_CENTER_BOX = (30, 30, 226, 226)

# 允许出现的矢量标签集合
_VECTOR_TAGS = {
    "path", "circle", "ellipse", "rect", "polygon", "line",
    "g", "defs", "linearGradient", "radialGradient", "stop",
}


# ---------------------------------------------------------------------------
# 文本提取与清洗
# ---------------------------------------------------------------------------

def _extract_svg(text: str) -> tuple[str, bool, dict[str, Any]]:
    """从模型输出文本中提取 SVG，返回提取结果与诊断信息。"""
    text = text.strip()
    details: dict[str, Any] = {"has_svg": False}

    # 若模型输出了 markdown 代码块包裹，先去掉包裹
    has_code_fence = text.startswith("```")
    if has_code_fence:
        text = re.sub(r"^```.*\n", "", text)
        text = re.sub(r"```$", "", text).strip()

    # 要求恰好提取一个顶层 <svg>...</svg>，且其后的无意义文本越少越好
    matches = list(re.finditer(r"<svg\b[^>]*>.*?</svg>", text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return "", False, details

    svg = matches[0].group(0)
    details["has_svg"] = True
    details["svg_count"] = len(matches)
    details["has_code_fence"] = has_code_fence
    details["extra_text_before"] = bool(text[: matches[0].start()].strip())
    trailing = text[matches[0].end():].strip()
    details["extra_text_after"] = bool(trailing)
    details["trailing_text"] = trailing[:200]
    return svg, True, details


def _has_viewbox(svg: str) -> bool:
    """检查 SVG 根元素是否具备 viewBox 属性。"""
    return bool(re.search(r"<svg\b[^>]*\bviewBox\s*=\s*['\"][^'\"]+['\"]", svg, re.IGNORECASE))


def _has_xmlns(svg: str) -> bool:
    """检查 SVG 根元素是否具备 xmlns 命名空间声明。"""
    return bool(re.search(r"xmlns\s*=\s*['\"]http://www\.w3\.org/2000/svg['\"]", svg, re.IGNORECASE))


# ---------------------------------------------------------------------------
# 结构异常与畸形检测
# ---------------------------------------------------------------------------

def _malformed_penalty(svg: str) -> float:
    """检测模型是否幻觉出 XML/HTML 畸形标记或属性。"""
    penalty = 0.0
    # 模型常见幻觉标记，如残余控制符、语料内泄漏片段
    bad = ["<eos>", "<|eot_id|>", "<|end_of_text|>", "<core", "<tropical", "<coral"]
    penalty += sum(0.1 for token in bad if token in svg)
    # 畸形渐变引用或 URL / 属性
    if "url(#" in svg:
        penalty += 0.1
    if "fill-alpha=" in svg or 'runat="black"' in svg or 'stroke-turned=' in svg:
        penalty += 0.1
    # 明显的非 SVG 闭合标签
    if re.search(r"</(?!svg|g|defs|path|circle|rect|ellipse|polygon|line|linearGradient|radialGradient|stop|clipPath|mask)[a-zA-Z]+>", svg):
        penalty += 0.15
    return min(0.5, penalty)


def _count_tags(svg: str, tag: str) -> int:
    """统计 SVG 内某类标签出现的次数。"""
    return len(re.findall(rf"<{tag}\b", svg, re.IGNORECASE))


# ---------------------------------------------------------------------------
# 颜色相关
# ---------------------------------------------------------------------------

def _valid_color(color: str) -> bool:
    """判断颜色字符串是否为合法的 3/6 位十六进制颜色。"""
    return bool(re.fullmatch(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", color))


def _collect_colors(svg: str) -> set[str]:
    """从 fill、stroke、stop-color 等属性中提取颜色值。"""
    colors = set(re.findall(r"fill\s*=\s*['\"](#[0-9a-fA-F]{3,6})['\"]", svg, re.IGNORECASE))
    colors.update(re.findall(r"stroke\s*=\s*['\"](#[0-9a-fA-F]{3,6})['\"]", svg, re.IGNORECASE))
    colors.update(re.findall(r"stop-color\s*=\s*['\"](#[0-9a-fA-F]{3,6})['\"]", svg, re.IGNORECASE))
    return colors


# ---------------------------------------------------------------------------
# 坐标相关
# ---------------------------------------------------------------------------

def _collect_numeric_attrs(svg: str, attrs: tuple[str, ...]) -> list[float]:
    """从 SVG 属性中批量提取数值。"""
    vals: list[float] = []
    for attr in attrs:
        vals.extend(
            float(v)
            for v in re.findall(rf"{attr}\s*=\s*['\"']?(-?\d+(?:\.\d+)?)['\"']?", svg, re.IGNORECASE)
        )
    return vals


def _prompt_keywords(prompt: str, max_keywords: int = 10) -> set[str]:
    """从提示词中抽取内容关键词，过滤停用词后按词频截断。"""
    words = re.findall(r"[A-Za-z]{3,}", prompt.lower())
    stop = {
        "the", "and", "with", "for", "from", "that", "this", "into", "inside", "behind",
        "top", "bottom", "like", "over", "under", "into", "small", "thin", "soft", "warm",
        "deep", "large", "single", "thin", "clean", "bold", "simple", "small", "large",
        "center", "centered", "centred", "without", "along", "across", "just", "inside",
        "outside", "while", "with", "without", "across",
    }
    words = [w for w in words if w not in stop]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return {w for w, _ in ranked[:max_keywords]}


def _keyword_coverage(svg: str, keywords: set[str]) -> tuple[float, dict[str, bool]]:
    """计算 SVG 文本空间中对提示词关键词的覆盖程度。"""
    if not keywords:
        return 1.0, {}
    token_space = set(re.findall(r"[A-Za-z]{3,}", svg.lower()))
    hits = {kw: kw in token_space for kw in keywords}
    return sum(hits.values()) / len(keywords), hits


def _center_box_score(vals: list[float], box: tuple[int, int, int, int]) -> float:
    """检查数值坐标是否落在中心框内，偶数下标视为 x，奇数下标视为 y。"""
    if not vals:
        return 1.0
    xmin, ymin, xmax, ymax = box
    out = 0
    total = 0
    for i, v in enumerate(vals):
        total += 1
        if i % 2 == 0:
            if xmin <= v <= xmax:
                out += 1
        else:
            if ymin <= v <= ymax:
                out += 1
    return out / total if total else 1.0


def _truncation_score(svg: str) -> float:
    """根据开/闭标签比例判断是否存在截断或重复生成。"""
    open_count = len(re.findall(r"<([a-zA-Z][^/>]*)\b", svg))
    close_count = len(re.findall(r"</\s*([a-zA-Z][^>]*)\s*>", svg))
    if open_count == 0:
        return 1.0
    ratio = close_count / max(open_count, 1)
    return min(1.0, ratio) if ratio <= 1.0 else max(0.0, 1.0 / ratio)


def _element_diversity_score(svg: str) -> float:
    """衡量 SVG 中使用图元类型的丰富程度。"""
    present = {tag for tag in _VECTOR_TAGS if _count_tags(svg, tag) > 0}
    if not present:
        return 0.0
    return min(1.0, len(present) / 5.0)


# ---------------------------------------------------------------------------
# 主奖励计算逻辑
# ---------------------------------------------------------------------------


def compute_reward(
    text: str,
    prompt: str | None = None,
    *,
    min_chars: int = _DEFAULT_MIN_CHARS,
    max_chars: int = _DEFAULT_MAX_CHARS,
    max_colors: int = _DEFAULT_MAX_COLORS,
    max_coord: int = _DEFAULT_MAX_COORD,
    center_box: tuple[int, int, int, int] = _DEFAULT_CENTER_BOX,
) -> dict[str, Any]:
    """对单条模型输出进行多维度打分，返回奖励与诊断信息。"""
    subscores: dict[str, float] = {}
    details: dict[str, Any] = {}

    # 第一步：提取 SVG 并判断基本可解析性
    svg, ok, extract_details = _extract_svg(text)
    details.update(extract_details)
    if not ok:
        return {
            "reward": 0.0,
            "subscores": subscores,
            "details": details,
            "ok": False,
        }

    # 第二步：结构有效性，需要同时具备 viewBox 和 xmlns
    has_viewbox = _has_viewbox(svg)
    has_xmlns = _has_xmlns(svg)
    structure_ok = has_viewbox and has_xmlns
    subscores["valid_structure"] = float(structure_ok)
    details["has_viewbox"] = has_viewbox
    details["has_xmlns"] = has_xmlns

    # 第三步：清洗质量，代码块包裹或前后多余文本会扣分
    clean_score = 1.0
    if details.get("has_code_fence"):
        clean_score *= 0.7
    if details.get("extra_text_before") or details.get("extra_text_after"):
        clean_score *= 0.7
    subscores["clean_extraction"] = clean_score

    # 第四步：长度合理性，过短视为退化，过长视为重复或截断
    length = len(svg)
    if length < min_chars:
        length_score = max(0.0, length / max(min_chars, 1))
    elif length > max_chars:
        length_score = max(0.0, 1.0 - (length - max_chars) / max(max_chars, 1))
    else:
        length_score = 1.0
    subscores["length"] = max(0.0, min(1.0, length_score))

    # 第五步：配色合理性，颜色数量与格式均需受控
    colors = _collect_colors(svg)
    details["color_count"] = len(colors)
    color_score = 1.0 if colors else 0.0
    if len(colors) > max_colors:
        color_score *= max(0.0, 1.0 - (len(colors) - max_colors) / max(len(colors), 1))
    valid_color_ratio = 1.0 if not colors else sum(1 for c in colors if _valid_color(c)) / len(colors)
    subscores["palette"] = max(0.0, min(1.0, color_score * valid_color_ratio))

    # 第六步：坐标合理性，兼顾不出界与居中分布
    coords = _collect_numeric_attrs(svg, ("x", "y", "cx", "cy", "r", "rx", "ry", "width", "height"))
    out_of_bounds = [v for v in coords if v > max_coord or v < -max_coord]
    details["out_of_bounds_count"] = len(out_of_bounds)
    coord_score = 1.0 if not coords else max(0.0, 1.0 - len(out_of_bounds) / max(len(coords), 1))
    center_score = _center_box_score(coords, center_box)
    subscores["coordinates"] = max(0.0, min(1.0, 0.6 * coord_score + 0.4 * center_score))

    # 第七步：提示词覆盖度，关键词命中比例
    if prompt:
        keywords = _prompt_keywords(prompt)
        details["keywords"] = sorted(keywords)
        coverage, hits = _keyword_coverage(svg, keywords)
        subscores["prompt_coverage"] = coverage
        details["keyword_hits"] = hits
    else:
        subscores["prompt_coverage"] = 0.5  # 无提示词时给中性分

    # 第八步：图元多样性，鼓励使用多种基本图形
    subscores["element_diversity"] = _element_diversity_score(svg)
    details["element_types"] = sorted({tag for tag in _VECTOR_TAGS if _count_tags(svg, tag) > 0})

    # 第九步：平滑度/规则性，标签开闭平衡与截断惩罚
    tag_open = len(re.findall(r"<([a-zA-Z][^/>]*)\b", svg))
    tag_close = len(re.findall(r"</\s*([a-zA-Z][^>]*)\s*>", svg))
    balance = min(tag_open, tag_close) / max(tag_open, tag_close, 1)
    subscores["smoothness"] = max(0.0, min(1.0, balance * _truncation_score(svg)))

    # 最终加权聚合，输出 [0,1] 区间内的综合奖励
    weights = {
        "valid_structure": 0.18,
        "clean_extraction": 0.10,
        "length": 0.10,
        "palette": 0.10,
        "coordinates": 0.10,
        "prompt_coverage": 0.22,
        "element_diversity": 0.10,
        "smoothness": 0.10,
    }
    reward = sum(weights[k] * subscores[k] for k in weights)
    reward = max(0.0, min(1.0, reward))

    return {
        "reward": reward,
        "subscores": subscores,
        "details": details,
        "ok": ok and structure_ok,
    }


# ---------------------------------------------------------------------------
# 批量打分入口
# ---------------------------------------------------------------------------


def score_file(
    predictions_jsonl: str,
    ground_truth_jsonl: str | None = None,
) -> dict[str, Any]:
    """对 predictions_jsonl 逐条打分并汇总统计信息。"""
    rows: list[dict[str, Any]] = []
    with open(predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = __import__("json").loads(line)
            prompt = ex.get("prompt")
            reward_info = compute_reward(ex.get("text", ""), prompt=prompt)
            rows.append(
                {
                    "prompt": prompt,
                    "reward": reward_info["reward"],
                    "ok": reward_info["ok"],
                    "subscores": reward_info["subscores"],
                    "details": reward_info["details"],
                }
            )

    rewards = [r["reward"] for r in rows]
    validity = [1.0 if r["ok"] else 0.0 for r in rows]
    agg = {
        "count": len(rows),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "valid_rate": sum(validity) / len(validity) if validity else 0.0,
    }
    return {"aggregate": agg, "rows": rows}


if __name__ == "__main__":
    # 手工测试样例，覆盖正常 SVG、纯文本、截断、长重复等常见情形
    examples = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="128" cy="128" r="64" fill="#ffcc00"/></svg>',
        "not an svg at all",
        "<svg>broken",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">' + "<g>" * 200 + "</g>" * 200 + "</svg>",
    ]
    for i, ex in enumerate(examples, 1):
        out = compute_reward(ex, prompt="circle badge amber warm simple centered")
        print(f"Example {i}: reward={out['reward']:.3f} ok={out['ok']}")
