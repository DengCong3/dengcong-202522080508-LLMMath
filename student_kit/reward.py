"""Robust programmatic SVG reward/validation for logo generation.

Design goals
------------
- Be a *training proxy*: cheap, deterministic, and sensitive to common failure modes
  of a tiny decoder (Gemma 3 270M) after LoRA fine-tuning.
- Avoid overfitting to superficial markers: reward should reflect whether the SVG
  is structurally valid, visually plausible, and responsive to the prompt.
- Be explainable in the final report: each subscore has a clear rationale.

Scoring dimensions
------------------
1. Clean extraction
   - Is there exactly one top-level <svg>?
   - No markdown/code fences, no extra prose before/after.
2. Structural validity
   - Does it close properly?
   - Is viewBox and xmlns present?
3. Completeness / no degeneration
   - Length in tokens/chars is within a sensible range.
   - No truncation artifacts (e.g., dangling unclosed tags, repeated fragments).
4. Prompt coverage
   - Prompt keywords appear in SVG tags/attributes/content.
5. Element diversity
   - Uses a reasonable mix of vector primitives.
6. Palette sanity
   - Colors used are mostly valid and within a limited set.
7. Coordinate sanity
   - Elements stay roughly inside the intended canvas center region.
8. Smoothness / regularity
   - Tag balance and no obvious malformed markers.

Outputs
-------
- A scalar reward in [0, 1].
- Detailed subscores for analysis and ablation.
"""

from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_MIN_CHARS = 120
_DEFAULT_MAX_CHARS = 4000
_DEFAULT_MAX_COLORS = 12
_DEFAULT_MAX_COORD = 260
_DEFAULT_CENTER_BOX = (30, 30, 226, 226)  # xmin, ymin, xmax, ymax

_VECTOR_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "line", "g", "defs", "linearGradient", "radialGradient", "stop"}


def _extract_svg(text: str) -> tuple[str, bool, dict[str, Any]]:
    """Return (svg_content, ok, details)."""
    text = text.strip()
    details: dict[str, Any] = {"has_svg": False}

    # detect obvious wrappers
    has_code_fence = text.startswith("```")
    if has_code_fence:
        text = re.sub(r"^```.*\n", "", text)
        text = re.sub(r"```$", "", text).strip()

    m = re.search(r"<svg\b[^>]*>.*?</svg>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return "", False, details

    svg = m.group(0)
    details["has_svg"] = True
    details["has_code_fence"] = has_code_fence
    details["extra_text_before"] = bool(text[: m.start()].strip())
    details["extra_text_after"] = bool(text[m.end():].strip())
    return svg, True, details


def _has_viewbox(svg: str) -> bool:
    return bool(re.search(r"<svg\b[^>]*\bviewBox\s*=\s*['\"][^'\"]+['\"]", svg, re.IGNORECASE))


def _has_xmlns(svg: str) -> bool:
    return bool(re.search(r"xmlns\s*=\s*['\"]http://www\.w3\.org/2000/svg['\"]", svg, re.IGNORECASE))


def _count_tags(svg: str, tag: str) -> int:
    return len(re.findall(rf"<{tag}\b", svg, re.IGNORECASE))


def _valid_color(color: str) -> bool:
    return bool(re.fullmatch(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", color))


def _collect_colors(svg: str) -> set[str]:
    colors = set(re.findall(r"fill\s*=\s*['\"](#[0-9a-fA-F]{3,6})['\"]", svg, re.IGNORECASE))
    colors.update(re.findall(r"stroke\s*=\s*['\"](#[0-9a-fA-F]{3,6})['\"]", svg, re.IGNORECASE))
    colors.update(re.findall(r"stop-color\s*=\s*['\"](#[0-9a-fA-F]{3,6})['\"]", svg, re.IGNORECASE))
    return colors


def _collect_numeric_attrs(svg: str, attrs: tuple[str, ...]) -> list[float]:
    vals: list[float] = []
    for attr in attrs:
        vals.extend(
            float(v)
            for v in re.findall(rf"{attr}\s*=\s*['\"']?(-?\d+(?:\.\d+)?)['\"']?", svg, re.IGNORECASE)
        )
    return vals


def _prompt_keywords(prompt: str, max_keywords: int = 10) -> set[str]:
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
    if not keywords:
        return 1.0, {}
    token_space = set(re.findall(r"[A-Za-z]{3,}", svg.lower()))
    hits = {kw: kw in token_space for kw in keywords}
    return sum(hits.values()) / len(keywords), hits


def _center_box_score(vals: list[float], box: tuple[int, int, int, int]) -> float:
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
    open_count = len(re.findall(r"<([a-zA-Z][^/>]*)\b", svg))
    close_count = len(re.findall(r"</\s*([a-zA-Z][^>]*)\s*>", svg))
    if open_count == 0:
        return 1.0
    ratio = close_count / max(open_count, 1)
    return min(1.0, ratio) if ratio <= 1.0 else max(0.0, 1.0 / ratio)


def _element_diversity_score(svg: str) -> float:
    present = {tag for tag in _VECTOR_TAGS if _count_tags(svg, tag) > 0}
    if not present:
        return 0.0
    # reward using at least 2 distinct element types, capped at 5
    return min(1.0, len(present) / 5.0)


# ---------------------------------------------------------------------------
# Main reward
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
    """Compute reward components for one model output.

    Returns a dict with:
      - reward: float in [0,1] (weighted composite)
      - subscores: dict of named subscores
      - details: extra diagnostics
      - ok: bool whether basic validity holds
    """
    subscores: dict[str, float] = {}
    details: dict[str, Any] = {}

    # 0) Extraction / cleanliness
    svg, ok, extract_details = _extract_svg(text)
    details.update(extract_details)
    if not ok:
        return {
            "reward": 0.0,
            "subscores": subscores,
            "details": details,
            "ok": False,
        }

    # 1) Structural validity
    has_viewbox = _has_viewbox(svg)
    has_xmlns = _has_xmlns(svg)
    structure_ok = has_viewbox and has_xmlns
    subscores["valid_structure"] = float(structure_ok)
    details["has_viewbox"] = has_viewbox
    details["has_xmlns"] = has_xmlns

    # clean extraction penalties
    clean_score = 1.0
    if details.get("has_code_fence"):
        clean_score *= 0.7
    if details.get("extra_text_before") or details.get("extra_text_after"):
        clean_score *= 0.7
    subscores["clean_extraction"] = clean_score

    # 2) Length / degeneration
    length = len(svg)
    if length < min_chars:
        length_score = max(0.0, length / max(min_chars, 1))
    elif length > max_chars:
        length_score = max(0.0, 1.0 - (length - max_chars) / max(max_chars, 1))
    else:
        length_score = 1.0
    subscores["length"] = max(0.0, min(1.0, length_score))

    # 3) Palette sanity
    colors = _collect_colors(svg)
    details["color_count"] = len(colors)
    color_score = 1.0 if colors else 0.0
    if len(colors) > max_colors:
        color_score *= max(0.0, 1.0 - (len(colors) - max_colors) / max(len(colors), 1))
    valid_color_ratio = 1.0 if not colors else sum(1 for c in colors if _valid_color(c)) / len(colors)
    subscores["palette"] = max(0.0, min(1.0, color_score * valid_color_ratio))

    # 4) Coordinate sanity / center bias
    coords = _collect_numeric_attrs(svg, ("x", "y", "cx", "cy", "r", "rx", "ry", "width", "height"))
    out_of_bounds = [v for v in coords if v > max_coord or v < -max_coord]
    details["out_of_bounds_count"] = len(out_of_bounds)
    coord_score = 1.0 if not coords else max(0.0, 1.0 - len(out_of_bounds) / max(len(coords), 1))
    center_score = _center_box_score(coords, center_box)
    subscores["coordinates"] = max(0.0, min(1.0, 0.6 * coord_score + 0.4 * center_score))

    # 5) Prompt coverage
    if prompt:
        keywords = _prompt_keywords(prompt)
        details["keywords"] = sorted(keywords)
        coverage, hits = _keyword_coverage(svg, keywords)
        subscores["prompt_coverage"] = coverage
        details["keyword_hits"] = hits
    else:
        subscores["prompt_coverage"] = 0.5  # neutral when prompt absent

    # 6) Element diversity
    subscores["element_diversity"] = _element_diversity_score(svg)
    details["element_types"] = sorted({tag for tag in _VECTOR_TAGS if _count_tags(svg, tag) > 0})

    # 7) Smoothness / regularity
    tag_open = len(re.findall(r"<([a-zA-Z][^/>]*)\b", svg))
    tag_close = len(re.findall(r"</\s*([a-zA-Z][^>]*)\s*>", svg))
    balance = min(tag_open, tag_close) / max(tag_open, tag_close, 1)
    subscores["smoothness"] = max(0.0, min(1.0, balance * _truncation_score(svg)))

    # Composite
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


def score_file(
    predictions_jsonl: str,
    ground_truth_jsonl: str | None = None,
) -> dict[str, Any]:
    """Score a predictions jsonl (with 'text' field) and aggregate stats.

    Each line is expected to be a JSON object with at least:
      - "text": generated output
      - optionally "prompt": prompt text

    Returns aggregate metrics and per-example rows.
    """
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
    examples = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="128" cy="128" r="64" fill="#ffcc00"/></svg>',
        "not an svg at all",
        "<svg>broken",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">' + "<g>" * 200 + "</g>" * 200 + "</svg>",
    ]
    for i, ex in enumerate(examples, 1):
        out = compute_reward(ex, prompt="circle badge amber warm simple centered")
        print(f"Example {i}: reward={out['reward']:.3f} ok={out['ok']}")
