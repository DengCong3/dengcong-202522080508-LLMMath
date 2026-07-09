import re
import json
from typing import Dict, Any, List
import xml.etree.ElementTree as ET

# ====================== 全局超参数 ======================
# 权重：语法0.3 / 结构0.2 / 防退化0.15 / 语义0.35
WEIGHT_SYNTAX = 0.30
WEIGHT_STRUCT = 0.20
WEIGHT_NO_DEGRADE = 0.15
WEIGHT_SEMANTIC = 0.35
# 总分缩放 0~1原始加权分 → 0~10展示分
SCORE_SCALE = 10.0

# SVG规范常量
SVG_XMLNS = "http://www.w3.org/2000/svg"
VIEWBOX_STD = "0 0 256 256"
ALLOW_TAGS = {"svg", "defs", "g", "path", "circle", "ellipse", "rect", "polygon", "line"}
FORBIDDEN_TAGS = {"image", "script", "iframe", "a"}
MIN_ELEMENTS = 5
MAX_ELEMENTS = 120
CANVAS_MIN = 16
CANVAS_MAX = 240

# 语义匹配映射：关键词→对应SVG标签集合（解决star只用polygon绘制匹配不到问题）
SHAPE_MAPPING = {
    "star": {"polygon"},
    "circle": {"circle"},
    "square": {"rect"},
    "shield": {"path", "polygon"},
    "rocket": {"path"},
    "leaf": {"path"},
    "mountain": {"path", "polygon"},
    "wheat": {"path"},
    "coffee": {"path", "ellipse"},
    "beer": {"path", "polygon"},
    "cheese": {"path", "polygon"},
    "house": {"rect", "polygon"},
    "car": {"path", "rect"}
}
# 颜色文本关键词映射标准十六进制
COLOR_WORD_MAP = {
    "navy": "#1B3A5C",
    "gold": "#FFD700",
    "red": "#FF0000",
    "white": "#FFFFFF",
    "black": "#000000",
    "green": "#008000",
    "blue": "#0000FF"
}

class LogoSVGReward:
    def parse_safe_svg(self, svg_str: str) -> tuple[bool, ET.Element | None, str]:
        try:
            svg_str = svg_str.strip()
            if not svg_str.startswith("<svg") or not svg_str.endswith("</svg>"):
                return False, None, "Not wrapped by single <svg> tag"
            tree = ET.ElementTree(ET.fromstring(svg_str))
            root = tree.getroot()
            tag_raw = root.tag
            if "}" in tag_raw:
                tag = tag_raw.split("}")[-1]
            else:
                tag = tag_raw
            if tag != "svg":
                return False, None, "Root tag is not svg"
            return True, root, ""
        except Exception as e:
            return False, None, f"XML parse error: {str(e)[:100]}"

    def check_syntax_rule(self, root: ET.Element) -> tuple[float, List[str]]:
        score = 1.0
        logs = []
        # 命名空间
        xmlns_val = root.attrib.get("xmlns", "")
        if SVG_XMLNS not in xmlns_val:
            score -= 0.25
            logs.append(f"Missing standard xmlns={SVG_XMLNS}")
        # viewBox
        vb = root.attrib.get("viewBox", "")
        if vb != VIEWBOX_STD:
            score -= 0.2
            logs.append(f"Invalid viewBox: {vb}, required {VIEWBOX_STD}")
        # 禁止标签
        bad_tags = set()
        for elem in root.iter():
            tag_raw = elem.tag
            t = tag_raw.split("}")[-1] if "}" in tag_raw else tag_raw
            if t in FORBIDDEN_TAGS:
                bad_tags.add(t)
        if bad_tags:
            score -= 0.3
            logs.append(f"Forbidden tags exist: {bad_tags}")
        return max(score, 0.0), logs

    def check_struct_rule(self, root: ET.Element, svg_text: str) -> tuple[float, List[str]]:
        score = 1.0
        logs = []
        all_elems = list(root.iter())
        elem_count = len(all_elems) - 1
        # 元素数量梯度扣分
        if elem_count < MIN_ELEMENTS:
            deduct = 0.08 * (MIN_ELEMENTS - elem_count)
            score -= deduct
            logs.append(f"Primitive elements too few({elem_count})")
        if elem_count > MAX_ELEMENTS:
            deduct = 0.002 * (elem_count - MAX_ELEMENTS)
            score -= deduct
            logs.append(f"Excessive primitives({elem_count})")
        # 渐变必须在defs内
        grad_nodes = list(root.findall(".//linearGradient") + root.findall(".//radialGradient"))
        grad_outside = [g for g in grad_nodes if g.getparent() and g.getparent().tag.split("}")[-1] != "defs"]
        if grad_outside:
            score -= 0.15
            logs.append(f"{len(grad_outside)} gradients outside <defs>")
        # 坐标越界比例扣分
        coords = re.findall(r"\d+(\.\d+)?", svg_text)
        nums = [float(c) for c in coords if float(c) > 0]
        if nums:
            out_range = [x for x in nums if x < CANVAS_MIN or x > CANVAS_MAX]
            out_ratio = len(out_range) / len(nums)
            if out_ratio > 0.1:
                score -= 0.2 * out_ratio
                logs.append(f"Coord out of canvas ratio: {out_ratio:.2f}")
        return max(score, 0.0), logs

    def detect_degradation(self, svg_str: str) -> tuple[float, List[str]]:
        score = 1.0
        logs = []
        svg_clean = svg_str.replace("\n", "").replace(" ", "")
        # 过短空SVG
        if len(svg_clean) < 150:
            score -= 0.6
            logs.append("SVG too short, degenerate empty output")
        # 大量重复path坍缩
        repeat_pattern = re.search(r"(<path.*?>){15,}", svg_clean)
        if repeat_pattern:
            score -= 0.4
            logs.append("Mass repeated path, model collapse")
        # 超长冗余代码
        if len(svg_clean) > 12000:
            score -= 0.3
            logs.append("Overlong meaningless SVG")
        # 新增退化：全透明图形过多
        if svg_clean.count('fill-opacity="0"') > 8:
            score -= 0.2
            logs.append("Too many fully transparent shapes")
        return max(score, 0.0), logs

    def calc_semantic_match(self, prompt: str, root: ET.Element, svg_text: str) -> tuple[float, List[str]]:
        logs = []
        prompt_lower = prompt.lower()
        svg_lower = svg_text.lower()

        # 1.颜色匹配：十六进制 + 颜色单词映射
        prompt_color_hex = re.findall(r"#([0-9A-Fa-f]{3,6})", prompt)
        prompt_color_words = [w for w in COLOR_WORD_MAP.keys() if w in prompt_lower]
        target_color_codes = set(prompt_color_hex)
        for w in prompt_color_words:
            target_color_codes.add(COLOR_WORD_MAP[w].strip("#"))
        svg_hex = set(re.findall(r"#([0-9A-Fa-f]{3,6})", svg_text))
        hit_color = sum(1 for c in target_color_codes if c in svg_hex)
        color_rate = hit_color / len(target_color_codes) if target_color_codes else 1.0

        # 2.形状宽松匹配（关键词映射标签，不再严格相等）
        prompt_shape_words = [k for k in SHAPE_MAPPING.keys() if k in prompt_lower]
        svg_tags = set()
        for e in root.iter():
            t_raw = e.tag
            t = t_raw.split("}")[-1] if "}" in t_raw else t_raw
            svg_tags.add(t)
        hit_shape = 0
        for word in prompt_shape_words:
            match_tags = SHAPE_MAPPING[word]
            if match_tags & svg_tags:
                hit_shape += 1
        shape_rate = hit_shape / len(prompt_shape_words) if prompt_shape_words else 1.0

        # 3.布局匹配：圆形徽章宽松判定
        layout_hit = 0
        if "circle badge" in prompt_lower or "medallion" in prompt_lower:
            if "circle" in svg_tags:
                layout_hit = 1
        layout_rate = layout_hit / 1 if ("circle badge" in prompt_lower or "medallion" in prompt_lower) else 1.0

        semantic_total = 0.4 * color_rate + 0.4 * shape_rate + 0.2 * layout_rate
        logs.append(f"Color:{color_rate:.2f} Shape:{shape_rate:.2f} Layout:{layout_rate:.2f}")
        return round(semantic_total, 3), logs

    def compute_total_reward(self, prompt: str, svg_output: str) -> Dict[str, Any]:
        res = {
            "total_reward_raw": 0.0,    # 0~1原始加权分
            "total_reward": 0.0,        # 0~10展示分
            "syntax_score": 0.0,
            "struct_score": 0.0,
            "degrade_score": 0.0,
            "semantic_score": 0.0,
            "logs": []
        }
        valid, root, parse_log = self.parse_safe_svg(svg_output)
        # 新增兜底：XML解析失败时使用简易正则粗打分，不直接归零
        if not valid:
            res["logs"].append(f"Parse Failed Fallback: {parse_log}")
            rough_syntax = 0.0
            if "<svg" in svg_output and "</svg>" in svg_output:
                rough_syntax += 0.3
            if SVG_XMLNS in svg_output:
                rough_syntax += 0.2
            if VIEWBOX_STD in svg_output:
                rough_syntax += 0.2
            tag_cnt = len(re.findall(r"<(path|circle|ellipse|rect|polygon|line|g)", svg_output))
            if 3 <= tag_cnt <= 150:
                rough_syntax += 0.15
            # 简易语义匹配
            prompt_words = set(re.findall(r"[a-z0-9#]+", prompt.lower()))
            svg_words = set(re.findall(r"[a-z0-9#]+", svg_output.lower()))
            overlap = prompt_words & svg_words
            sem_rough = 0.15 * min(len(overlap)/max(len(prompt_words),1), 1.0)
            rough_syntax = min(rough_syntax, 1.0)

            res["syntax_score"] = rough_syntax
            res["struct_score"] = rough_syntax * 0.8
            res["degrade_score"] = 0.5 if len(svg_output) > 100 else 0.2
            res["semantic_score"] = sem_rough

            total_raw = WEIGHT_SYNTAX * res["syntax_score"] + WEIGHT_STRUCT * res["struct_score"] + WEIGHT_NO_DEGRADE * res["degrade_score"] + WEIGHT_SEMANTIC * res["semantic_score"]
            total_raw = max(min(total_raw, 1.0), 0.0)
            res["total_reward_raw"] = round(total_raw, 4)
            res["total_reward"] = round(total_raw * SCORE_SCALE, 2)
            return res

        # XML解析成功，走完整结构化打分
        s_score, s_log = self.check_syntax_rule(root)
        st_score, st_log = self.check_struct_rule(root, svg_output)
        d_score, d_log = self.detect_degradation(svg_output)
        se_score, se_log = self.calc_semantic_match(prompt, root=root, svg_text=svg_output)

        res["syntax_score"] = s_score
        res["struct_score"] = st_score
        res["degrade_score"] = d_score
        res["semantic_score"] = se_score
        res["logs"].extend(s_log + st_log + se_log)

        total_raw = WEIGHT_SYNTAX * s_score + WEIGHT_STRUCT * st_score + WEIGHT_NO_DEGRADE * d_score
        total_raw = max(min(total_raw, 1.0), 0.0)
        res["total_reward_raw"] = round(total_raw, 4)
        res["total_reward"] = round(total_raw * SCORE_SCALE, 2)
        return res

reward_calculator = LogoSVGReward()

# 对外统一接口：直接返回0~10总分，供eval调用
def calculate_reward(prompt: str, svg_text: str) -> float:
    res = reward_calculator.compute_total_reward(prompt, svg_text)
    return res["total_reward"]

# 子维度完整打分接口（eval可按需读取分项）
def get_full_reward_detail(prompt: str, svg_text: str) -> dict:
    return reward_calculator.compute_total_reward(prompt, svg_text)

if __name__ == "__main__":
    test_prompt = "A circular badge deep navy, golden star in center"
    test_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="128" cy="128" r="100" fill="#1B3A5C"/><polygon points="128,80 135,100 156,100 140,114 144,136 128,126 112,136 116,114 100,100 121,100"/></svg>'''
    out = reward_calculator.compute_total_reward(test_prompt, test_svg)
    print(json.dumps(out, indent=2, ensure_ascii=False))