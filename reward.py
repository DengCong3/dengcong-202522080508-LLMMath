import re
import json
from typing import Dict, Any, List
import xml.etree.ElementTree as ET

# ====================== 权重超参数 ======================
WEIGHT_SYNTAX = 0.30
WEIGHT_STRUCT = 0.20
WEIGHT_NO_DEGRADE = 0.15
WEIGHT_SEMANTIC = 0.35

# SVG 规范常量
ALLOW_TAGS = {"svg", "defs", "g", "path", "circle", "ellipse", "rect", "polygon", "line"}
FORBIDDEN_TAGS = {"image", "script", "iframe", "a"}
MIN_ELEMENTS = 5
MAX_ELEMENTS = 120
VIEWBOX_STD = "0 0 256 256"
CANVAS_MIN = 16
CANVAS_MAX = 240

class LogoSVGReward:
    def parse_safe_svg(self, svg_str: str) -> tuple[bool, ET.Element | None, str]:
        try:
            svg_str = svg_str.strip()
            if not svg_str.startswith("<svg") or not svg_str.endswith("</svg>"):
                return False, None, "Not wrapped by single <svg> tag"
            tree = ET.ElementTree(ET.fromstring(svg_str))
            root = tree.getroot()
            if root.tag.split("}")[-1] != "svg":
                return False, None, "Root tag is not svg"
            return True, root, ""
        except Exception as e:
            return False, None, f"XML parse error: {str(e)[:100]}"

    def check_syntax_rule(self, root: ET.Element) -> tuple[float, List[str]]:
        score = 1.0
        logs = []
        if "xmlns" not in root.attrib:
            score -= 0.3
            logs.append("Missing xmlns attribute")
        vb = root.attrib.get("viewBox", "")
        if vb != VIEWBOX_STD:
            score -= 0.25
            logs.append(f"Invalid viewBox: {vb}, need {VIEWBOX_STD}")
        bad_tags = set()
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            if tag in FORBIDDEN_TAGS:
                bad_tags.add(tag)
        if bad_tags:
            score -= 0.35
            logs.append(f"Forbidden tags found: {bad_tags}")
        return max(score, 0.0), logs

    def check_struct_rule(self, root: ET.Element, svg_text: str) -> tuple[float, List[str]]:
        score = 1.0
        logs = []
        all_elems = list(root.iter())
        elem_count = len(all_elems) - 1
        if elem_count < MIN_ELEMENTS:
            score -= 0.4
            logs.append(f"Too few primitives({elem_count})")
        if elem_count > MAX_ELEMENTS:
            score -= 0.3
            logs.append(f"Excessive elements({elem_count})")
        defs_node = root.find(".//defs")
        grad_nodes = list(root.findall(".//linearGradient") + root.findall(".//radialGradient"))
        grad_outside = [g for g in grad_nodes if g.getparent() and g.getparent().tag.split("}")[-1] != "defs"]
        if grad_outside:
            score -= 0.2
            logs.append(f"{len(grad_outside)} gradients outside <defs>")
        coords = re.findall(r"\d+(\.\d+)?", svg_text)
        nums = [float(c) for c if float(c) > 0]
        out = [x for x in nums if x < CANVAS_MIN or x > CANVAS_MAX]
        if len(out) / (len(nums)+1) > 0.1:
            score -= 0.25
            logs.append("Many coords out of canvas 16~240")
        return max(score, 0.0), logs

    def detect_degradation(self, svg_str: str) -> tuple[float, List[str]]:
        score = 1.0
        logs = []
        svg_clean = svg_str.replace("\n", "").replace(" ", "")
        if len(svg_clean) < 150:
            score -= 0.6
            logs.append("SVG too short, degenerate empty output")
        repeat_pattern = re.search(r"(<path.*?>){15,}", svg_clean)
        if repeat_pattern:
            score -= 0.4
            logs.append("Mass repeated path, model collapse")
        if len(svg_clean) > 12000:
            score -= 0.3
            logs.append("Overlong meaningless SVG")
        return max(score, 0.0), logs

    def calc_semantic_match(self, prompt: str, root: ET.Element, svg_text: str) -> tuple[float, List[str]]:
        logs = []
        color_hex = re.findall(r"#([0-9A-Fa-f]{3,6})", prompt)
        svg_hex = set(re.findall(r"#([0-9A-Fa-f]{3,6})", svg_text))
        hit_color = sum(1 for c in color_hex if c in svg_hex)
        color_rate = hit_color / len(color_hex) if color_hex else 1.0
        shape_keywords = {"circle", "square", "shield", "book", "rocket", "leaf", "star", "mountain", "wheat", "coffee", "beer", "cheese", "house", "car"}
        prompt_words = set(re.findall(r"[a-z]+", prompt.lower()))
        svg_tags = {e.tag.split("}")[-1] for e in root.iter()}
        hit_shape = len(prompt_words & shape_keywords & svg_tags)
        total_shape = len(prompt_words & shape_keywords)
        shape_rate = hit_shape / total_shape if total_shape else 1.0
        layout_words = {"circle badge", "medallion", "centered"}
        layout_hit = sum(1 for w in layout_words if w in prompt.lower() and "circle" in svg_tags)
        layout_rate = layout_hit / len(layout_words) if any(lw in prompt for lw in layout_words) else 1.0
        semantic_total = 0.4 * color_rate + 0.4 * shape_rate + 0.2 * layout_rate
        logs.append(f"Color:{color_rate:.2f} Shape:{shape_rate:.2f} Layout:{layout_rate:.2f}")
        return round(semantic_total, 3), logs

    def compute_total_reward(self, prompt: str, svg_output: str) -> Dict[str, Any]:
        res = {
            "total_reward": 0.0,
            "syntax_score": 0.0,
            "struct_score": 0.0,
            "degrade_score": 0.0,
            "semantic_score": 0.0,
            "logs": []
        }
        valid, root, parse_log = self.parse_safe_svg(svg_output)
        if not valid:
            res["logs"].append(f"Parse Failed: {parse_log}")
            return res
        s_score, s_log = self.check_syntax_rule(root)
        st_score, st_log = self.check_struct_rule(root, svg_output)
        d_score, d_log = self.detect_degradation(svg_output)
        se_score, se_log = self.calc_semantic_match(prompt, svg_text=svg_output, root=root)
        res["syntax_score"] = s_score
        res["struct_score"] = st_score
        res["degrade_score"] = d_score
        res["semantic_score"] = se_score
        res["logs"].extend(s_log + st_log + d_log + se_log)
        total = WEIGHT_SYNTAX * s_score + WEIGHT_STRUCT * st_score + WEIGHT_NO_DEGRADE * d_score + WEIGHT_SEMANTIC * se_score
        res["total_reward"] = round(max(total, 0.0), 4)
        return res

reward_calculator = LogoSVGReward()

def calculate_reward(prompt: str, svg_text: str) -> float:
    res = reward_calculator.compute_total_reward(prompt, svg_text)
    return res["total_reward"]

if __name__ == "__main__":
    test_prompt = "A circular badge deep navy, golden star in center"
    test_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="128" cy="128" r="100" fill="#1B3A5C"/><polygon points="128,80 135,100 156,100 140,114 144,136 128,126 112,136 116,114 100,100 121,100"/></svg>'''
    out = reward_calculator.compute_total_reward(test_prompt, test_svg)
    print(json.dumps(out, indent=2, ensure_ascii=False))
