import json
import re
import ast
from typing import List, Optional
COMMON_LIST_KEYS = ["answers", "result", "results", "output", "outputs", "data", "list", "items"]

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    fence = re.compile(r"^\s*```(?:json|javascript|js|py|python)?\s*(.*?)\s*```\s*$", re.S | re.I)
    m = fence.match(s)
    if m:
        return m.group(1).strip()
    return s

def _normalize_quotes(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
    s = s.replace("‘", "'").replace("’", "'").replace("‚", "'")
    return s

def _extract_list_from_object(obj) -> Optional[List]:
    
    if isinstance(obj, dict):
        for k in COMMON_LIST_KEYS:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        list_values = [v for v in obj.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
    return None

def _json_loads_lenient(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def _find_first_bracket_array(s: str) -> Optional[str]:
    i, n = 0, len(s)
    in_str = False
    str_ch = ""
    esc = False
    stack = []
    start_idx = -1
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == str_ch:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                str_ch = ch
            elif ch == "[":
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch == "]":
                if stack:
                    stack.pop()
                    if not stack and start_idx != -1:
                        return s[start_idx:i+1]
        i += 1
    return None

def _fallback_bullets(s: str) -> List[str]:
    lines = s.splitlines()
    out = []
    bullet_re = re.compile(r"^\s*(?:[-*•]+|\d+[\.)])\s+(.*\S)\s*$")
    for line in lines:
        m = bullet_re.match(line)
        if m:
            out.append(m.group(1).strip())
    if not out:
        m = re.search(r"(?:Answer | Answer | Output | Final | Result)[:：]\s*(.+)$", s, re.I | re.S)
        cand = m.group(1).strip() if m else s.strip()
        if any(sep in cand for sep in ["；", ";", "、", "，", ","]):
            parts = re.split(r"[；;、，,]\s*", cand)
            parts = [p.strip() for p in parts if p.strip()]
            if parts and all(len(p) <= 200 for p in parts):
                out = parts
    return out

def _coerce_to_str_list(x) -> Optional[List[str]]:
    if isinstance(x, list):
        out = []
        for item in x:
            if item is None:
                continue
            out.append(item if isinstance(item, str) else str(item))
        return out
    return None

def parse_json_list(output: str) -> List[str]:
    if not isinstance(output, str):
        output = str(output)

    text = _strip_code_fences(_normalize_quotes(output)).strip()

    parsed = _json_loads_lenient(text)
    if parsed is not None:
        if isinstance(parsed, dict):
            lst = _extract_list_from_object(parsed)
            if lst is not None:
                co = _coerce_to_str_list(lst)
                if co:
                    return _post_clean(co)
        co = _coerce_to_str_list(parsed)
        if co:
            return _post_clean(co)

    arr = _find_first_bracket_array(text)
    if arr:
        parsed = _json_loads_lenient(arr)
        if parsed is None:
            inner = arr[1:-1]
            if '"' not in inner and "'" in inner:
                arr2 = arr.replace("'", '"')
                parsed = _json_loads_lenient(arr2)
        if parsed is not None:
            co = _coerce_to_str_list(parsed)
            if co:
                return _post_clean(co)

    obj_match = re.search(r"\{.*\}", text, re.S)
    if obj_match:
        cand = obj_match.group(0)
        parsed = _json_loads_lenient(cand)
        if isinstance(parsed, dict):
            lst = _extract_list_from_object(parsed)
            co = _coerce_to_str_list(lst) if lst is not None else None
            if co:
                return _post_clean(co)

    bullets = _fallback_bullets(text)
    if bullets:
        return _post_clean(bullets)

    quoted = re.findall(r'"([^"\n]{1,200})"|\'([^\'\n]{1,200})\'', text)
    flat = [a or b for (a, b) in quoted]
    if flat:
        return _post_clean(flat)

    return []

def _post_clean(items: List[str]) -> List[str]:
    cleaned = []
    for s in items:
        t = s.strip()
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"\s*\[\[?\d+\]?\]\s*$", "", t)
        t = re.sub(r"^(?:Answer | Answer | Output | Final | Result)[:：]\s*", "", t, flags=re.I)
        if t:
            cleaned.append(t)
    seen = set()
    out = []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out