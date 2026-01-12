"""
Input a Markdown file, extract the lists from it, and return them as a list.
"""

import re
from typing import List, Tuple
import json
import bisect

LIST_ITEM_RE = re.compile(r'^(\s*)([*+-]|\d+[.)])\s+(.*)$')
FENCE_RE = re.compile(r'^\s*(```|~~~)')  # 进入/退出代码围栏
HEADING_RE = re.compile(r'^\s{0,3}(#{1,6})\s*(.+?)\s*(#+\s*)?$') 
SETEXT_UNDERLINE_RE = re.compile(r'^\s{0,3}(=+|-+)\s*$')    

def _clean_heading_text(text: str) -> str:
    """清洗标题文本：去掉链接/图片标记、括号URL、重复空白等。"""
    text = remove_parentheses_by_prefixes(text)
    text = re.sub(r'!\[|\[|\]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _scan_headings(lines: List[str]) -> List[Tuple[int, str]]:
    """
    Scans and returns a list of headings [(line_no, heading_text)], ignoring headings within code fences.
    Supports both ATX (#) and Setext (=/-) headings.
    line_no is 0-based.
    """
    headings: List[Tuple[int, str]] = []
    in_fence = False

    for i, raw in enumerate(lines):
        line = raw.rstrip('\n')

        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        m = HEADING_RE.match(line)
        if m:
            heading_text = _clean_heading_text(m.group(2))
            if heading_text:
                headings.append((i, heading_text))
            continue

        if SETEXT_UNDERLINE_RE.match(line) and i > 0:
            prev = lines[i-1].rstrip('\n')
            if prev.strip() and not FENCE_RE.match(prev) and not HEADING_RE.match(prev):
                heading_text = _clean_heading_text(prev)
                if heading_text:
                    headings.append((i-1, heading_text))

    headings.sort(key=lambda x: x[0])
    return headings

def extract_lists_with_nearest_heading(markdown_path: str) -> List[List[object]]:
    """
    Read the Markdown file, extract the "list blocks," and locate the nearest heading above each block.
    Return: [[idx, heading_text], ...], where idx is the 1-based index of the list block.
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        md = f.read()

    lines = md.splitlines()
    headings = _scan_headings(lines)
    heading_lines = [ln for ln, _ in headings] 

    in_fence = False
    blocks: List[List[str]] = []
    block_starts: List[int] = [] 

    current_block: List[str] = []
    current_item_parts: List[str] = []
    current_item_prefix: str = ""
    current_block_start_line: int = -1

    def flush_item():
        nonlocal current_item_parts, current_block
        if not current_item_parts:
            return
        item = " ".join(part.strip() for part in current_item_parts if part is not None)
        item = re.sub(r'\s+', ' ', item).strip()
        if item:
            current_block.append(current_item_prefix + item)
        current_item_parts = []

    def flush_block():
        nonlocal current_block, blocks, current_block_start_line
        flush_item()
        if current_block:
            for j in range(len(current_block)):
                cur = current_block[j]
                cur = remove_parentheses_by_prefixes(cur)
                cur = re.sub(r"\[|\]|\!", "", cur)
                cur = re.sub(r"\*\*\^\*\*", "", cur)
                current_block[j] = cur
            blocks.append(current_block)
            block_starts.append(current_block_start_line if current_block_start_line >= 0 else 0)
        current_block = []
        current_block_start_line = -1

    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")

        if FENCE_RE.match(line):
            in_fence = not in_fence
            if in_fence:
                flush_block()
            continue
        if in_fence:
            continue

        start = _is_list_start(line)
        if start:
            indent, marker, text = start
            if not current_block:
                current_block_start_line = i
            if current_item_parts:
                flush_item()
            current_item_prefix = f"{marker} "
            current_item_parts = [text]
            continue

        if _should_join_prev_item(line):
            if current_item_parts:
                current_item_parts.append(line)
                continue
            else:
                flush_block()
        else:
            flush_block()

    flush_block()

    results: List[List[object]] = []
    for idx, start_line in enumerate(block_starts, start=1):
        pos = bisect.bisect_left(heading_lines, start_line) - 1
        heading_text = ""
        if pos >= 0:
            heading_text = headings[pos][1]
        results.append([idx, heading_text])

    return results


def _is_list_start(line: str):
    m = LIST_ITEM_RE.match(line)
    if not m:
        return None
    indent = len(m.group(1).expandtabs(4)) 
    marker = m.group(2)
    text = m.group(3)
    return indent, marker, text

def _should_join_prev_item(line: str) -> bool:
    """
    Determine if this line should be treated as a continuation of the previous list item:
    - It is not the start of a new list item.
    - It is not a blank line (blank lines usually end the list item/block).
    - Common continuation lines: lines indented with at least 1-2 spaces, or lines with the same indentation as the previous item but without a matching list marker.
    A lenient strategy is adopted here: as long as it's not the start of a list item and not entirely blank, it will be appended.
    """
    return bool(line.strip()) and _is_list_start(line) is None

def remove_paren_urls(text: str, wiki_only: bool = False) -> str:
    """
    Remove URLs enclosed in parentheses. 
    - wiki_only=True: Only remove URLs containing "/wiki" (original requirement)
    - wiki_only=False: Remove any URL starting with http(s):// or //
    """
    if wiki_only:
        pattern = re.compile(r'\(\s*(?:https?://[^)\s]+)?/wiki[^)]*\)', re.IGNORECASE)
    else:
        pattern = re.compile(r'\(\s*(?:https?://|//)[^)]*\)', re.IGNORECASE)

    return pattern.sub('', text)

def extract_markdown_lists(md_text: str) -> List[List[str]]:
    """
    Extracts list blocks from Markdown text. 
    Returns: List[List[str]], where each sublist is a "block of consecutively appearing list items,"
    each string being a list item (preserving the starting markers '*', '-', '+', '1.', etc.),
    and multiple lines are merged into a single line (internal whitespace is collapsed into a single space).
    """
    lines = md_text.splitlines()
    in_fence = False

    blocks: List[List[str]] = []
    current_block: List[str] = []
    current_item_parts: List[str] = []  
    current_item_prefix: str = ""  

    def flush_item():
        nonlocal current_item_parts, current_block
        if not current_item_parts:
            return
        item = " ".join(part.strip() for part in current_item_parts if part is not None)
        item = re.sub(r'\s+', ' ', item).strip()
        if item:
            current_block.append(current_item_prefix + item)
        current_item_parts = []

    def flush_block():
        nonlocal current_block, blocks
        flush_item()
        if current_block:
            blocks.append(current_block)
        current_block = []

    for raw in lines:
        line = raw.rstrip("\n")

        if FENCE_RE.match(line):
            in_fence = not in_fence
            if in_fence:
                flush_block()
            continue
        if in_fence:
            continue

        start = _is_list_start(line)
        if start:
            indent, marker, text = start
        
            if current_item_parts:
                flush_item()
            if not current_block:
                current_block = []
            current_item_prefix = f"{marker} "
            current_item_parts = [text]
            continue

        if _should_join_prev_item(line):
            if current_item_parts:
                current_item_parts.append(line)
                continue
            else:
                flush_block()
        else:
            flush_block()

    flush_block()


    return blocks

def remove_parentheses_by_prefixes(s: str) -> str:
    """
    Delete fragments of the form (...), including the parentheses themselves, when the first non-whitespace character inside the parentheses starts with one of the specified prefixes. Nested parentheses are supported.
    Target prefixes: /wiki, //en, https, /w/, #, /static
    """
    prefixes = ("/wiki", "//en", "https", "/w/", "#", "/static")

    out = []
    i, n = 0, len(s)

    while i < n:
        ch = s[i]
        if ch != '(':
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n and s[j].isspace():
            j += 1

        if j >= n or not any(s.startswith(pfx, j) for pfx in prefixes):
            out.append('(')
            i += 1
            continue

        depth = 1
        k = i + 1
        while k < n and depth > 0:
            if s[k] == '(':
                depth += 1
            elif s[k] == ')':
                depth -= 1
            k += 1

        if depth > 0:
            i += 1
        else:
            i = k

    return ''.join(out)

def extract_list_from_markdown(markdown_path: str):
    with open(markdown_path, "r", encoding="utf-8") as file:
        md = file.read()
    blocks = extract_markdown_lists(md)

    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            blocks[i][j] = remove_parentheses_by_prefixes(blocks[i][j])
            blocks[i][j] = re.sub(r"\[|\]|\!", "", blocks[i][j])
            blocks[i][j] = re.sub(r"\*\*\^\*\*", "", blocks[i][j])
    return blocks




