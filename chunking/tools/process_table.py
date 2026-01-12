"""
Given a Markdown file as input, extract the tables from it and return them as a list.
"""

import re

# 链接和图片的基础正则（不跨行）
LINK_PAREN_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')    
IMG_PAREN_RE  = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')      
LINK_REF_RE   = re.compile(r'\[([^\]]+)\]\[[^\]]*\]')        
IMG_REF_RE    = re.compile(r'!\[([^\]]*)\]\[[^\]]*\]')       
CITE_RE       = re.compile(r'\[\[([^\]]+)\]\]\(#cite[^\)]*\)', re.IGNORECASE) 

URL_HINT_RE = re.compile(
    r'/(?:wiki)\b|//upload\b|#cite|\.jpe?g\b|\.png\b|\.gif\b|\.svg\b|\.webp\b',
    re.IGNORECASE
)

def _clean_links_outside_paren(text: str) -> str:

    t = IMG_PAREN_RE.sub(lambda m: f'[{m.group(1)}]', text)
    t = LINK_PAREN_RE.sub(lambda m: f'[{m.group(1)}]', t)
    t = IMG_REF_RE.sub(lambda m: f'[{m.group(1)}]', t)
    t = LINK_REF_RE.sub(lambda m: f'[{m.group(1)}]', t)
    t = CITE_RE.sub(lambda m: m.group(1), t)  
    t = t.replace('!', '')
    return t

def _clean_links_inside_paren(text: str) -> str:

    t = IMG_PAREN_RE.sub(lambda m: m.group(1), text)
    t = LINK_PAREN_RE.sub(lambda m: m.group(1), t)
    t = IMG_REF_RE.sub(lambda m: m.group(1), t)
    t = LINK_REF_RE.sub(lambda m: m.group(1), t)
    t = CITE_RE.sub(lambda m: m.group(1), t)
    t = t.replace('!', '')
    return t

def _process_parentheses_context_aware(s: str) -> str:
    stack = []
    pairs = []
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                l = stack.pop()
                pairs.append((l, i))

    if not pairs:
        return s

    out = []
    last = 0
    for l, r in sorted(pairs):
        inside = s[l+1:r]
        if URL_HINT_RE.search(inside) or LINK_PAREN_RE.search(inside) or IMG_PAREN_RE.search(inside):
            cleaned_inside = _clean_links_inside_paren(inside)
            out.append(s[last:l])
            out.append('(' + cleaned_inside + ')')
            last = r + 1
        else:
            continue
    if last == 0:
        return s

    out.append(s[last:])
    return ''.join(out)

def _normalize_spaces(t: str) -> str:
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

_SEP_CELL_RE = re.compile(r'^\s*:?-{3,}:?\s*$')
def _is_separator_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    cells = re.split(r'(?<!\\)\|', s)
    if not cells:
        return False
    return all(_SEP_CELL_RE.match(c.strip() or '---') for c in cells)

def _split_row(line: str) -> list[str]:
    s = line.rstrip('\r\n')
    has_leading = s.startswith('|')
    has_trailing = s.endswith('|')
    if has_leading:
        s = s[1:]
    if has_trailing:
        s = s[:-1]
    parts = re.split(r'(?<!\\)\|', s)
    parts = [p.replace(r'\|', '|') for p in parts]
    return parts


def remove_parens_starting_with_wiki(text: str) -> str:
    pattern = re.compile(r'\(\s*/wiki[^()]*\)')
    return pattern.sub('', text)


def _clean_cell(cell: str) -> str:
    t = _process_parentheses_context_aware(cell)
    t = _clean_links_outside_paren(t)
    t = _normalize_spaces(t)
    t = remove_parens_starting_with_wiki(t)
    t = re.sub(r"\[|\]", "", t)
    return t

def extract_tables_from_markdown(md_path: str) -> list[list[str]]:
    tables: list[list[str]] = []
    in_code_block = False

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    n = len(lines)
    i = 0

    def _is_table_header(idx: int) -> bool:
        if idx + 1 >= n or in_code_block:
            return False
        head = lines[idx].rstrip('\r\n')
        sep = lines[idx + 1].rstrip('\r\n')
        if '|' not in head:
            return False
        return _is_separator_line(sep)

    while i < n:
        line = lines[i].rstrip('\r\n')

        if re.match(r'^\s*```', line):
            in_code_block = not in_code_block
            i += 1
            continue

        if not in_code_block and _is_table_header(i):
            table_rows_raw = [lines[i].rstrip('\r\n'), lines[i+1].rstrip('\r\n')]
            i += 2
            while i < n:
                l = lines[i].rstrip('\r\n')
                if re.match(r'^\s*```', l):
                    break
                if '|' not in l:
                    break
                table_rows_raw.append(l)
                i += 1

            table_rows_clean = []
            for raw in table_rows_raw:
                cells = _split_row(raw)
                cleaned = []
                for c in cells:
                    tmp = _clean_cell(c)
                    cleaned.append(tmp)

                row_str = '| ' + ' | '.join(cleaned) + ' |'
                row_str = re.sub(r'\s+\|', ' |', re.sub(r'\|\s+', '| ', row_str)).strip()
                table_rows_clean.append(row_str)
            tables.append(table_rows_clean)
            continue
        i += 1
    return tables



HEADING_ATX_RE = re.compile(r'^\s{0,3}(#{1,6})\s*(.*?)\s*#*\s*$')
SETEXT_EQ_RE  = re.compile(r'^\s*=+\s*$')   
SETEXT_DASH_RE = re.compile(r'^\s*-+\s*$')  

def _clean_heading_text(t: str) -> str:
    t = _process_parentheses_context_aware(t)
    t = _clean_links_outside_paren(t)
    t = remove_parens_starting_with_wiki(t)
    t = _normalize_spaces(t)
    t = re.sub(r'^\[|\]$', '', t) 
    t = re.sub(r'[\[\]]', '', t)  
    return t.strip()

def _is_code_fence(line: str) -> bool:
    return re.match(r'^\s*(```|~~~)', line) is not None

def find_table_prev_headings(md_path: str) -> list[list]:
    """
    Returns a list in the format [[1, "Recent heading above"], [2, "Recent heading above"], ...]
    - Table numbers are 1-based, counted in the order they appear in the file.
    - Headings are matched using ATX (#...) and Setext (===/---) syntax.
    - Headings and tables within code fences are ignored.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\r\n') for ln in f.readlines()]

    n = len(lines)
    in_code = False

    headings: list[tuple[int, str]] = []
    table_starts: list[int] = []

    i = 0
    while i < n:
        line = lines[i]

        if _is_code_fence(line):
            in_code = not in_code
            i += 1
            continue

        if not in_code:
            m = HEADING_ATX_RE.match(line)
            if m:
                heading_text = _clean_heading_text(m.group(2))
                if heading_text:
                    headings.append((i, heading_text))
                i += 1
                continue

            if i > 0 and ('|' not in line): 
                if SETEXT_EQ_RE.match(line) or (SETEXT_DASH_RE.match(line) and not _is_separator_line(line)):
                    raw = lines[i - 1].strip()
                    if raw and not _is_code_fence(raw):
                        heading_text = _clean_heading_text(raw)
                        if heading_text:
                            headings.append((i, heading_text)) 
                    i += 1
                    continue

            if '|' in line and i + 1 < n and _is_separator_line(lines[i + 1]) and not in_code:
                table_starts.append(i)  # 记录表头行索引
                j = i + 2
                while j < n:
                    l = lines[j]
                    if _is_code_fence(l):
                        break
                    if '|' not in l:
                        break
                    j += 1
                i = j
                continue

        i += 1

    results: list[list] = []
    if not table_starts:
        return results

    h_idx = 0
    for t_idx, t_line in enumerate(table_starts, start=1):  # t_idx 为 1-based 序号
        while h_idx < len(headings) and headings[h_idx][0] < t_line:
            h_idx += 1
        if h_idx == 0:
            title = "" 
        else:
            title = headings[h_idx - 1][1]
        results.append([t_idx, title])

    return results
