"""
For an original HTML webpage, this file is used to clean the HTML and construct the input for SFT.
"""

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
import re
import html as _html

def clean_html(html: str, remove_style_tags: bool = True, pretty: bool = False) -> str:
    """
    Cleaning HTML. 

    Parameters:
    html: The original HTML text
    remove_style_tags: Whether to remove <style> and <noscript> tags (default True)
    pretty: Whether to output with indentation for better readability (default False)
    Returns:
    The cleaned HTML string
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    if soup.head is not None:
        title_tag = soup.head.title
        new_head = soup.new_tag("head")
        if title_tag is not None:
            copied_title = soup.new_tag("title")
            copied_title.string = title_tag.get_text()
            new_head.append(copied_title)
        soup.head.replace_with(new_head)

    to_remove = ["script", "img", "input"] 
    if remove_style_tags:
        to_remove += ["style", "noscript"]
    for tag in soup.find_all(to_remove):
        tag.decompose()

    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    for tag in soup.find_all(True):
        tag.attrs = {}

    return soup.prettify() if pretty else str(soup)

def remove_buttons(html: str, pretty: bool = False) -> str:
    """
    Removes all `<button>...</button>` elements (including all their contents). 

    Parameters:
    html: The original HTML text
    pretty: Whether to format the output (indentation, line breaks)
    Returns:
    The HTML string after removing the buttons
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for btn in soup.find_all("button"):
        btn.decompose()

    return soup.prettify() if pretty else str(soup)


_WHITESPACE_LIKE = ("\u00A0", "\u200B", "\u200C", "\u200D", "\uFEFF")

def _is_visible_text(s: str) -> bool:
    """
    Determine whether a piece of text contains visible characters (excluding spaces, line breaks, tabs, non-breaking spaces, and zero-width characters).
    """
    if s is None:
        return False
    for ch in _WHITESPACE_LIKE:
        s = s.replace(ch, " ")
    return bool(s.strip()) 

def _tag_has_visible_text(tag) -> bool:
    """
    Determine whether the tag itself (including its descendants) contains any visible text.
    """
    for node in tag.descendants:
        if isinstance(node, NavigableString) and _is_visible_text(str(node)):
            return True
    return False

def _prune_empty_tags_once(soup: BeautifulSoup) -> int:
    """
    Perform a round of "bottom-up" empty tag removal.
    Return the number of tags removed.
    """
    removed = 0

    for tag in list(soup.find_all(True))[::-1]:

        if not _tag_has_visible_text(tag):
            tag.decompose()
            removed += 1
    return removed

def remove_empty_or_tagonly_blocks(html: str, pretty: bool = True) -> str:
    """
    Remove block-level/inline elements that are "empty" or "contain only other (also empty) tags". 
    Rules:
    - If an element and all its descendants contain no visible text, the entire subtree of that element will be removed. 
    - The cleaning process is performed bottom-up; until no more tags can be removed. 

    - If a parent element only contains child tags, but those child tags contain visible text, the parent element will **not be removed**. 
    - Non-breaking spaces, zero-width characters, etc., are considered "whitespace" and do not count as content.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    while True:
        removed = _prune_empty_tags_once(soup)
        if removed == 0:
            break

    return soup.prettify() if pretty else str(soup)

def _escape_attr(value):
    return _html.escape(str(value), quote=True)

def _format_open_tag(tag: Tag) -> str:
    if not tag.attrs:
        return f"<{tag.name}>"
    parts = [f"<{tag.name}"]

    for k, v in tag.attrs.items():
        if isinstance(v, list):
            v = " ".join(map(str, v))
        parts.append(f'{k}="{_escape_attr(v)}"')
    return " ".join(parts) + ">"

def _format_close_tag(tag: Tag) -> str:
    return f"</{tag.name}>"

def _is_leaf_text_tag(tag: Tag) -> bool:
    for child in tag.contents:
        if isinstance(child, Tag):
            return False
        if isinstance(child, Comment):
            return False
    text = tag.get_text(" ", strip=True)
    return text != ""

def _serialize(node, out_lines: list[str]):
    """
    Depth-first serialization:
    - Leaf node text labels are inline: <tag>text</tag>
    - Others: Start tag, child nodes, and end tag each occupy a separate line.
    - No indentation for any lines.
    """
    if isinstance(node, NavigableString):
        s = str(node)
        s = " ".join(s.split()) 
        if s:
            out_lines.append(s)
        return

    if not isinstance(node, Tag):
        return

    if _is_leaf_text_tag(node):
        open_tag = _format_open_tag(node)
        text = node.get_text(" ", strip=True)
        close_tag = _format_close_tag(node)
        out_lines.append(f"{open_tag}{text}{close_tag}")
        return

    out_lines.append(_format_open_tag(node))

    for child in node.contents:
        if isinstance(child, Comment):
            continue  
        if isinstance(child, NavigableString):
            s = str(child)
            s = " ".join(s.split())
            if s:
                out_lines.append(s)
        else:
            _serialize(child, out_lines)

    out_lines.append(_format_close_tag(node))

def inline_minimal_tags_and_flush_left(html: str) -> str:
    """
    Inline the smallest tags (containing only text) into a single line; the opening and closing tags of other tags each occupy a separate line;
    The overall output should be left-aligned (no leading spaces).
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    root = soup.html if soup.html else soup

    lines: list[str] = []
    if isinstance(root, Tag):
        _serialize(root, lines)
    else:
        for child in soup.contents:
            _serialize(child, lines)

    lines = [ln for ln in lines if ln is not None and ln != ""]

    lines = [ln.lstrip() for ln in lines]

    return "\n".join(lines)


def strip_anchor_tags(html: str, collapse_whitespace: bool = False) -> str:
    """
    Removes the opening and closing tags of all `<a>` elements, preserving only the plain text within them. 

    Parameters:
    html: The original HTML
    collapse_whitespace: Whether to compress consecutive whitespace into a single space (default True)
    Returns:
    The processed HTML string
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a"):
        text = a.get_text(separator=" ", strip=True) 
        a.replace_with(text)

    result = str(soup)

    if collapse_whitespace:
        import re
        result = re.sub(r"[ \t\r\n]+", " ", result).strip()

    return result

def compress_redundant_divs(html: str) -> str:
    """
    Compress "redundant, attribute-less" `<div>` elements:
    - Remove empty `<div>` elements (containing only whitespace/comments);
    - Collapse attribute-less `<div>` elements that only contain another attribute-less `<div>` (unfold the inner element). 
    Other `<div>` elements with attributes will remain unchanged, minimizing impact on styles and functionality. 
    The return value will maintain the original "no indentation/unformatted" style (without using prettify).
    """
    def is_whitespace_node(node):
        return isinstance(node, NavigableString) and (str(node).strip() == "")

    def visible_children(tag):
        return [
            c for c in tag.contents
            if not (isinstance(c, Comment) or is_whitespace_node(c))
        ]

    def is_attrless_div(tag):
        return tag.name == "div" and not tag.attrs

    soup = BeautifulSoup(html, "html.parser")

    changed = True
    while changed:
        changed = False

        for div in list(soup.find_all("div")):
            if not is_attrless_div(div):
                continue
            if len(visible_children(div)) == 0:
                div.decompose()
                changed = True

        for div in list(soup.find_all("div")):
            if not is_attrless_div(div):
                continue
            vis = visible_children(div)
            if len(vis) == 1 and getattr(vis[0], "name", None) == "div":
                inner = vis[0]
                if is_attrless_div(inner):
                    inner.unwrap()  
                    changed = True

    return "".join(str(node) for node in soup.contents)


def harpa_clean(html_doc: str) -> str:
    s1 = clean_html(html_doc, remove_style_tags=True, pretty=True)
    s2 = remove_buttons(s1)
    s3 = remove_empty_or_tagonly_blocks(s2)
    s4 = inline_minimal_tags_and_flush_left(s3)
    s5 = re.sub("<span>", "", s4)
    s6 = re.sub("</span>", "", s5)
    s7 = strip_anchor_tags(s6)
    s8 = compress_redundant_divs(s7)
    s9 = re.sub("\n", "", s8)
    return s9




