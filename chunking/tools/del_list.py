from bs4 import BeautifulSoup
from bs4.element import Tag as _Bs4Tag
import re

def remove_all_lists(html: str, parser: str = "lxml") -> str:
    """
    Remove: ul/ol/li/menu/dir/dl/dt/dd elements, nodes with a list semantic role, and nodes with `style=display:list-item`. 
    Added protection against malformed HTML by setting `attrs=None`.
    """
    soup = BeautifulSoup(html, parser)

    list_tags = {"ul", "ol", "li", "menu", "dir", "dl", "dt", "dd"}
    for name in list_tags:
        for tag in list(soup.find_all(name)):
            if isinstance(tag, _Bs4Tag):
                tag.decompose()

    aria_list_roles = {"list", "listbox", "menu", "menubar", "tablist", "listitem", "group"}

    for tag in list(soup.find_all(True)):
        if not isinstance(tag, _Bs4Tag):
            continue

        attrs = getattr(tag, "attrs", None) or {}
        if not isinstance(attrs, dict):
            attrs = {}

        role_val = attrs.get("role", None)
        role_tokens = set()
        if role_val is not None:
            if isinstance(role_val, str):
                role_tokens = {tok.strip().lower() for tok in role_val.split() if tok.strip()}
            elif isinstance(role_val, (list, tuple, set)):
                role_tokens = {str(x).strip().lower() for x in role_val if x is not None and str(x).strip()}
            else:
                role_tokens = {str(role_val).strip().lower()} if str(role_val).strip() else set()

        if role_tokens & aria_list_roles:
            tag.decompose()
            continue  

        style_val = attrs.get("style", None)
        if isinstance(style_val, str):
            if re.search(r"display\s*:\s*list-item\b", style_val, flags=re.IGNORECASE):
                tag.decompose()
                continue

    return soup.decode(formatter=None)

