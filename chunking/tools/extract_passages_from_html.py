"""
Clean the HTML that does not contain tables or lists, keeping only the `h` and `p` tags.
"""

from bs4 import BeautifulSoup, Comment, NavigableString
import re
from functools import lru_cache

ALLOWED_INLINE = {"b", "i", "sup"}
ALLOWED_BLOCK = {f"h{i}" for i in range(1, 7)} | {"p"}
LOWER_RUN_RE = re.compile(r"[a-z]{6,}") 


FALLBACK_VOCAB = {
    "a","an","the","of","and","or","to","in","on","for","as","by","is","are","was","were","it","its","that",
    "with","at","from","this","be","been","into","over","after","before","than","but","not","no","one","two",
    "smart","phone","developed","apple","inc","browser","web","safari","multi","touch","display","features",
    "soc","screen","camera","glass","market","sold","price","model","apps","app","store","software","hardware",
    "release","announced","january","june","2007","2008","original","iphone","generation","jobs","steve",
    "europe","united","states","canada","wireless","sales","version","support","system","chip","gpu","battery",
    "million","devices",
}

def _collapse_whitespace_in_p(p_tag):
    """
    Collapse consecutive whitespace characters within all text nodes inside `<p>` tags into a single space, and remove leading and trailing whitespace.
    """
    for node in list(p_tag.descendants):
        if isinstance(node, NavigableString):
            s = str(node)
            s2 = re.sub(r"\s+", " ", s) 
            if s2 != s:
                node.replace_with(s2)
    return p_tag


def _split_word_with_libs(token: str):
    try:
        import wordninja
        return wordninja.split(token)
    except Exception:
        pass
    try:
        from wordfreq import zipf_frequency
        @lru_cache(maxsize=10000)
        def score(w):
            z = zipf_frequency(w, "en")
            return z if z > 0 else -10.0

        n = len(token)
        dp = [float("-inf")] * (n+1)
        cut = [-1] * (n+1)
        dp[0] = 0.0
        for i in range(n):
            if dp[i] == float("-inf"):
                continue
            for j in range(i+1, min(n, i+20)+1):
                w = token[i:j]
                dpj = dp[i] + score(w)
                if dpj > dp[j]:
                    dp[j] = dpj
                    cut[j] = i
        if dp[n] == float("-inf"):
            return [token]
        out = []
        idx = n
        while idx > 0:
            i = cut[idx]
            out.append(token[i:idx])
            idx = i
        return out[::-1]
    except Exception:
        return None

@lru_cache(maxsize=10000)
def _split_word_greedy(token: str, vocab_frozenset=None):
    vocab = vocab_frozenset or frozenset(FALLBACK_VOCAB)
    n = len(token)
    dp = [None]*(n+1) 
    score = [float("-inf")]*(n+1)
    dp[0] = []
    score[0] = 0.0

    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(min(n, i+20), i, -1):
            w = token[i:j]
            s = (2.0 if w in vocab else -1.0) + 0.1*len(w)
            if score[i] + s > score[j]:
                score[j] = score[i] + s
                dp[j] = dp[i] + [w]
    return dp[n] or [token]

def split_glued_english(token: str, custom_vocab: set | None = None) -> list[str]:
    """
    Perform English tokenization on a string consisting of only lowercase letters.
    """
    parts = _split_word_with_libs(token)
    if parts:
        return parts
    vocab = frozenset((custom_vocab or set()) | FALLBACK_VOCAB)
    return _split_word_greedy(token, vocab_frozenset=vocab)


def _clean_inline(p_tag):
    for bad in p_tag.find_all(["script", "style"]):
        bad.decompose()
    for node in list(p_tag.descendants):
        if getattr(node, "name", None):
            if node.name in ALLOWED_INLINE:
                node.attrs = {}
            elif node.name not in ALLOWED_BLOCK:
                node.unwrap()
    return p_tag

def _is_trivial_text(s):
    return s is None or (isinstance(s, str) and s.strip() == "")

def _strip_edit_label(s):
    if not s:
        return s
    return re.sub(r"\s*\[edit\]\s*$", "", s).strip()

def _fix_glued_words_in_text_nodes(tag, custom_vocab=None):
    for node in list(tag.descendants):
        if isinstance(node, NavigableString):
            text = str(node)
            def _repl(m):
                chunk = m.group(0)
                pieces = split_glued_english(chunk, custom_vocab=custom_vocab)
                return " ".join(pieces)
            new_text = LOWER_RUN_RE.sub(_repl, text)
            if new_text != text:
                node.replace_with(new_text)

def html_to_headings_and_paragraphs(html: str, custom_vocab: set | None = None) -> str:
    """
    Only retain h1–h6 and p tags:
    - Within paragraphs, only retain <b>/<i>/<sup> tags; other inline tags are expanded into plain text;
    - Remove scripts, styles, comments, and empty paragraphs; remove "[edit]" following titles;
    - If a title is not followed by any non-empty <p> tag until the next title (or the end of the document), discard that title;
    - Attempt to tokenize and insert spaces into **conjoined lowercase English strings** within paragraphs (e.g., asmartphonedeveloped → a smart phone developed).
    """
    soup = BeautifulSoup(html, "lxml")

    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    root = soup.find("main") or soup.body or soup

    out = BeautifulSoup("", "lxml")
    out_root = out.new_tag("div")
    out.append(out_root)

    pending_h = None 

    def flush_pending_heading_if_any():
        nonlocal pending_h
        if pending_h is not None:
            h_tag = out.new_tag(pending_h[0])
            h_tag.string = pending_h[1]
            out_root.append(h_tag)
            pending_h = None

    heading_names = {f"h{i}" for i in range(1, 7)}

    for el in root.descendants:
        if not getattr(el, "name", None):
            continue
        name = el.name.lower()

        if name in heading_names:
            text = _strip_edit_label(el.get_text(" ", strip=True))
            pending_h = (name, text) if text else None

        elif name == "p":
            p_clone = BeautifulSoup(str(el), "lxml").p
            p_clean = _clean_inline(p_clone)

            visible_text = p_clean.get_text(" ", strip=True)
            if _is_trivial_text(visible_text):
                continue

            _fix_glued_words_in_text_nodes(p_clean, custom_vocab=custom_vocab)

            _collapse_whitespace_in_p(p_clean)

            flush_pending_heading_if_any()

            p_clean.attrs = {}
            out_root.append(BeautifulSoup(str(p_clean), "lxml").p)

        else:
            continue

    res = "\n".join(
        c.decode(formatter=None).lstrip() 
        for c in out_root.children
    )
    res = re.sub("<b>|<i>|<sup>|</b>|</i>|</sup>", " ", res)
    res = re.sub("\n", "", res)
    return res





