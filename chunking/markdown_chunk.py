"""
The process is divided into two parts: lists/tables are processed separately, and everything else is treated as text.
"""

from tools.del_list import remove_all_lists
from tools.del_table import remove_tables_from_html
from tools.html4clean import harpa_clean
from tools.use_markitdown import convert_html_to_markdown
from tools.process_list import extract_list_from_markdown, extract_lists_with_nearest_heading
from tools.process_table import extract_tables_from_markdown, find_table_prev_headings
import re
import json
from pathlib import Path
import shutil
import os
import copy
import random

def remove_edit_tags(md_path: str, *, backup: bool = False, encoding: str = "utf-8") -> None:
    """
    Replace all instances of `edit` in the Markdown file with an empty string. 
    """
    p = Path(md_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {md_path}")

    text = p.read_text(encoding=encoding)

    edit_pattern = re.compile(r'\[\s*edit\s*\](?:\([^)]+\))?', flags=re.IGNORECASE)

    new_text = edit_pattern.sub('', text)

    new_text = re.sub(r'[ \t]+\n', '\n', new_text)

    if backup:
        shutil.copyfile(p, p.with_suffix(p.suffix + ".bak"))

    p.write_text(new_text, encoding=encoding)


def split_markdown_by_headings(path: str):
    """
    Using Markdown headings as separators, build blocks starting from the first heading. 
    Returns: List[List[str]]
    """
    heading_re = re.compile(r'^\s{0,3}#{1,6}\s')
    blocks = []
    current_block = None
    started = False 

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if heading_re.match(line):
                started = True
                if current_block and len(current_block) > 0:
                    blocks.append(current_block)
                current_block = [] 
                continue

            if not started:
                continue

            content = line.strip()
            if content:
                if current_block is None:
                    current_block = []
                current_block.append(content)

    if started and current_block and len(current_block) > 0:
        blocks.append(current_block)

    return blocks


def process_text(html_content: str):
    '''
    Handling non-list/table content
    '''
    
    no_list = remove_all_lists(html_content)
    
    no_table = remove_tables_from_html(no_list)

    html = harpa_clean(no_table)

    virtual = random.randint(1, 100000)
    tmp_html_path = f"tmp{virtual}.html"
    tmp_md_path = f"tmp{virtual}.markdown"
    
    with open(tmp_html_path, "w", encoding="utf-8") as tmp_html_file:
        tmp_html_file.write(html)
    
    success = convert_html_to_markdown(tmp_html_path, tmp_md_path)
    
    remove_edit_tags(tmp_md_path)

    text_chunks = split_markdown_by_headings(tmp_md_path)
    
    os.remove(tmp_html_path)
    os.remove(tmp_md_path)
    return text_chunks


def process_ListTable(markdown_path: str):
    '''
    Processing list/table content
    '''
    ListTable_chunks = []
    md = extract_tables_from_markdown(markdown_path)
    ls = extract_list_from_markdown(markdown_path)
    ListTable_chunks.extend(md)
    ListTable_chunks.extend(ls)
    return ListTable_chunks


def merge_chunks(text_chunks: list[list[str]], ListTable_chunks: list[list[str]]):
    '''
    For non-list/table blocks and list/table blocks, they need to be merged into a single block list.
    '''
    chunks = []
    chunks.extend(text_chunks)
    chunks.extend(ListTable_chunks)
    return chunks


def process_chunks(html_content: str):
    '''
    Main function
    '''
    html_x, html_y = copy.deepcopy(html_content), copy.deepcopy(html_content)
    text_chunk = process_text(html_x)
    
    virtual = random.randint(1, 100000)
    
    tmp_html_path = f"h{virtual}.html"
    tmp_md_path = f"m{virtual}.markdown"
    
    with open(tmp_html_path, "w", encoding="utf-8") as hf:
        hf.write(html_y)
        
    convert_html_to_markdown(tmp_html_path, tmp_md_path)
    
    list_table_chunk = process_ListTable(tmp_md_path)
    
    chunks = merge_chunks(text_chunk, list_table_chunk)
    
    os.remove(tmp_html_path)
    os.remove(tmp_md_path)
    return chunks
    


