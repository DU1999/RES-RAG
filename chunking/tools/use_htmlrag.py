"""
Call the HTML parsing rules for cleaning (excluding embedded cleaning and generation cleaning).
"""

from htmlrag import clean_html

def call_htmlrag(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        data = file.read()

    return clean_html(data)


