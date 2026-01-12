"""
Remove all tables from the HTML.
"""

from bs4 import BeautifulSoup

def remove_tables_from_html(html_doc: str, parser: str = "lxml") -> str:

    soup = BeautifulSoup(html_doc, parser)

    for tbl in soup.find_all("table"):
        tbl.decompose() 

    return soup.decode(formatter=None)




