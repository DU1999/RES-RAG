"""
Parsing JSON-LD from an HTML string.
"""
from bs4 import BeautifulSoup
import json

def read_jsonld(html_doc: str) -> list:
    soup = BeautifulSoup(html_doc, "html.parser")
    jsonld_data = []

    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string)
            jsonld_data.append(data)
        except Exception as e:
            print("JSON-LD parsing failed:", e)
    return jsonld_data