"""
Use ASQA's training set to construct SFT data.
"""
import json
from markdown_chunk import process_chunks
from tools.extract_jsonld import read_jsonld
from tqdm import tqdm
from tools.embed import topk_similarity
import random

SFT_SYSTEM_PROMPT = f"""
You are a Retrieval expert. Your sole responsibility is to receive a Query and Chunks from multiple documents, and then rigorously select the indexes(Integer) for the Chunks that are most relevant to the Query.
---
Input Instructions: 
You will receive a Query string and a dictionary of Chunks from multiple documents, where the Chunks are in the following form:
{{
    "Document_i": [[chunk_0], ... [chunk_n]]
}}
---
Selection and Sorting Rules: 
1. A Query typically corresponds to multiple documents, so you can select multiple Chunks from each document and output their indices (integers);
2. You need to carefully consider which Chunks in each document might contain the answer to the current Query. The Chunks you select will be input along with the Query to the downstream reader to generate the answer;
---
Output Format: 
Regardless, you must adhere to the following output rules:
1. Your output must be a parsable JSON object;
2. Do not output any ``` tags;
3. Do not output any introductory words;
4. Your output must be the indexes(Integer) of the selected Chunks in each document;
5. The Chunks you select must be sorted in descending order of similarity to the Query.
---
Output Example: 
{{
    "Document_0": [13, 2, 45, 7], 
    "Document_1": [5, 17, 0, 3, 8, 6, 33], 
}}
"""

def construct_sft(input_path, output_path):
    
    with open(input_path, "r", encoding="utf-8") as tf:
        train_data = json.load(tf)
    
    sft_data = []
    for td in tqdm(train_data, desc="processing"):
        print(len(td["wikipages"]))
        temp = {}
        temp["sample_id"] = td["sample_id"]
        temp["system"] = SFT_SYSTEM_PROMPT
        temp["instruction"] = f"Please select Chunks from each document that are relevant to the current Query.\nThe Query is: {td["ambiguous_question"]}\nThe document Chunks is as follows: \n"

        temp["input"] = {}
        temp["output"] = {}
        temp["score"] = {}
        for idx, wps in enumerate(td["wikipages"]):
            if wps["url"] and wps["htmlpage"]:
                chunks = process_chunks(wps["htmlpage"])
                temp["input"][f"Document_{idx}"] = json.dumps(chunks, ensure_ascii=False)
                
                string_chunks = []
                for chunk in chunks:
                    row_c = ""
                    for c in chunk:
                        row_c += c
                        row_c += "\n"
                    string_chunks.append(row_c)
                top_k = topk_similarity(td["ambiguous_question"], string_chunks)

                temp["output"][f"Document_{idx}"] = json.dumps([k[0] for k in top_k], ensure_ascii=False)
                temp["score"][f"Document_{idx}"] = json.dumps([k[1] for k in top_k], ensure_ascii=False)
                
        sft_data.append(temp)
    
    with open(output_path, "w", encoding="utf-8") as tof:
        json.dump(sft_data, tof, indent=4, ensure_ascii=False)






