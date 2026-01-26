import json
from transformers import AutoTokenizer
from openai import OpenAI
from tqdm import tqdm
import re
import copy
from typing import List, Union, Dict, Any, Tuple,  Iterable, Optional

vllm_client = OpenAI(
    api_key="",
    base_url="",
)

llama_client = OpenAI(api_key="", base_url="")

PROMPT_RETRIEVAL = f"""
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

PROMPT_GENERATOR = f"""
You are a Reader expert. Your only task is to answer the given questions based only on the provided context paragraphs; output a list of short answers.
For each Query, you will receive multiple documents associated with it. Each document consists of multiple Chunks, all in Markdown format.
---
# Answering Guidelines
1. Answer based on the context only; do not fabricate or introduce outside knowledge. If there is insufficient evidence, output [].
2. Short answers, using concise entity names/numbers/dates/phrases; no explanations, prefixes or suffixes (e.g., "The answer is").
3. When multiple answers are listed side by side (such as aliases, multiple entities, and multiple years), they are listed from high to low in order of confidence, and duplicates and noise are removed.
4. Normalization: Remove unnecessary spaces and punctuation; retain numbers/units of measure as is; format dates according to the original text; preserve capitalization of proper nouns.
---
# Output format
Your output must be a parseable JSON array. Do not include any ``` or any other descriptive terms other than JSON objects. Your output should look like this:
["ans1", ...]
"""


def call_llama(question, item, prompt):
    info = f"Please output your answer directly in the form of a parsable JSON array based on the following passages: \nThe current Query is: {question}.\nThe current Chunks is: \n{item}"
    messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": info},
            ]
    completion = llama_client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=messages,
    temperature=0.2,
    top_p=0.1,
    max_tokens=512
    )
    return completion.choices[0].message.content
    
def build_document_selection_schema(num_doc: int, num_chunks: int):
    """
    Construct a JSON Schema to constrain the model output to the following format:
    {
        "Document_0": [i_0, i_1, ...],
        "Document_1": [...],
        ...
    }
    """

    properties = {}
    required = []

    for i in range(num_doc):
        key = f"Document_{i}"
        required.append(key)

        properties[key] = {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": num_chunks - 1
            },
            "minItems": 1,
            "maxItems": num_chunks,
        }

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }

    return schema

def build_vllm_guided_decoding_regex(num_docs: int, num_chunks: int) -> str:
    """
    Constructing regular expressions for vLLM-guided decoding.
    """
    if num_docs < 1:
        raise ValueError("num_docs must be >= 1")
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")

    ids = list(range(num_chunks))
    ids.sort(key=lambda x: (-len(str(x)), x)) 
    choices = "|".join(str(i) for i in ids)
    num_pattern = rf"(?:{choices})"

    array_pattern = rf"\[{num_pattern}(?:,{num_pattern}){{0,{num_chunks - 1}}}\]"

    doc_entries = []
    for d in range(num_docs):
        entry = rf'"Document_{d}":{array_pattern}'
        doc_entries.append(entry)

    inner = ",".join(doc_entries)

    pattern = r"^\{" + inner + r"\}$"
    return pattern
    

tokenizer = AutoTokenizer.from_pretrained("RES-RAG-GRPO-MODEL")


def _chat_len(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> int:
    """
    Calculate the token length of chat messages.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if isinstance(ids, dict):
            return len(ids["input_ids"])
        return len(ids)
    else:
        text = ""
        for m in messages:
            text += f"{m.get('role','')}: {m.get('content','')}\n"
        return len(tokenizer(text).input_ids)


def _doc_index(k: str) -> int:
    try:
        return int(k.split("_")[-1])
    except Exception:
        return 10**9


def _sorted_doc_items(doc_dict: Dict[str, Any]) -> List[Tuple[str, List[Any]]]:
    """
    Sort by Document_i, filtering out non-list or empty lists.
    """
    items: List[Tuple[str, List[Any]]] = []
    for k, v in doc_dict.items():
        if isinstance(v, list) and len(v) > 0:
            items.append((k, v))
    items.sort(key=lambda kv: _doc_index(kv[0]))
    return items


def _extract_json_from_user(
    user_content: str,
    marker: str,
    start_tag: str = "<JSON>",
    end_tag: str = "</JSON>",
) -> Optional[Tuple[str, str, str]]:
    if marker not in user_content:
        return None

    prefix, rest = user_content.split(marker, 1)
    prefix_with_marker = prefix + marker

    s = rest.find(start_tag)
    e = rest.rfind(end_tag)
    if s == -1 or e == -1 or e < s:
        return None

    json_str = rest[s + len(start_tag) : e].strip()
    suffix = rest[e + len(end_tag) :]
    return prefix_with_marker, json_str, suffix


def _try_get_text_ref(chunk_obj: Any):
    """
    Attempt to locate a string reference that can be truncated:
    - if chunk is a string
    - if chunk is a list of strings
    - if chunk is a dictionary containing fields like text/content/chunk/value that are strings
    Returns a "setter" function `set_text(new_str)` and the original string `orig_str`; returns None if not found.
    """
    if isinstance(chunk_obj, str):
        return ("direct_str", chunk_obj)

    if isinstance(chunk_obj, list) and len(chunk_obj) > 0 and isinstance(chunk_obj[0], str):
        orig = chunk_obj[0]

        def setter(new_s: str):
            chunk_obj[0] = new_s

        return (setter, orig)

    if isinstance(chunk_obj, dict):
        for key in ("text", "content", "chunk", "value"):
            if key in chunk_obj and isinstance(chunk_obj[key], str):
                orig = chunk_obj[key]

                def setter(new_s: str, _k=key):
                    chunk_obj[_k] = new_s
                return (setter, orig)
    return None


def retrieval_truncation(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_prompt_tokens: int,
    marker: str = "The document Chunks is as follows: \n\n",
) -> Tuple[int, int, List[Dict[str, Any]]]:

    sys_idx = None
    user_idx = None
    for i, m in enumerate(messages):
        if sys_idx is None and m.get("role") == "system":
            sys_idx = i
        if user_idx is None and m.get("role") == "user":
            user_idx = i
        if sys_idx is not None and user_idx is not None:
            break

    if user_idx is None:
        num_docs, min_chunks = 1, 1
        new_messages = [dict(m) for m in messages]
        if sys_idx is not None:
            new_messages[sys_idx]["content"] = f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL
        else:
            new_messages.insert(0, {"role": "system", "content": f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL})
        return num_docs, min_chunks, new_messages

    user_content = messages[user_idx].get("content", "")
    extracted = _extract_json_from_user(user_content, marker=marker)
    if extracted is None:
        num_docs, min_chunks = 1, 1
        new_messages = [dict(m) for m in messages]
        if sys_idx is not None:
            new_messages[sys_idx]["content"] = f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL
        else:
            new_messages.insert(0, {"role": "system", "content": f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL})
        return num_docs, min_chunks, new_messages

    prefix_with_marker, json_str, suffix_after_json = extracted

    try:
        doc_dict_raw = json.loads(json_str)
    except Exception:
        num_docs, min_chunks = 1, 1
        new_messages = [dict(m) for m in messages]
        if sys_idx is not None:
            new_messages[sys_idx]["content"] = f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL
        else:
            new_messages.insert(0, {"role": "system", "content": f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL})
        return num_docs, min_chunks, new_messages

    if not isinstance(doc_dict_raw, dict):
        num_docs, min_chunks = 1, 1
        new_messages = [dict(m) for m in messages]
        if sys_idx is not None:
            new_messages[sys_idx]["content"] = f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL
        else:
            new_messages.insert(0, {"role": "system", "content": f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL})
        return num_docs, min_chunks, new_messages

    items = _sorted_doc_items(doc_dict_raw)

    if not items:
        items = [("Document_0", [[""]])]

    def build_messages_for(
        doc_items: List[Tuple[str, List[Any]]]
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        if not doc_items:
            doc_items = [("Document_0", [[""]])]

        fixed_items: List[Tuple[str, List[Any]]] = []
        for k, v in doc_items:
            if (not isinstance(v, list)) or len(v) == 0:
                fixed_items.append((k, [[""]]))
            else:
                fixed_items.append((k, v))

        new_doc_dict = {k: v for k, v in fixed_items}
        num_docs = len(new_doc_dict) 
        min_chunks = min(len(v) for v in new_doc_dict.values())

        new_json = json.dumps(new_doc_dict, ensure_ascii=False)
        new_user = prefix_with_marker + "<JSON>" + new_json + "\n</JSON>" + suffix_after_json

        new_messages = [dict(m) for m in messages]

        local_sys_idx = sys_idx
        local_user_idx = user_idx
        if local_sys_idx is None:
            new_messages.insert(0, {"role": "system", "content": ""})
            local_sys_idx = 0
            local_user_idx = local_user_idx + 1

        new_messages[local_sys_idx]["content"] = f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL
        new_messages[local_user_idx]["content"] = new_user

        return num_docs, min_chunks, new_messages

    def fits(doc_items: List[Tuple[str, List[Any]]]) -> bool:
        _, _, cand_messages = build_messages_for(doc_items)
        return _chat_len(cand_messages, tokenizer) <= max_prompt_tokens

    if fits(items):
        return build_messages_for(items)

    kept = items[:]
    removed_docs: List[Tuple[str, List[Any]]] = []

    while len(kept) > 1 and (not fits(kept)):
        removed_docs.append(kept.pop())

    if not fits(kept):
        doc_key, doc_chunks = kept[0]
        while len(doc_chunks) > 1 and (not fits([(doc_key, doc_chunks)])):
            doc_chunks.pop()
        kept = [(doc_key, doc_chunks)]

    if not fits(kept):
        doc_key, doc_chunks = kept[0]
        doc_chunks = doc_chunks[:1]
        kept = [(doc_key, doc_chunks)]

        chunk0 = doc_chunks[0]

        if isinstance(chunk0, str):
            orig_text = chunk0

            def setter(new_s: str):
                doc_chunks[0] = new_s
        else:
            ref = _try_get_text_ref(chunk0)
            if ref is None:
                doc_chunks[0] = [""]
                return build_messages_for(kept)

            if ref[0] == "direct_str":
                orig_text = ref[1]

                def setter(new_s: str):
                    doc_chunks[0] = new_s
            else:
                setter, orig_text = ref

        lo, hi = 0, len(orig_text)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            setter(orig_text[:mid])
            if fits(kept):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        setter(orig_text[:best])

    if fits(kept):
        for doc_key, doc_chunks in reversed(removed_docs):
            if not doc_chunks:
                continue

            cand_full = kept + [(doc_key, doc_chunks)]
            if fits(cand_full):
                kept = cand_full
                continue

            lo, hi = 1, len(doc_chunks)
            best_k = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = kept + [(doc_key, doc_chunks[:mid])]
                if fits(cand):
                    best_k = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best_k > 0:
                kept = kept + [(doc_key, doc_chunks[:best_k])]
            break

    return build_messages_for(kept)


def build_prompt(question: str, html: str, max_prompt_length: int, tokenizer: AutoTokenizer):
    user_msg = (
        "Please select Chunks from each document that are relevant to the current Query. \n"
        f"The Query is: {question}\n"
        "The document Chunks is as follows: \n\n"
        "<JSON>"
        f"{html}"
        "\n</JSON>"
    )

    messages = [
        {"role": "system", "content": f"【1, 1】" + PROMPT_RETRIEVAL},
        {"role": "user", "content": user_msg},
    ]

    num_docs, min_chunks, new_messages = retrieval_truncation(
        messages=messages,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_length,
    )
    return num_docs, min_chunks, new_messages



def call_vllm(messages, schema):

    chat_response = vllm_client.chat.completions.create(
        model="RES-RAG-GRPO-MODEL", 
        messages=messages, 
        temperature=1.0,
        top_p=0.8,
        extra_body={"guided_regex": schema},
    )
    try:
        return "success", json.loads(chat_response.choices[0].message.content)
    except:
        print(chat_response.choices[0].message.content)
        return "fail", chat_response.choices[0].message.content

def increment_suffix(s):

    match = re.search(r'^(.*?)(?:_(\d+))?$', s)
    if match:
        prefix = match.group(1) 
        number_str = match.group(2)

        if number_str is not None:
            new_number = int(number_str) + 1
            return f"{prefix}_{new_number}"
        else:
            return f"{prefix}_1"
    else:
        return s + "_1"
    
def asqa(path):
    with open(path, "r", encoding="utf-8") as ed:
        eval_data = json.load(ed)

    for data in tqdm(eval_data, desc="process"):
        # try:
        temp = {}
        temp["sample_id"] = data["sample_id"]
        temp["question"] = data["ambiguous_question"]
        temp["short_answers"] = []
        for qpa in data["qa_pairs"]:
            for sas in qpa["short_answers"]:
                if sas not in temp["short_answers"]:
                    temp["short_answers"].append(sas)
        num_docs, num_chunks, prompt = build_prompt(
                question=temp["question"],
                html=json.dumps(data["chunks"]),
                max_prompt_length=8192,
                tokenizer=tokenizer,
            )
        schema = build_vllm_guided_decoding_regex(num_docs, num_chunks)
        status, policy_outputs = call_vllm(prompt, schema)
        if status == "success":
            temp["indices_for_model_select"] = policy_outputs
            policy_outputs = [policy_outputs]
            input_data = [data["chunks"]]
            jsonld = [data["json"]]
            batch_chunks = []
            
            for i, sample in enumerate(policy_outputs):   
                chunks = {}
                for doc_name, doc_index in sample.items():
                    _data = input_data[i]
                    
                    if doc_name in _data.keys():
                        chunks[doc_name] = []
                        no_lap_doc_index = list(dict.fromkeys(doc_index))
                        for idx in no_lap_doc_index:
                            chunks[doc_name].append(_data[doc_name][idx])
                    else:
                        print(f'{data["sample_id"]} no exits {doc_name}: {_data.keys()}')
                
                batch_chunks.append({"JSON": jsonld[i], "Chunks": chunks})
            pred_answers = call_llama(temp["question"], batch_chunks, PROMPT_GENERATOR)
            temp["model_ans"] = pred_answers
            
        print(temp)

