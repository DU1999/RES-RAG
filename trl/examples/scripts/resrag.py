"""
accelerate launch examples/scripts/grpo_retriever.py \
  --model_name_or_path \
  --output_dir  \
  --learning_rate  \
  --dtype  \
  --max_prompt_length  \
  --max_completion_length  \
  --temperature  \
  --top_p  
"""
import os
import torch
from datasets import load_dataset
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from typing import List, Union, Dict, Any, Tuple,  Iterable, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from parse_json import parse_json_list
import json
import unicodedata
import re
from transformers import AutoTokenizer
import math
from collections import defaultdict
from args import TrainingHyperparameters
from prompts import PROMPT_GENERATOR, PROMPT_RETRIEVAL
resragArgs = TrainingHyperparameters()

generator_client = OpenAI(api_key=resragArgs.generator_api, base_url=resragArgs.generator_url)


def _normalize_text(s: str) -> str:
    '''
    NFKC normalization: full-width/half-width characters, compatibility forms
    Removes common enclosing symbols such as quotation marks and parentheses, but retains alphanumeric characters and spaces.
    Compresses multiple spaces.
    Removes articles.
    '''
    _ARTICLES = {"a", "an", "the"}
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"[“”\"'《》〈〉«»（）()【】\[\]{}]", " ", s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    toks = [t for t in toks if t not in _ARTICLES]
    return " ".join(toks)


def _maybe_parse_json_list(x: str):
    x = x.strip()
    if (x.startswith("[") and x.endswith("]")) or (x.startswith("(") and x.endswith(")")):
        try:
            v = json.loads(x.replace("(", "[").replace(")", "]"))
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return None


def _as_items(ans: Union[str, Iterable[str]]) -> List[str]:

    _SEPS = ["\n", ";", "；", "，", "、"]
    if isinstance(ans, (list, tuple, set)):
        raw_items = list(ans)
    elif isinstance(ans, str):
        parsed = _maybe_parse_json_list(ans)
        if parsed is not None:
            raw_items = parsed
        else:
            # It only splits by strong delimiters (such as line breaks, semicolons, etc.), and by default does not split by commas to avoid unintended consequences.
            tmp = ans
            for sep in _SEPS:
                if sep != "，": 
                    tmp = tmp.replace(sep, "\n")
            raw_items = [t for t in (t.strip() for t in tmp.split("\n")) if t]
    else:
        raw_items = [str(ans)]

    items = [_normalize_text(str(t)) for t in raw_items if str(t).strip()]
    return sorted(set([t for t in items if t]))


def f1_score(pred, gold) -> float:
    """
    Item-level (set) F1: Compare each answer as an atomic string.
    """
    p = set(_as_items(pred))
    g = set(_as_items(gold))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    precision = inter / len(p)
    recall = inter / len(g)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def compute_reward_scalar_contrastive(
    pred_answers_batch: List[Any],
    gold_answers_batch: List[Any],

    pred_selections_batch: List[Any],

    labels_batch: List[Any],

    sample_id: Optional[List[Any]],

    device,
    chunk_budget: List[Dict[str, Tuple[float, float]]],
    indicator: Optional[List[int]] = None, 

    alpha_f1: float = resragArgs.alpha_f1,
    alpha_ret: float = resragArgs.alpha_ret,

    bad_answer_penalty: float = resragArgs.bad_answer_penalty,
    sys_fail_penalty: float = resragArgs.sys_fail_penalty,

    scale: float = resragArgs.scale,
    clip_min: float = resragArgs.clip_min,
    clip_max: float = resragArgs.clip_max,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Indicator semantics:
    -1: External/system failure, not attributable to the strategy, keep=False (upper layer returns None and skips)
    0: Strategy output format error/empty output/unparsable, considered a learnable error, keep=True (fixed negative reward)
    1: Normal sample
    """
    def _safe_json_loads(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (dict, list)):
            return x
        if not isinstance(x, str):
            x = str(x)
        s = x.strip()
        if not s:
            return None
        if s.startswith("(") and s.endswith(")"):
            s = "[" + s[1:-1] + "]"
        try:
            return json.loads(s)
        except Exception:
            return None

    def _normalize_selection_obj(obj: Any) -> Optional[Dict[str, List[int]]]:
        parsed = _safe_json_loads(obj)
        if not isinstance(parsed, dict):
            return None
        out: Dict[str, List[int]] = {}
        for k, v in parsed.items():
            dk = str(k)
            if not dk:
                continue
            if isinstance(v, (list, tuple, set)):
                vals = list(v)
            else:
                vals = [v]
            idxs: List[int] = []
            for t in vals:
                try:
                    idxs.append(int(t))
                except Exception:
                    continue
            out[dk] = idxs
        return out

    def retrieval_f1_macro(pred_sel: Dict[str, List[int]], gold_sel: Dict[str, List[int]]) -> float:
        if not isinstance(gold_sel, dict) or len(gold_sel) == 0:
            return 0.0
        scores: List[float] = []
        for doc, gold_list in gold_sel.items():
            g = set(int(x) for x in (gold_list or []))
            p = set(int(x) for x in (pred_sel.get(doc, []) or []))
            if not p and not g:
                scores.append(1.0)
                continue
            if not p or not g:
                scores.append(0.0)
                continue
            inter = len(p & g)
            prec = inter / len(p)
            rec = inter / len(g)
            scores.append(0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec)))
        return float(sum(scores) / max(1, len(scores)))

    def is_bad_answer(p: Any) -> bool:
        if p is None:
            return True
        if isinstance(p, str):
            return len(p.strip()) == 0
        if isinstance(p, (list, tuple, set)):
            if len(p) == 0:
                return True
            return all(len(str(x).strip()) == 0 for x in p)
        if isinstance(p, dict):
            return len(p) == 0
        return len(str(p).strip()) == 0

    def budget_penalty(chunk: Dict[str, Tuple[float, float]]) -> float:
        total_penalty = 0.0
        num = 0
        for _, (pre_chunks, ref_chunks) in chunk.items():
            sqrt_ref = 0.0 if ref_chunks <= 0 else math.sqrt(ref_chunks)
            if pre_chunks <= sqrt_ref:
                penalty = 0.0
            elif pre_chunks <= ref_chunks:
                penalty = 0.5
            else:
                penalty = 1.0
            total_penalty += penalty
            num += 1
        return total_penalty / max(1, num) * 0.2

    bs = len(pred_answers_batch)
    if not (len(gold_answers_batch) == len(pred_selections_batch) == len(labels_batch) == len(chunk_budget) == bs):
        raise ValueError("Batch size mismatch.")

    if indicator is None:
        indicator = [1] * bs
    if len(indicator) != bs:
        raise ValueError(f"indicator length ({len(indicator)}) must match batch size ({bs})")

    group2idx = defaultdict(list)
    if sample_id is None:
        for i in range(bs):
            group2idx[i].append(i)
    else:
        if len(sample_id) != bs:
            raise ValueError(f"sample_id length ({len(sample_id)}) must match batch size ({bs})")
        for i, sid in enumerate(sample_id):
            group2idx[str(sid)].append(i)

    rewards: List[float] = [0.0] * bs
    keep: List[bool] = [True] * bs

    for _, idxs in group2idx.items():
        for i in idxs:
            if indicator[i] == -1:
                keep[i] = False
                rewards[i] = sys_fail_penalty 
            elif indicator[i] == 0:
                keep[i] = True
                rewards[i] = sys_fail_penalty

        gold_sel = _normalize_selection_obj(labels_batch[idxs[0]])

        if gold_sel is None:

            for i in idxs:
                if indicator[i] == 1:
                    keep[i] = False
                    rewards[i] = 0.0
            continue

        for i in idxs:
            if indicator[i] != 1:
                continue

            pen = budget_penalty(chunk_budget[i])

            if is_bad_answer(pred_answers_batch[i]):
                ans_term = bad_answer_penalty
            else:
                ans_f1_val = float(f1_score(pred_answers_batch[i], gold_answers_batch[i]))
                ans_term = alpha_f1 * ans_f1_val

            pred_sel = _normalize_selection_obj(pred_selections_batch[i])
            if pred_sel is None:
                ret_term = 0.0
            else:
                ret_term = alpha_ret * float(retrieval_f1_macro(pred_sel, gold_sel))

            r = (ans_term + ret_term) - pen
            r = max(clip_min, min(clip_max, r))
            rewards[i] = scale * r
            keep[i] = True

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    keep_mask_t = torch.tensor(keep, dtype=torch.bool, device=device)
    return rewards_t, keep_mask_t


def call_llama(
    question_batch: List, 
    batch_chunks: List[Union[str, List[Dict[str, str]]]],
    system_prompt: str, 
    *,
    model: str = resragArgs.generator_model,
    max_tokens: int = resragArgs.generator_tokens,
    temperature: float = resragArgs.generator_temperature,
    top_p: float = resragArgs.generator_topp,
    max_workers: int = resragArgs.generator_workers,
    max_retries: int = resragArgs.generator_retries,
    retry_base_sleep: float = resragArgs.generator_sleep
    ) -> List[Dict[str, Any]]:

    def _build_messages(question, item: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        
        info = f"Please output your answer directly in the form of a parsable JSON array based on the following passages: \nThe current Query is: {question}.\nThe current Chunks are: \n{item}"
        return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": info},
            ]

    def _call_once(messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], None]:
        resp = generator_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature, 
            top_p=top_p
        )
        content = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None)
        return {
            "ok": True,
            "content": content,
            "usage": usage.model_dump() if hasattr(usage, "dict") else (usage or {})
        }, None

    def _with_retries(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                data, _ = _call_once(messages)
                return data
            except Exception as e:
                last_err = e
                # Simple backoff: base * 2^(attempt-1) + jitter
                sleep_s = retry_base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                time.sleep(sleep_s)
        return {"ok": False, "content": None, "usage": None, "error": str(last_err)}

    # Concurrent submission
    results: List[Dict[str, Any]] = [None] * len(batch_chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {}
        for idx, item in enumerate(batch_chunks):
            messages = _build_messages(question_batch[idx], item)
            fut = ex.submit(_with_retries, messages)
            future_map[fut] = idx

        for fut in as_completed(future_map):
            idx = future_map[fut]
            data = fut.result()
            data["index"] = idx
            results[idx] = data

    res = []
    for r in results:
        if r["ok"]:
            res.append(r['content'])
        else:
            res.append("None")
    return res


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def _chat_len(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> int:
    """
    Calculate the token length of chat messages (compatible with `apply_chat_template`).
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
) -> List[Dict[str, Any]]:
    """
    A) If excessively long: Delete entire documents from right to left (but keep at least one document).
    B) If only one document remains and it's still too long: Delete chunks from right to left (but keep at least one chunk).
    C) If one document + one chunk is still too long: Truncate the text of that chunk character by character from the right until it fits.
    D) After fitting: Add back the deleted documents chunk by chunk (as prefixes) (add as many as possible; stop once a document can only be partially added back to maintain continuity).
    E) After truncation, write system.content = [num_docs, min_chunks] + PROMPT_RETRIEVAL
    """
    # locate system / user
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
        return messages

    user_content = messages[user_idx].get("content", "")
    extracted = _extract_json_from_user(user_content, marker=marker)
    if extracted is None:
        return messages

    prefix_with_marker, json_str, suffix_after_json = extracted

    try:
        doc_dict_raw = json.loads(json_str)
    except Exception:
        return messages
    if not isinstance(doc_dict_raw, dict):
        return messages

    items = _sorted_doc_items(doc_dict_raw)

    if not items:
        items = [("Document_0", [[""]])]

    def build_messages_for(doc_items: List[Tuple[str, List[Any]]]) -> List[Dict[str, Any]]:
        # Ensure >= 1
        if not doc_items:
            doc_items = [("Document_0", [[""]])]
        # Ensure that each document has at least one chunk.
        fixed_items = []
        for k, v in doc_items:
            if not isinstance(v, list) or len(v) == 0:
                fixed_items.append((k, [[""]]))
            else:
                fixed_items.append((k, v))

        new_doc_dict = {k: v for k, v in fixed_items}
        num_docs = len(new_doc_dict)
        min_chunks = min(len(v) for v in new_doc_dict.values())

        new_json = json.dumps(new_doc_dict, ensure_ascii=False)
        new_user = prefix_with_marker + "<JSON>" + new_json + "\n</JSON>" + suffix_after_json

        new_messages = [dict(m) for m in messages]
        if sys_idx is not None:
            new_messages[sys_idx]["content"] = f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL
        else:
            new_messages.insert(0, {"role": "system", "content": f"【{num_docs}, {min_chunks}】" + PROMPT_RETRIEVAL})
        new_messages[user_idx]["content"] = new_user
        return new_messages

    def fits(doc_items: List[Tuple[str, List[Any]]]) -> bool:
        return _chat_len(build_messages_for(doc_items), tokenizer) <= max_prompt_tokens


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
            orig = chunk0

            def set_text(new_s: str):
                doc_chunks[0] = new_s

            setter = set_text
            orig_text = orig
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

        # Binary truncation of character length
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

    if not fits(kept):
        return build_messages_for(kept)

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


def build_prompt(
    question: str,
    html: str,
    max_prompt_length: int,
    tokenizer: AutoTokenizer,
):
    user_msg = (
        "Please select Chunks from each document that are relevant to the current Query. \n"
        f"The Query is: {question}\n"
        "The document Chunks is as follows: \n\n"
        "<JSON>"
        f"{html}"
        "\n</JSON>"
    )

    messages = [
        {
            "role": "system",
            "content": "【1, 1】" + PROMPT_RETRIEVAL, 
        },
        {"role": "user", "content": user_msg},
    ]

    return retrieval_truncation(
        messages=messages,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_length,
    )



def get_content_from_completion(completion):
    """
    GRPOTrainer places the generation results for each sample in `completions`,
    where `completion[0]['content']` is the text generated by the model for this sample.
    """
    try:
        return completion[0]["content"]
    except Exception:

        if isinstance(completion, str):
            return completion
        return str(completion)


def parse_one(x):
    msg = x[0]
    raw = msg["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print("JSON parsing failed: ", e, "raw:", repr(raw))
        parsed = {}
    return [{**msg, "content": parsed}]


def res_rag_reward(
    completions, 
    sample_id, 
    question: list[str], 
    short_answers: list, 
    input_data, 
    jsonld: list[dict], 
    labels, 
    **kwargs
):
    """
    GRPO will pass the batch columns through parameters with the same names.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    completions = [parse_one(x) for x in completions]

    indicator = []
    policy_outputs = []
    for i, c in enumerate(completions):
        try:
            sample = get_content_from_completion(c)
        except Exception:
            indicator.append(-1)
            policy_outputs.append(None)
            continue

        if sample is None or (isinstance(sample, str) and len(sample.strip()) == 0):
            indicator.append(0)
            policy_outputs.append(sample)
        else:
            indicator.append(1)
            policy_outputs.append(sample)

    chunk_budget = []

    batch_chunks = []
    for i, sample in enumerate(policy_outputs):    

        chunks = {}

        doc_chunk_index = {}
        for doc_name, doc_index in sample.items():
            _data = json.loads(input_data[i])
            
            if doc_name in _data.keys():
                doc_chunk_index[doc_name] = (len(doc_index), len(_data[doc_name]))
                chunks[doc_name] = []
                no_lap_doc_index = list(dict.fromkeys(doc_index))
                for idx in no_lap_doc_index:
                    chunks[doc_name].append(_data[doc_name][idx])

        chunk_budget.append(doc_chunk_index)
        
        # Constructing batch input
        batch_chunks.append({"JSON": jsonld[i], "Chunks": chunks})
    
    pred_answers = call_llama(question, batch_chunks, PROMPT_GENERATOR)
    parse_pred_answers = []
    for pa in pred_answers:
        pa = parse_json_list(pa)
        parse_pred_answers.append(pa)
        
    # Calculate the reward
    rewards_t, keep_mask_t = compute_reward_scalar_contrastive(
        pred_answers_batch=parse_pred_answers,  
        gold_answers_batch=short_answers,
        pred_selections_batch=completions, 
        labels_batch=labels, 
        sample_id=sample_id,
        device=device,
        chunk_budget=chunk_budget,
        indicator=indicator,
        alpha_f1=resragArgs.alpha_f1,
        alpha_ret=resragArgs.alpha_ret,
    )
    rewards_cpu = rewards_t.detach().cpu().tolist()
    keep_cpu = keep_mask_t.detach().cpu().tolist()

    return [float(r) if k else None for r, k in zip(rewards_cpu, keep_cpu)]


if __name__ == "__main__":
    # Parsing parameters
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # ----------------- Model loading parameters -----------------
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    dataset = load_dataset("json", data_files="asqa.jsonl")["train"]
    
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    def to_conversation(example):
        q = example["question"]
        h = example["input_data"]

        prompt = build_prompt(
            question=q,
            html=h,
            max_prompt_length=training_args.max_prompt_length,
            tokenizer=tokenizer,
        )
        return {"prompt": prompt}

    train_dataset = train_dataset.map(to_conversation, remove_columns=[], num_proc=64)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(to_conversation, remove_columns=[], num_proc=64)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[res_rag_reward],  
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train(training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    

