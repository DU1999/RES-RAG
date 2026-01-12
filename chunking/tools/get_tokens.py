

from transformers import AutoTokenizer

def count_tokens(text, model_name="Qwen_Qwen3-4B-Instruct-2507") ->int:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)
        return token_count
    except Exception as e:
        print(f"Error calculating the number of tokens: {str(e)}")
        return -1




