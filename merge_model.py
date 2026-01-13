import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen_Qwen3-4B-Instruct-2507"
lora_ckpt_path = "checkpoint"  
merged_model_path = "RES-RAG-MODEL" 

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cuda" 
)

# Loading LoRA adapter
model = PeftModel.from_pretrained(model, lora_ckpt_path)

# Merge the LoRA weights back into the base model and uninstall the PEFT wrapper.
model = model.merge_and_unload()

# Save the merged model.
model.save_pretrained(merged_model_path)

# Synchronously save the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
tokenizer.save_pretrained(merged_model_path)


