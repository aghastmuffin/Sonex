from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


#model_name = "xlm-roberta-base"  # or use a causal model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def compute_perplexity(text):
    print(f"Computing perplexity... {model_name}")
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    return torch.exp(outputs.loss).item()