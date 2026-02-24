from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


#model_name = "xlm-roberta-base"  # or use a causal model
model_name = "gpt2"

# Lazy loading: cache model and tokenizer to avoid re-downloading on every import
_model = None
_tokenizer = None

def _load_model_and_tokenizer():
    """Load model and tokenizer only once, with caching."""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        print("[perplex] Loading model and tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        print("[perplex] Model and tokenizer loaded.")
    
    return _model, _tokenizer

def compute_perplexity(text):
    m, t = _load_model_and_tokenizer()
    print(f"Computing perplexity... {model_name}")
    inputs = t.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = m(inputs, labels=inputs)
    return torch.exp(outputs.loss).item()