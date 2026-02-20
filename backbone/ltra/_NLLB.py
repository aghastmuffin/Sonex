# Replacement library for ollama
# Using a model by META's No Language Left Behind Project
# guion desarrollado por: Levi B, Berkeley, CA
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

# Create offload folder for disk offloading
offload_folder = os.path.join(os.path.dirname(__file__), "..", "..", "model_offload")
os.makedirs(offload_folder, exist_ok=True)

# Lazy loading: cache model and tokenizer to avoid re-downloading on every import
_model = None
_tokenizer = None

def _load_model_and_tokenizer():
    """Load model and tokenizer only once, with caching."""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        print("[_NLLB] Loading model and tokenizer...")
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder=offload_folder,
            torch_dtype=torch_dtype,
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[_NLLB] Model and tokenizer loaded.")
    
    return _model, _tokenizer

def get_model():
    """Get the cached model, loading it on first access."""
    m, _ = _load_model_and_tokenizer()
    return m

def get_tokenizer():
    """Get the cached tokenizer, loading it on first access."""
    _, t = _load_model_and_tokenizer()
    return t

# Mapping from Whisper language codes (ISO 639-1) to NLLB FLORES codes
WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ar": "ara_Arab",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
}

def translate(text: str, target_language: str = "fra_Latn", source_language: str = None) -> str:
    """
    Translate text using NLLB model.
    
    Args:
        text: Text to translate
        target_language: NLLB FLORES code (e.g., "fra_Latn") or Whisper code (e.g., "fr")
        source_language: NLLB FLORES code or Whisper code (auto-detect if None)
    
    Returns:
        Translated text
    """
    m = get_model()
    t = get_tokenizer()
    
    # Convert Whisper codes to NLLB if needed
    if target_language in WHISPER_TO_NLLB:
        target_language = WHISPER_TO_NLLB[target_language]
    
    inputs = t(text, return_tensors="pt").to(m.device)
    translated_tokens = m.generate(
        **inputs, 
        forced_bos_token_id=t.convert_tokens_to_ids(target_language), 
        max_new_tokens=256,
    )
    return t.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Example usage
if __name__ == "__main__":
    article = "UN Chief says there is no military solution in Syria"
    
    # Using NLLB code directly
    result1 = translate(article, target_language="fra_Latn")
    print(f"To French (direct): {result1}")
    
    # Using Whisper code (what you'd get from letter_toolkit.py)
    result2 = translate(article, target_language="fr")
    print(f"To French (Whisper code): {result2}")