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

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder=offload_folder,
    torch_dtype=torch_dtype,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    # Convert Whisper codes to NLLB if needed
    if target_language in WHISPER_TO_NLLB:
        target_language = WHISPER_TO_NLLB[target_language]
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language), 
        max_new_tokens=256,
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Example usage
if __name__ == "__main__":
    article = "UN Chief says there is no military solution in Syria"
    
    # Using NLLB code directly
    result1 = translate(article, target_language="fra_Latn")
    print(f"To French (direct): {result1}")
    
    # Using Whisper code (what you'd get from letter_toolkit.py)
    result2 = translate(article, target_language="fr")
    print(f"To French (Whisper code): {result2}")