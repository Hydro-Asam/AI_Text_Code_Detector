import math
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sdadas/polish-gpt2-small"

_tokenizer = None
_model = None


def split_sentences(text: str) -> list[str]:
    """Dzieli tekst na zdania po znakach końca zdania (. ! ?)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def count_words(sentence: str) -> int:
    """Liczy słowa w pojedynczym zdaniu."""
    return len(re.findall(r"\b\w+\b", sentence))


def calculate_burstiness(text: str) -> dict[str, float | int | list[int]]:
    """
    Liczy podstawowe metryki burstiness na bazie długości zdań (w słowach).
    """
    sentences = split_sentences(text)
    lengths = [count_words(s) for s in sentences if count_words(s) > 0]

    if not lengths:
        return {
            "sentence_count": 0,
            "lengths": [],
            "mean": 0.0,
            "std_dev": 0.0,
            "burstiness": 0.0,
        }

    sentence_count = len(lengths)
    mean = sum(lengths) / sentence_count

    variance = sum((x - mean) ** 2 for x in lengths) / sentence_count
    std_dev = math.sqrt(variance)
    burstiness = (std_dev / mean) if mean > 0 else 0.0

    return {
        "sentence_count": sentence_count,
        "lengths": lengths,
        "mean": round(mean, 2),
        "std_dev": round(std_dev, 2),
        "burstiness": round(burstiness, 4),
    }


def _load_model(model_name: str = MODEL_NAME):
    """Ładuje tokenizer i model leniwie przy pierwszym wywołaniu."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Ładuję model {model_name} (pierwsze uruchomienie pobierze wagi)...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.eval()
    return _tokenizer, _model


def calculate_perplexity(text: str, model_name: str = MODEL_NAME) -> float:
    """
    Liczy perplexity tekstu dla polskiego modelu językowego.
    Niższa wartość = tekst bardziej przewidywalny dla modelu.
    """
    cleaned = text.strip()
    if not cleaned:
        return 0.0

    tokenizer, model = _load_model(model_name)
    encodings = tokenizer(cleaned, return_tensors="pt")
    input_ids = encodings["input_ids"]

    if input_ids.shape[1] < 2:
        return 0.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return round(math.exp(loss.item()), 2)


if __name__ == "__main__":
    print("Wpisz lub wklej tekst po polsku.")
    print("Gdy skończysz, w nowej linii wciśnij Ctrl+D:")

    user_text = sys.stdin.read()

    if not user_text.strip():
        print("Nie podano żadnego tekstu — nic nie policzono.")
    else:
        burst = calculate_burstiness(user_text)
        ppl = calculate_perplexity(user_text)

        print("\n--- WYNIK ---")
        print("Liczba zdań:", burst["sentence_count"])
        print("Długości zdań (słowa):", burst["lengths"])
        print("Średnia długość:", burst["mean"])
        print("Odchylenie std:", burst["std_dev"])
        print("Burstiness:", burst["burstiness"])
        print("Perplexity:", ppl)
