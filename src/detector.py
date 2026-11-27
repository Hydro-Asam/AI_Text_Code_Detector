import math
import re
import sys


def split_sentences(text: str) -> list[str]:
    """Dzieli tekst na zdania po znakach końca zdania (. ! ?)."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def count_words(sentence: str) -> int:
    """Liczy słowa w pojedynczym zdaniu."""
    return len(re.findall(r"\b\w+\b", sentence))


def calculate_burstiness(text: str) -> dict[str, float | int | list[int]]:
    """
    Liczy podstawowe metryki burstiness na bazie długości zdań (w słowach):
    - sentence_count: liczba zdań
    - lengths: lista długości każdego zdania
    - mean: średnia długość zdania
    - std_dev: odchylenie standardowe długości zdań
    - burstiness: współczynnik zmienności (std_dev / mean), czyli jak bardzo skacze rytm
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

if __name__ == "__main__":
    print("Wpisz lub wklej tekst do analizy.")
    print("Gdy skończysz, w nowej linii wciśnij Ctrl+D (i Enter, jeśli trzeba):")

    # Czyta cały tekst ze standardowego wejścia (stdin) aż do EOF (Ctrl+D)
    user_text = sys.stdin.read()

    if not user_text.strip():
        print("Nie podano żadnego tekstu — nic nie policzono.")
    else:
        result = calculate_burstiness(user_text)
        print("\n--- WYNIK ---")
        print("Liczba zdań:", result["sentence_count"])
        print("Długości zdań (słowa):", result["lengths"])
        print("Średnia długość:", result["mean"])
        print("Odchylenie std:", result["std_dev"])
        print("Burstiness:", result["burstiness"])
