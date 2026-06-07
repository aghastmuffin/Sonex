import argparse
import math
import os
import re
from statistics import mean, stdev
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0].lower()
    base = base.replace("_", " ")
    base = re.sub(r"[^\w\s]", "", base, flags=re.UNICODE)
    base = re.sub(r"\s+", " ", base)
    return base.strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m] / max(1, n)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def index_text_files(directory: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if not os.path.isfile(full_path):
            continue
        if not filename.lower().endswith(".txt"):
            continue
        out[normalize_name(filename)] = full_path
    return out


def compare_dirs(equiv_dir: str, clean_dir: str) -> List[Tuple[str, float, float]]:
    equiv_files = index_text_files(equiv_dir)
    clean_files = index_text_files(clean_dir)

    results: List[Tuple[str, float, float]] = []

    for name_key, equiv_path in sorted(equiv_files.items()):
        clean_path = clean_files.get(name_key)
        if not clean_path:
            continue

        equiv_text = normalize_text(read_text_file(equiv_path))
        clean_text = normalize_text(read_text_file(clean_path))

        wer = word_error_rate(clean_text, equiv_text)
        accuracy = max(0.0, (1.0 - wer) * 100.0)
        error = 100.0 - accuracy
        results.append((name_key, accuracy, error))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Whisper output lyrics in equivlyrics against clean_lyrics references "
            "and report accuracy with margin of error."
        )
    )
    parser.add_argument(
        "--equiv-dir",
        default=os.path.join(os.path.dirname(__file__), "equivlyrics"),
        help="Directory containing generated equivalence lyrics (default: lyrics_study/equivlyrics)",
    )
    parser.add_argument(
        "--clean-dir",
        default=os.path.join(os.path.dirname(__file__), "clean_lyrics"),
        help="Reference lyrics directory (default: lyrics_study/clean_lyrics)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.equiv_dir):
        raise FileNotFoundError(f"equivlyrics directory not found: {args.equiv_dir}")
    if not os.path.isdir(args.clean_dir):
        raise FileNotFoundError(f"clean_lyrics directory not found: {args.clean_dir}")

    results = compare_dirs(args.equiv_dir, args.clean_dir)
    if not results:
        print("No matching .txt files found between equivlyrics and clean_lyrics.")
        return

    print("Per-file Whisper Accuracy (reference = clean_lyrics):")
    for name_key, accuracy, error in results:
        print(f"- {name_key}: {accuracy:.2f}% accuracy (margin of error: +/-{error:.2f}%)")

    accuracies = [row[1] for row in results]
    avg_accuracy = mean(accuracies)
    sample_count = len(accuracies)

    if sample_count > 1:
        sample_std = stdev(accuracies)
        standard_error = sample_std / math.sqrt(sample_count)
        margin_95 = 1.96 * standard_error
        margin_2se = 2.0 * standard_error
    else:
        standard_error = 0.0
        margin_95 = 0.0
        margin_2se = 0.0

    print("\nSummary:")
    print(f"- Compared files: {sample_count}")
    print(f"- Mean accuracy: {avg_accuracy:.2f}%")
    print(f"- Standard error (SE): {standard_error:.2f}%")
    print(f"- Dataset margin (2*SE): +/-{margin_2se:.2f}%")
    print(f"- Dataset margin of error (95% CI): +/-{margin_95:.2f}%")


if __name__ == "__main__":
    main()
