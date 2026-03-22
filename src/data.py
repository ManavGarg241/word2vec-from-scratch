from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def load_corpus(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: str, lowercase: bool = True) -> str:
    if lowercase:
        text = text.lower()
    # Keep letters, apostrophes, and spaces. Remove everything else.
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    # Words like "don't" stay one token.
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def build_vocab(
    tokens: List[str],
    min_count: int = 1,
    max_vocab_size: int | None = None,
) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    freq = Counter(tokens)

    items = [(w, c) for w, c in freq.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        items = items[:max_vocab_size]

    word_to_idx = {w: i for i, (w, _) in enumerate(items)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word, freq


def encode_tokens(tokens: List[str], word_to_idx: Dict[str, int]) -> List[int]:
    return [word_to_idx[t] for t in tokens if t in word_to_idx]


def generate_skipgram_pairs(token_ids: List[int], window_size: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)
        for j in range(left, right):
            if j == i:
                continue
            context = token_ids[j]
            pairs.append((center, context))
    return pairs


def generate_cbow_samples(token_ids: List[int], window_size: int) -> List[Tuple[List[int], int]]:
    samples: List[Tuple[List[int], int]] = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)

        context: List[int] = []
        for j in range(left, right):
            if j == i:
                continue
            context.append(token_ids[j])

        if context:
            samples.append((context, center))
    return samples


def build_unigram_distribution(
    word_to_idx: Dict[str, int],
    full_freq: Counter,
    power: float = 0.75,
) -> np.ndarray:
    counts = np.array([full_freq[w] for w in word_to_idx.keys()], dtype=np.float64)
    probs = counts**power
    probs = probs / probs.sum()
    return probs


def save_processed_data(
    out_dir: str,
    tokens: List[str],
    token_ids: List[int],
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    skipgram_pairs: List[Tuple[int, int]],
    cbow_samples: List[Tuple[List[int], int]],
    unigram_probs: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "vocab_word_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "vocab_idx_to_word.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_word.items()}, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(out_dir, "token_ids.npy"), np.array(token_ids, dtype=np.int32))
    np.save(os.path.join(out_dir, "unigram_probs.npy"), unigram_probs.astype(np.float32))

    # Save skip-gram pairs as (N, 2)
    if skipgram_pairs:
        np.save(os.path.join(out_dir, "skipgram_pairs.npy"), np.array(skipgram_pairs, dtype=np.int32))
    else:
        np.save(os.path.join(out_dir, "skipgram_pairs.npy"), np.zeros((0, 2), dtype=np.int32))

    # Save CBOW samples as JSON because context length can vary at sequence boundaries
    with open(os.path.join(out_dir, "cbow_samples.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"context": ctx, "target": target} for ctx, target in cbow_samples],
            f,
            ensure_ascii=False,
        )

    summary = {
        "num_raw_tokens": len(tokens),
        "num_encoded_tokens": len(token_ids),
        "vocab_size": len(word_to_idx),
        "num_skipgram_pairs": len(skipgram_pairs),
        "num_cbow_samples": len(cbow_samples),
    }
    with open(os.path.join(out_dir, "data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for Word2Vec (CBOW + Skip-Gram).")
    parser.add_argument("--input", type=str, required=True, help="Path to raw text corpus.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory.")
    parser.add_argument("--window_size", type=int, default=2, help="Context window size.")
    parser.add_argument("--min_count", type=int, default=2, help="Minimum token frequency to keep.")
    parser.add_argument("--max_vocab_size", type=int, default=8000, help="Maximum vocab size.")
    args = parser.parse_args()

    raw = load_corpus(args.input)
    cleaned = normalize_text(raw, lowercase=True)
    tokens = tokenize(cleaned)

    word_to_idx, idx_to_word, freq = build_vocab(
        tokens=tokens,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
    )

    token_ids = encode_tokens(tokens, word_to_idx)
    skipgram_pairs = generate_skipgram_pairs(token_ids, window_size=args.window_size)
    cbow_samples = generate_cbow_samples(token_ids, window_size=args.window_size)
    unigram_probs = build_unigram_distribution(word_to_idx, freq, power=0.75)

    save_processed_data(
        out_dir=args.output_dir,
        tokens=tokens,
        token_ids=token_ids,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        skipgram_pairs=skipgram_pairs,
        cbow_samples=cbow_samples,
        unigram_probs=unigram_probs,
    )

    print("Data preparation complete.")
    print(f"Vocab size: {len(word_to_idx)}")
    print(f"Skip-Gram pairs: {len(skipgram_pairs)}")
    print(f"CBOW samples: {len(cbow_samples)}")


if __name__ == "__main__":
    main()
