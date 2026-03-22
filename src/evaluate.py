from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis import analogy, load_vocab, nearest_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned embeddings with neighbors and analogies.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to W_combined.npy")
    parser.add_argument("--out_file", type=str, default="results/logs/evaluation.json")
    args = parser.parse_args()

    word_to_idx, idx_to_word = load_vocab(args.processed_dir)
    emb = np.load(args.embeddings)

    query_words = ["king", "queen", "man", "woman", "love", "hate", "god", "death"]
    neighbors = {
        w: nearest_neighbors(w, emb, word_to_idx, idx_to_word, top_k=8)
        for w in query_words
        if w in word_to_idx
    }

    analogies = {
        "king - man + woman": analogy("man", "king", "woman", emb, word_to_idx, idx_to_word, top_k=5),
        "queen - woman + man": analogy("woman", "queen", "man", emb, word_to_idx, idx_to_word, top_k=5),
    }

    report = {
        "neighbors": neighbors,
        "analogies": analogies,
    }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved evaluation report: {args.out_file}")


if __name__ == "__main__":
    main()
