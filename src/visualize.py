from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_vocab(processed_dir: str) -> tuple[dict[str, int], dict[int, str]]:
    with open(os.path.join(processed_dir, "vocab_word_to_idx.json"), "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def plot_projection(
    embeddings: np.ndarray,
    words: list[str],
    out_path: str,
    method: str,
) -> None:
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=12, alpha=0.75)

    for i, w in enumerate(words):
        plt.annotate(w, (embeddings[i, 0], embeddings[i, 1]), fontsize=8, alpha=0.85)

    plt.title(f"Word Embedding {method.upper()} Projection")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize word embeddings with PCA and t-SNE.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to W_combined.npy")
    parser.add_argument("--top_words", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="results/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    word_to_idx, idx_to_word = load_vocab(args.processed_dir)
    emb = np.load(args.embeddings)

    # Top words by index order (already sorted by frequency in data prep)
    top_n = min(args.top_words, len(word_to_idx))
    chosen_idx = np.arange(top_n)
    chosen_emb = emb[chosen_idx]
    chosen_words = [idx_to_word[int(i)] for i in chosen_idx]

    pca = PCA(n_components=2, random_state=args.seed)
    pca_2d = pca.fit_transform(chosen_emb)
    plot_projection(
        embeddings=pca_2d,
        words=chosen_words,
        out_path=os.path.join(args.out_dir, "pca_top_words.png"),
        method="pca",
    )

    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto")
    tsne_2d = tsne.fit_transform(chosen_emb)
    plot_projection(
        embeddings=tsne_2d,
        words=chosen_words,
        out_path=os.path.join(args.out_dir, "tsne_top_words.png"),
        method="tsne",
    )

    print("Saved PCA and t-SNE plots.")


if __name__ == "__main__":
    main()
