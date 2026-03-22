from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np

from word2vec import TrainConfig, Word2VecNegativeSampling


def load_skipgram_pairs(processed_dir: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, "skipgram_pairs.npy"))


def load_cbow_samples(processed_dir: str) -> List[Tuple[List[int], int]]:
    path = os.path.join(processed_dir, "cbow_samples.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(row["context"], row["target"]) for row in data]


def load_vocab(processed_dir: str) -> dict:
    with open(os.path.join(processed_dir, "vocab_word_to_idx.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def save_training_outputs(
    out_dir: str,
    mode: str,
    embedding_dim: int,
    neg_samples: int,
    lr: float,
    epochs: int,
    losses: List[float],
    W_in: np.ndarray,
    W_out: np.ndarray,
    W_combined: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    run_name = f"{mode}_d{embedding_dim}_neg{neg_samples}_lr{lr}_ep{epochs}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, "W_in.npy"), W_in)
    np.save(os.path.join(run_dir, "W_out.npy"), W_out)
    np.save(os.path.join(run_dir, "W_combined.npy"), W_combined)

    with open(os.path.join(run_dir, "losses.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "losses": losses}, f, indent=2)

    with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "embedding_dim": embedding_dim,
                "negative_samples": neg_samples,
                "learning_rate": lr,
                "epochs": epochs,
            },
            f,
            indent=2,
        )

    print(f"Saved outputs to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Word2Vec (Skip-Gram or CBOW) with negative sampling.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mode", type=str, choices=["skipgram", "cbow"], required=True)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--neg_samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples_per_epoch", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/logs")
    args = parser.parse_args()

    vocab = load_vocab(args.processed_dir)
    vocab_size = len(vocab)
    unigram_probs = np.load(os.path.join(args.processed_dir, "unigram_probs.npy"))

    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        negative_samples=args.neg_samples,
        epochs=args.epochs,
        seed=args.seed,
    )

    model = Word2VecNegativeSampling(
        vocab_size=vocab_size,
        config=config,
        unigram_probs=unigram_probs,
    )

    max_samples = args.max_samples_per_epoch if args.max_samples_per_epoch > 0 else None

    if args.mode == "skipgram":
        pairs = load_skipgram_pairs(args.processed_dir)
        losses = model.train_skipgram(pairs, max_samples_per_epoch=max_samples)
    else:
        samples = load_cbow_samples(args.processed_dir)
        losses = model.train_cbow(samples, max_samples_per_epoch=max_samples)

    save_training_outputs(
        out_dir=args.output_dir,
        mode=args.mode,
        embedding_dim=args.embedding_dim,
        neg_samples=args.neg_samples,
        lr=args.learning_rate,
        epochs=args.epochs,
        losses=losses,
        W_in=model.get_input_embeddings(),
        W_out=model.get_output_embeddings(),
        W_combined=model.get_combined_embeddings(),
    )


if __name__ == "__main__":
    main()
