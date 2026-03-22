from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from analysis import frequent_vs_rare_stats
from data import generate_cbow_samples, generate_skipgram_pairs
from train import load_cbow_samples, load_skipgram_pairs, load_vocab
from word2vec import TrainConfig, Word2VecNegativeSampling


def run_single(
    processed_dir: str,
    mode: str,
    embedding_dim: int,
    neg_samples: int,
    epochs: int,
    lr: float,
    seed: int,
    max_samples_per_epoch: int,
    window_size: int | None = None,
) -> Dict:
    vocab = load_vocab(processed_dir)
    vocab_size = len(vocab)
    unigram_probs = np.load(os.path.join(processed_dir, "unigram_probs.npy"))

    cfg = TrainConfig(
        embedding_dim=embedding_dim,
        learning_rate=lr,
        negative_samples=neg_samples,
        epochs=epochs,
        seed=seed,
    )

    model = Word2VecNegativeSampling(
        vocab_size=vocab_size,
        config=cfg,
        unigram_probs=unigram_probs,
    )

    max_samples = max_samples_per_epoch if max_samples_per_epoch > 0 else None
    token_ids = np.load(os.path.join(processed_dir, "token_ids.npy"))

    if mode == "skipgram":
        if window_size is None:
            pairs = load_skipgram_pairs(processed_dir)
        else:
            pairs = np.asarray(generate_skipgram_pairs(token_ids.tolist(), window_size=window_size), dtype=np.int32)
        losses = model.train_skipgram(pairs, max_samples_per_epoch=max_samples)
    else:
        if window_size is None:
            samples = load_cbow_samples(processed_dir)
        else:
            samples = generate_cbow_samples(token_ids.tolist(), window_size=window_size)
        losses = model.train_cbow(samples, max_samples_per_epoch=max_samples)

    emb = model.get_combined_embeddings()
    fr_stats = frequent_vs_rare_stats(embeddings=emb, token_ids=token_ids, top_n=100)

    return {
        "mode": mode,
        "config": asdict(cfg),
        "losses": losses,
        "final_loss": float(losses[-1]) if losses else None,
        "freq_rare": fr_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact Word2Vec experiments.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mode", type=str, choices=["skipgram", "cbow"], default="skipgram")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples_per_epoch", type=int, default=6000)
    parser.add_argument("--out_file", type=str, default="results/logs/experiments_summary.json")
    args = parser.parse_args()

    results: Dict[str, List[Dict]] = {
        "window_size_effect": [],
        "embedding_dim_effect": [],
        "negative_samples_effect": [],
    }

    for window in [2, 3, 4]:
        res = run_single(
            processed_dir=args.processed_dir,
            mode=args.mode,
            embedding_dim=50,
            neg_samples=5,
            epochs=args.epochs,
            lr=args.learning_rate,
            seed=args.seed,
            max_samples_per_epoch=args.max_samples_per_epoch,
            window_size=window,
        )
        res["window_size"] = window
        results["window_size_effect"].append(res)

    for dim in [50, 75, 100]:
        res = run_single(
            processed_dir=args.processed_dir,
            mode=args.mode,
            embedding_dim=dim,
            neg_samples=5,
            epochs=args.epochs,
            lr=args.learning_rate,
            seed=args.seed,
            max_samples_per_epoch=args.max_samples_per_epoch,
            window_size=2,
        )
        results["embedding_dim_effect"].append(res)

    for neg in [2, 5, 10]:
        res = run_single(
            processed_dir=args.processed_dir,
            mode=args.mode,
            embedding_dim=50,
            neg_samples=neg,
            epochs=args.epochs,
            lr=args.learning_rate,
            seed=args.seed,
            max_samples_per_epoch=args.max_samples_per_epoch,
            window_size=2,
        )
        results["negative_samples_effect"].append(res)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved experiment summary: {args.out_file}")


if __name__ == "__main__":
    main()
