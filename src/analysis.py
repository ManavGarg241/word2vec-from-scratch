from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def load_vocab(processed_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(os.path.join(processed_dir, "vocab_word_to_idx.json"), "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def load_token_ids(processed_dir: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, "token_ids.npy"))


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def nearest_neighbors(
    word: str,
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    if word not in word_to_idx:
        return []

    e = l2_normalize_rows(embeddings)
    q_idx = word_to_idx[word]
    q = e[q_idx]
    sims = e @ q
    sims[q_idx] = -1.0

    top_idx = np.argsort(-sims)[:top_k]
    return [(idx_to_word[int(i)], float(sims[i])) for i in top_idx]


def analogy(
    a: str,
    b: str,
    c: str,
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    if any(w not in word_to_idx for w in [a, b, c]):
        return []

    e = l2_normalize_rows(embeddings)
    va = e[word_to_idx[a]]
    vb = e[word_to_idx[b]]
    vc = e[word_to_idx[c]]

    query = vb - va + vc
    query = query / (np.linalg.norm(query) + 1e-9)

    sims = e @ query
    for w in [a, b, c]:
        sims[word_to_idx[w]] = -1.0

    top_idx = np.argsort(-sims)[:top_k]
    return [(idx_to_word[int(i)], float(sims[i])) for i in top_idx]


def frequent_vs_rare_stats(
    embeddings: np.ndarray,
    token_ids: np.ndarray,
    top_n: int = 100,
) -> Dict[str, float]:
    freq = Counter(token_ids.tolist())
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    top_ids = [wid for wid, _ in sorted_items[:top_n]]
    rare_ids = [wid for wid, _ in sorted_items[-top_n:]]

    emb_norms = np.linalg.norm(embeddings, axis=1)

    top_mean_norm = float(np.mean(emb_norms[top_ids])) if top_ids else 0.0
    rare_mean_norm = float(np.mean(emb_norms[rare_ids])) if rare_ids else 0.0

    return {
        "top_n": top_n,
        "frequent_mean_norm": top_mean_norm,
        "rare_mean_norm": rare_mean_norm,
        "norm_gap": top_mean_norm - rare_mean_norm,
    }
