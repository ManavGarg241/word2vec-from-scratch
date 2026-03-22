from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Array1D = np.ndarray
Array2D = np.ndarray


@dataclass
class TrainConfig:
    embedding_dim: int = 50
    learning_rate: float = 0.025
    negative_samples: int = 5
    epochs: int = 3
    seed: int = 42


class Word2VecNegativeSampling:
    """Word2Vec with manual forward/backward for Skip-Gram and CBOW.

    No autograd. Pure NumPy parameter updates.
    """

    def __init__(
        self,
        vocab_size: int,
        config: TrainConfig,
        unigram_probs: Array1D,
    ) -> None:
        self.vocab_size = vocab_size
        self.config = config

        self.rng = np.random.default_rng(config.seed)

        bound = 0.5 / max(1, config.embedding_dim)
        self.W_in: Array2D = self.rng.uniform(
            low=-bound,
            high=bound,
            size=(vocab_size, config.embedding_dim),
        ).astype(np.float32)
        self.W_out: Array2D = np.zeros((vocab_size, config.embedding_dim), dtype=np.float32)

        probs = np.asarray(unigram_probs, dtype=np.float64)
        probs = probs / probs.sum()
        self.unigram_probs = probs

    @staticmethod
    def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        # Numerically stable sigmoid
        x_arr = np.asarray(x)
        out = np.empty_like(x_arr, dtype=np.float64)
        pos_mask = x_arr >= 0
        neg_mask = ~pos_mask
        out[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
        exp_x = np.exp(x_arr[neg_mask])
        out[neg_mask] = exp_x / (1.0 + exp_x)
        if np.isscalar(x):
            return float(out.item())
        return out

    @staticmethod
    def _neg_log_sigmoid(x: np.ndarray) -> np.ndarray:
        # -log(sigmoid(x)) = log(1 + exp(-x))
        return np.logaddexp(0.0, -x)

    @staticmethod
    def _neg_log_sigmoid_neg(x: np.ndarray) -> np.ndarray:
        # -log(sigmoid(-x)) = log(1 + exp(x))
        return np.logaddexp(0.0, x)

    def _sample_negative_indices(self, positive_idx: int, k: int) -> np.ndarray:
        negatives: List[int] = []
        while len(negatives) < k:
            cand = int(self.rng.choice(self.vocab_size, p=self.unigram_probs))
            if cand != positive_idx:
                negatives.append(cand)
        return np.asarray(negatives, dtype=np.int32)

    def skipgram_step(self, center_idx: int, positive_context_idx: int) -> float:
        v_c = self.W_in[center_idx].astype(np.float64)  # (D,)

        neg_indices = self._sample_negative_indices(
            positive_idx=positive_context_idx,
            k=self.config.negative_samples,
        )
        all_indices = np.concatenate(
            [np.asarray([positive_context_idx], dtype=np.int32), neg_indices], axis=0
        )

        u = self.W_out[all_indices].astype(np.float64)  # (K+1, D)
        scores = u @ v_c  # (K+1,)

        labels = np.zeros(scores.shape[0], dtype=np.float64)
        labels[0] = 1.0

        probs = self.sigmoid(scores)  # (K+1,)

        loss_pos = self._neg_log_sigmoid(scores[:1]).sum()
        loss_neg = self._neg_log_sigmoid_neg(scores[1:]).sum()
        loss = float(loss_pos + loss_neg)

        grad_scores = probs - labels  # dL/dscore_i
        grad_v_c = grad_scores @ u  # (D,)
        grad_u = np.outer(grad_scores, v_c)  # (K+1, D)

        self.W_in[center_idx] -= self.config.learning_rate * grad_v_c.astype(np.float32)
        np.add.at(self.W_out, all_indices, -self.config.learning_rate * grad_u.astype(np.float32))

        return loss

    def cbow_step(self, context_indices: Sequence[int], positive_target_idx: int) -> float:
        if len(context_indices) == 0:
            return 0.0

        ctx = np.asarray(context_indices, dtype=np.int32)
        v_ctx = self.W_in[ctx].astype(np.float64)  # (C, D)
        v_hat = v_ctx.mean(axis=0)  # (D,)

        neg_indices = self._sample_negative_indices(
            positive_idx=positive_target_idx,
            k=self.config.negative_samples,
        )
        all_indices = np.concatenate(
            [np.asarray([positive_target_idx], dtype=np.int32), neg_indices], axis=0
        )

        u = self.W_out[all_indices].astype(np.float64)
        scores = u @ v_hat

        labels = np.zeros(scores.shape[0], dtype=np.float64)
        labels[0] = 1.0
        probs = self.sigmoid(scores)

        loss_pos = self._neg_log_sigmoid(scores[:1]).sum()
        loss_neg = self._neg_log_sigmoid_neg(scores[1:]).sum()
        loss = float(loss_pos + loss_neg)

        grad_scores = probs - labels
        grad_v_hat = grad_scores @ u  # (D,)
        grad_u = np.outer(grad_scores, v_hat)  # (K+1, D)

        grad_each_ctx = grad_v_hat / len(ctx)

        np.add.at(self.W_in, ctx, -self.config.learning_rate * grad_each_ctx.astype(np.float32))
        np.add.at(self.W_out, all_indices, -self.config.learning_rate * grad_u.astype(np.float32))

        return loss

    def train_skipgram(
        self,
        pairs: np.ndarray,
        max_samples_per_epoch: int | None = None,
    ) -> List[float]:
        losses: List[float] = []
        n = len(pairs)
        for epoch in range(self.config.epochs):
            order = self.rng.permutation(n)
            if max_samples_per_epoch is not None:
                order = order[:max_samples_per_epoch]

            total_loss = 0.0
            for idx in order:
                center_idx, context_idx = pairs[idx]
                total_loss += self.skipgram_step(int(center_idx), int(context_idx))

            avg_loss = total_loss / max(1, len(order))
            losses.append(float(avg_loss))
            print(f"[Skip-Gram] Epoch {epoch + 1}/{self.config.epochs} - avg loss: {avg_loss:.4f}")

        return losses

    def train_cbow(
        self,
        samples: Iterable[Tuple[Sequence[int], int]],
        max_samples_per_epoch: int | None = None,
    ) -> List[float]:
        samples_list = list(samples)
        losses: List[float] = []
        n = len(samples_list)

        for epoch in range(self.config.epochs):
            order = self.rng.permutation(n)
            if max_samples_per_epoch is not None:
                order = order[:max_samples_per_epoch]

            total_loss = 0.0
            for idx in order:
                context, target = samples_list[idx]
                total_loss += self.cbow_step(context, int(target))

            avg_loss = total_loss / max(1, len(order))
            losses.append(float(avg_loss))
            print(f"[CBOW] Epoch {epoch + 1}/{self.config.epochs} - avg loss: {avg_loss:.4f}")

        return losses

    def get_input_embeddings(self) -> np.ndarray:
        return self.W_in.copy()

    def get_output_embeddings(self) -> np.ndarray:
        return self.W_out.copy()

    def get_combined_embeddings(self) -> np.ndarray:
        return (self.W_in + self.W_out) / 2.0
