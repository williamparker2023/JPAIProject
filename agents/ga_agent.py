from __future__ import annotations

from typing import Sequence
import numpy as np

from agents.common import Agent
from env.features import compact_features
from env.snake_env import Action


class GAAgent(Agent):
    """
    MLP policy with 1 hidden layer. Genome is a flat numpy array containing:
      W1 (H x I), b1 (H,), W2 (O x H), b2 (O,)
    Input size = 11 (compact features), hidden = 16, output = 4 (actions).
    """

    IN_SIZE = 11
    HIDDEN = 16
    OUT_SIZE = 4

    def __init__(self, genome: np.ndarray):
        self.genome = genome.astype(np.float32)
        self._unpack()

    @classmethod
    def genome_size(cls) -> int:
        return cls.HIDDEN * cls.IN_SIZE + cls.HIDDEN + cls.OUT_SIZE * cls.HIDDEN + cls.OUT_SIZE

    @classmethod
    def random_genome(cls, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        return rng.normal(scale=scale, size=(cls.genome_size(),)).astype(np.float32)

    def _unpack(self) -> None:
        g = self.genome
        i = 0
        e = self.HIDDEN * self.IN_SIZE
        self.W1 = g[i : i + e].reshape(self.HIDDEN, self.IN_SIZE); i += e
        e = self.HIDDEN
        self.b1 = g[i : i + e].reshape(self.HIDDEN); i += e
        e = self.OUT_SIZE * self.HIDDEN
        self.W2 = g[i : i + e].reshape(self.OUT_SIZE, self.HIDDEN); i += e
        e = self.OUT_SIZE
        self.b2 = g[i : i + e].reshape(self.OUT_SIZE)

    def forward_logits(self, feat: Sequence[float]) -> np.ndarray:
        x = np.asarray(feat, dtype=np.float32)
        h = np.tanh(self.W1.dot(x) + self.b1)
        logits = self.W2.dot(h) + self.b2
        return logits

    def act(self, obs: dict) -> Action:
        feats = compact_features(obs)  # numpy array (11,)
        logits = self.forward_logits(feats)
        # pick highest-scoring legal action (avoid 180-degree)
        legal = obs["legal_actions"] if "legal_actions" in obs else None
        # env provides legal_actions() method, but obs may not include it; fallback to all
        cand_actions = list(range(4)) if legal is None else [int(a) for a in legal]
        # sort logits by descending and pick first legal
        order = np.argsort(-logits)
        for idx in order:
            if int(idx) in cand_actions:
                return Action(int(idx))
        # fallback
        return Action(int(order[0]))