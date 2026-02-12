from __future__ import annotations

from typing import Optional
import numpy as np

from env.features_extended_v2 import extended_features_v2
from env.snake_env import Action


class DQNExtendedV2Agent:
    """
    DQN agent using FIXED extended 21-feature representation (v2 with safety checks).
    
    Architecture: input (21) -> hidden (128) tanh -> output (4)
    Same proven training loop as original DQN, but with improved features.
    """

    IN = 21  # Extended features with safety checks
    OUT = 4
    H = 128

    def __init__(self, W1: Optional[np.ndarray] = None, b1: Optional[np.ndarray] = None,
                 W2: Optional[np.ndarray] = None, b2: Optional[np.ndarray] = None):
        rng = np.random.default_rng(0)
        self.W1 = W1.astype(np.float32) if W1 is not None else rng.normal(0, 0.1, size=(self.H, self.IN)).astype(np.float32)
        self.b1 = b1.astype(np.float32) if b1 is not None else np.zeros((self.H,), dtype=np.float32)
        self.W2 = W2.astype(np.float32) if W2 is not None else rng.normal(0, 0.1, size=(self.OUT, self.H)).astype(np.float32)
        self.b2 = b2.astype(np.float32) if b2 is not None else np.zeros((self.OUT,), dtype=np.float32)

    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "DQNExtendedV2Agent":
        z = np.load(path)
        return cls(W1=z['W1'], b1=z['b1'], W2=z['W2'], b2=z['b2'])

    def _forward_feat(self, feat: np.ndarray):
        z1 = self.W1.dot(feat) + self.b1
        h = np.tanh(z1)
        out = self.W2.dot(h) + self.b2
        return out, h, z1

    def q_values(self, obs: dict) -> np.ndarray:
        feat = extended_features_v2(obs)
        q, _, _ = self._forward_feat(feat)
        return q

    def act(self, obs: dict, epsilon: float = 0.0) -> Action:
        legal = obs.get('legal_actions', None)
        if np.random.random() < epsilon:
            if legal is None:
                return Action(np.random.randint(0, 4))
            return Action(int(np.random.choice(legal)))
        qs = self.q_values(obs)
        order = np.argsort(-qs)
        if legal is None:
            return Action(int(order[0]))
        for idx in order:
            if int(idx) in legal:
                return Action(int(idx))
        return Action(int(order[0]))

    def train_on_batch(self, batch, lr: float, gamma: float, target_net: Optional['DQNExtendedV2Agent'] = None):
        """Batch training with Double DQN target."""
        grads = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1), 
                 'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        loss = 0.0
        
        for obs, action, reward, next_obs, done in batch:
            feat = extended_features_v2(obs)
            q, h, z1 = self._forward_feat(feat)
            q_a = float(q[int(action)])

            if done:
                target = reward
            else:
                # Double DQN: use online network to select, target to evaluate
                next_qs_online = self.q_values(next_obs)
                best_action = int(np.argmax(next_qs_online))
                
                if target_net is not None:
                    next_qs_target = target_net.q_values(next_obs)
                    target_val = float(next_qs_target[best_action])
                else:
                    next_feat = extended_features_v2(next_obs)
                    next_q, _, _ = self._forward_feat(next_feat)
                    target_val = float(next_q[best_action])
                
                target = reward + gamma * target_val

            td_error = target - q_a
            loss += td_error ** 2

            # Backprop
            dq = np.zeros((self.OUT,), dtype=np.float32)
            dq[int(action)] = -2.0 * td_error

            grad_W2 = np.outer(dq, h)
            grad_b2 = dq

            dh = self.W2.T.dot(dq)
            dz1 = dh * (1.0 - np.tanh(z1) ** 2)
            grad_W1 = np.outer(dz1, feat)
            grad_b1 = dz1

            grads['W2'] += grad_W2
            grads['b2'] += grad_b2
            grads['W1'] += grad_W1
            grads['b1'] += grad_b1

        # Apply gradients
        mb = float(len(batch))
        for k in grads:
            grads[k] = np.clip(grads[k], -10, 10)
        
        self.W1 -= (lr / mb) * grads['W1']
        self.b1 -= (lr / mb) * grads['b1']
        self.W2 -= (lr / mb) * grads['W2']
        self.b2 -= (lr / mb) * grads['b2']
        
        return float(loss / mb)
