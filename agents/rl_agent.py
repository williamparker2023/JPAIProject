from __future__ import annotations

import numpy as np
from typing import Optional

from env.features import compact_features
from env.snake_env import Action


class DQNAgent:
    """Small MLP DQN implemented in numpy.

    Architecture: input (11) -> hidden (128) tanh -> output (4)
    Methods: act(epsilon), q_values, train_on_batch, save/load
    """

    def __init__(self, W1: Optional[np.ndarray] = None, b1: Optional[np.ndarray] = None,
                 W2: Optional[np.ndarray] = None, b2: Optional[np.ndarray] = None,
                 hidden: int = 128):
        self.IN = 11  # Original features with scaled hidden layer
        self.OUT = 4
        self.H = int(hidden)

        rng = np.random.default_rng(0)
        self.W1 = W1.astype(np.float32) if W1 is not None else rng.normal(0, 0.1, size=(self.H, self.IN)).astype(np.float32)
        self.b1 = b1.astype(np.float32) if b1 is not None else np.zeros((self.H,), dtype=np.float32)
        self.W2 = W2.astype(np.float32) if W2 is not None else rng.normal(0, 0.1, size=(self.OUT, self.H)).astype(np.float32)
        self.b2 = b2.astype(np.float32) if b2 is not None else np.zeros((self.OUT,), dtype=np.float32)

    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        z = np.load(path)
        return cls(W1=z['W1'], b1=z['b1'], W2=z['W2'], b2=z['b2'])

    def _forward_feat(self, feat: np.ndarray):
        z1 = self.W1.dot(feat) + self.b1
        h = np.tanh(z1)
        out = self.W2.dot(h) + self.b2
        return out, h, z1

    def q_values(self, obs: dict) -> np.ndarray:
        feat = compact_features(obs)
        q, _, _ = self._forward_feat(feat)
        return q

    def act(self, obs: dict, epsilon: float = 0.0) -> Action:
        legal = obs.get('legal_actions', None)
        if np.random.random() < epsilon:
            if legal is None:
                a = np.random.randint(0, self.OUT)
            else:
                a = np.random.choice([int(x) for x in legal])
            return Action(a)
        qs = self.q_values(obs)
        order = np.argsort(-qs)
        if legal is None:
            return Action(int(order[0]))
        for idx in order:
            if int(idx) in [int(x) for x in legal]:
                return Action(int(idx))
        return Action(int(order[0]))

    def train_on_batch(self, batch, lr: float, gamma: float, target_net: Optional['DQNAgent'] = None):
        # batch: list of tuples (obs, action, reward, next_obs, done)
        # perform SGD on MSE loss with Double DQN target
        grads = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1), 'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        loss = 0.0
        for obs, action, reward, next_obs, done in batch:
            feat = compact_features(obs)
            q, h, z1 = self._forward_feat(feat)
            q_a = q[int(action)]

            # compute target with Double DQN
            if done:
                target = float(reward)
            else:
                # Double DQN: online net selects argmax, target net evaluates
                if target_net is None:
                    # Fallback to regular DQN if no target network
                    q_next, _, _ = self._forward_feat(compact_features(next_obs))
                    max_q_next = float(np.max(q_next))
                else:
                    # Online net picks best action
                    q_next_online, _, _ = self._forward_feat(compact_features(next_obs))
                    a_max = int(np.argmax(q_next_online))
                    # Target net evaluates that action
                    q_next_target, _, _ = target_net._forward_feat(compact_features(next_obs))
                    max_q_next = float(q_next_target[a_max])
                target = float(reward) + gamma * max_q_next

            td_error = target - q_a
            loss += td_error ** 2

            # compute gradients
            dq = np.zeros((self.OUT,), dtype=np.float32)
            dq[int(action)] = -2.0 * td_error

            grad_W2 = np.outer(dq, h)
            grad_b2 = dq

            dh = self.W2.T.dot(dq)
            dz1 = dh * (1.0 - np.tanh(z1) ** 2)
            grad_W1 = np.outer(dz1, feat)
            grad_b1 = dz1

            grads['W1'] += grad_W1
            grads['b1'] += grad_b1
            grads['W2'] += grad_W2
            grads['b2'] += grad_b2

        # apply gradients (average over batch)
        mb = float(len(batch))
        # gradient clipping to stabilize updates
        for k in grads:
            grads[k] = np.clip(grads[k], -10.0, 10.0)
        self.W1 -= (lr / mb) * grads['W1']
        self.b1 -= (lr / mb) * grads['b1']
        self.W2 -= (lr / mb) * grads['W2']
        self.b2 -= (lr / mb) * grads['b2']
        return float(loss / mb)

    def update_step(self, obs: dict, action: Action, reward: float, next_obs: dict, done: bool, lr: float = 1e-3, gamma: float = 0.99, target_net: Optional['DQNAgent'] = None) -> float:
        """Compatibility single-step TD update.

        Returns TD error (target - q).
        """
        feat = compact_features(obs)
        q, h, z1 = self._forward_feat(feat)
        q_a = float(q[int(action)])

        if done:
            target = float(reward)
        else:
            # Double DQN single-step
            if target_net is None:
                q_next, _, _ = self._forward_feat(compact_features(next_obs))
                max_q_next = float(np.max(q_next))
            else:
                q_next_online, _, _ = self._forward_feat(compact_features(next_obs))
                a_max = int(np.argmax(q_next_online))
                q_next_target, _, _ = target_net._forward_feat(compact_features(next_obs))
                max_q_next = float(q_next_target[a_max])
            target = float(reward) + gamma * max_q_next

        td = target - q_a

        # compute gradients for single-sample
        dq = np.zeros((self.OUT,), dtype=np.float32)
        dq[int(action)] = -2.0 * td

        grad_W2 = np.outer(dq, h)
        grad_b2 = dq

        dh = self.W2.T.dot(dq)
        dz1 = dh * (1.0 - np.tanh(z1) ** 2)
        grad_W1 = np.outer(dz1, feat)
        grad_b1 = dz1

        # apply gradients
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

        return float(td)


# keep backward compatibility name
RLAgent = DQNAgent