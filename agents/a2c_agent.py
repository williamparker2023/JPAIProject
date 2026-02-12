from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from env.features_extended import extended_features
from env.snake_env import Action


class A2CAgent:
    """
    Actor-Critic agent for Snake.
    
    Architecture:
    - Shared trunk: input (21) -> hidden (64) tanh
    - Actor head: hidden (32) -> softmax over 4 actions
    - Critic head: hidden (32) -> 1 value estimate
    
    Training: A2C with advantage-based updates and entropy regularization.
    """

    IN = 21  # Extended features
    TRUNK = 64
    ACTOR_HEAD = 32
    CRITIC_HEAD = 32
    OUT = 4  # Actions
    
    def __init__(self, W_trunk: Optional[np.ndarray] = None, b_trunk: Optional[np.ndarray] = None,
                 W_actor: Optional[np.ndarray] = None, b_actor: Optional[np.ndarray] = None,
                 W_actor_out: Optional[np.ndarray] = None, b_actor_out: Optional[np.ndarray] = None,
                 W_critic: Optional[np.ndarray] = None, b_critic: Optional[np.ndarray] = None,
                 W_critic_out: Optional[np.ndarray] = None, b_critic_out: Optional[np.ndarray] = None):
        
        rng = np.random.default_rng(42)
        std = 0.1
        
        # Shared trunk
        self.W_trunk = W_trunk.astype(np.float32) if W_trunk is not None else rng.normal(0, std, (self.TRUNK, self.IN)).astype(np.float32)
        self.b_trunk = b_trunk.astype(np.float32) if b_trunk is not None else np.zeros((self.TRUNK,), dtype=np.float32)
        
        # Actor head
        self.W_actor = W_actor.astype(np.float32) if W_actor is not None else rng.normal(0, std, (self.ACTOR_HEAD, self.TRUNK)).astype(np.float32)
        self.b_actor = b_actor.astype(np.float32) if b_actor is not None else np.zeros((self.ACTOR_HEAD,), dtype=np.float32)
        self.W_actor_out = W_actor_out.astype(np.float32) if W_actor_out is not None else rng.normal(0, std, (self.OUT, self.ACTOR_HEAD)).astype(np.float32)
        self.b_actor_out = b_actor_out.astype(np.float32) if b_actor_out is not None else np.zeros((self.OUT,), dtype=np.float32)
        
        # Critic head
        self.W_critic = W_critic.astype(np.float32) if W_critic is not None else rng.normal(0, std, (self.CRITIC_HEAD, self.TRUNK)).astype(np.float32)
        self.b_critic = b_critic.astype(np.float32) if b_critic is not None else np.zeros((self.CRITIC_HEAD,), dtype=np.float32)
        self.W_critic_out = W_critic_out.astype(np.float32) if W_critic_out is not None else rng.normal(0, 0.05, (1, self.CRITIC_HEAD)).astype(np.float32)
        self.b_critic_out = b_critic_out.astype(np.float32) if b_critic_out is not None else np.zeros((1,), dtype=np.float32)

    def save(self, path: str) -> None:
        np.savez(path, 
                 W_trunk=self.W_trunk, b_trunk=self.b_trunk,
                 W_actor=self.W_actor, b_actor=self.b_actor, 
                 W_actor_out=self.W_actor_out, b_actor_out=self.b_actor_out,
                 W_critic=self.W_critic, b_critic=self.b_critic,
                 W_critic_out=self.W_critic_out, b_critic_out=self.b_critic_out)

    @classmethod
    def load(cls, path: str) -> "A2CAgent":
        z = np.load(path)
        return cls(
            W_trunk=z['W_trunk'], b_trunk=z['b_trunk'],
            W_actor=z['W_actor'], b_actor=z['b_actor'],
            W_actor_out=z['W_actor_out'], b_actor_out=z['b_actor_out'],
            W_critic=z['W_critic'], b_critic=z['b_critic'],
            W_critic_out=z['W_critic_out'], b_critic_out=z['b_critic_out']
        )

    def _forward_trunk(self, feat: np.ndarray) -> np.ndarray:
        """Shared trunk: tanh activation"""
        z = self.W_trunk.dot(feat) + self.b_trunk
        h = np.tanh(z)
        return h

    def _forward_actor(self, trunk_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Actor head: returns logits and hidden state for backprop"""
        h = np.tanh(self.W_actor.dot(trunk_feat) + self.b_actor)
        logits = self.W_actor_out.dot(h) + self.b_actor_out
        return logits.flatten(), h

    def _forward_critic(self, trunk_feat: np.ndarray) -> Tuple[float, np.ndarray]:
        """Critic head: returns value estimate and hidden state for backprop"""
        h = np.tanh(self.W_critic.dot(trunk_feat) + self.b_critic)
        value_out = self.W_critic_out.dot(h) + self.b_critic_out
        value = float(value_out[0]) if value_out.ndim > 0 else float(value_out)
        return value, h

    def forward(self, obs: dict) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Full forward pass.
        Returns: (policy_logits, value_estimate, trunk_hidden_state)
        """
        feat = extended_features(obs)
        trunk_h = self._forward_trunk(feat)
        logits, _ = self._forward_actor(trunk_h)
        value, _ = self._forward_critic(trunk_h)
        return logits, value, trunk_h

    def act(self, obs: dict, training: bool = False) -> Action:
        """
        Sample action from policy.
        If training=True, returns sampled action (exploration).
        If training=False, returns greedy action (max policy).
        """
        logits, _, _ = self.forward(obs)
        
        # Filter to legal actions
        legal = obs.get('legal_actions', None)
        cand_actions = list(range(4)) if legal is None else [int(a) for a in legal]
        
        if training:
            # Softmax + sampling
            probs = self._softmax(logits)
            # Zero out illegal actions
            probs_legal = np.zeros(4, dtype=np.float32)
            for a in cand_actions:
                probs_legal[a] = probs[a]
            probs_legal /= probs_legal.sum()
            
            action = np.random.choice(4, p=probs_legal)
            return Action(action)
        else:
            # Greedy
            order = np.argsort(-logits)
            for idx in order:
                if int(idx) in cand_actions:
                    return Action(int(idx))
            return Action(int(order[0]))

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits)

    def train_on_rollout(self, rollout: list, lr: float = 1e-3, gamma: float = 0.99, entropy_beta: float = 0.01) -> Tuple[float, float]:
        """
        Train on a rollout trajectory using simple advantage updates.
        rollout: list of (obs, action, reward, next_obs, done)
        Returns: (actor_loss, critic_loss)
        """
        actor_loss = 0.0
        critic_loss = 0.0
        batch_size = len(rollout)
        
        # First, compute values and advantages
        values = []
        for obs, _, _, _, _ in rollout:
            _, v, _ = self.forward(obs)
            values.append(v)
        
        # Compute advantages (1-step TD)
        advantages = []
        returns = []
        for i in range(len(rollout)):
            obs, action, reward, next_obs, done = rollout[i]
            v_t = values[i]
            
            if done:
                target = reward
                advantage = reward - v_t
            else:
                _, v_next, _ = self.forward(next_obs)
                target = reward + gamma * v_next
                advantage = reward + gamma * v_next - v_t
            
            advantages.append(advantage)
            returns.append(target)
        
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = [(a - adv_mean) / adv_std for a in advantages]
        
        # Train on each step
        for i, (obs, action, reward, next_obs, done) in enumerate(rollout):
            feat = extended_features(obs)
            trunk_h = self._forward_trunk(feat)
            logits, actor_h = self._forward_actor(trunk_h)
            policy = self._softmax(logits)
            _, v, _ = self.forward(obs)
            
            advantage = advantages[i]
            target = returns[i]
            
            # Actor loss: policy gradient
            log_prob = np.log(policy[int(action)] + 1e-8)
            entropy = -np.sum(policy * np.log(policy + 1e-8))
            actor_step_loss = -(log_prob * advantage + entropy_beta * entropy)
            actor_loss += actor_step_loss
            
            # Critic loss
            critic_step_loss = (target - v) ** 2
            critic_loss += critic_step_loss
            
            # === Backprop through policy ===
            dlogits = policy.copy()
            dlogits[int(action)] -= 1.0
            dlogits = dlogits * advantage - entropy_beta * (np.log(policy + 1e-8) + 1.0)
            
            # Actor output layer gradient
            grad_actor_out = np.outer(dlogits, actor_h)
            d_actor_h = self.W_actor_out.T.dot(dlogits)
            
            # Actor hidden layer
            dz_actor = d_actor_h * (1.0 - actor_h ** 2)
            grad_actor = np.outer(dz_actor, trunk_h)
            
            # Trunk gradient from actor
            d_trunk_actor = self.W_actor.T.dot(dz_actor)
            
            # === Backprop through value ===
            dv = 2.0 * (v - target)
            
            # Critic output layer (W_critic_out shape: (1, 32))
            critic_h = np.tanh(self.W_critic.dot(trunk_h) + self.b_critic)
            # dv is scalar, critic_h is (32,), W_critic_out is (1, 32)
            # grad = dv * outer(1, critic_h) = dv * [[critic_h]]  (shape: 1 x 32, but we want to accumulate)
            grad_critic_out_accum = dv * critic_h[np.newaxis, :]  # (1, 32)
            
            # Critic hidden layer
            d_critic_h = dv * self.W_critic_out.flatten()  # (32,)
            dz_critic = d_critic_h * (1.0 - critic_h ** 2)
            grad_critic = np.outer(dz_critic, trunk_h)
            
            # Trunk gradient from critic
            d_trunk_critic = self.W_critic.T.dot(dz_critic)
            
            # === Combined trunk update ===
            d_trunk = d_trunk_actor + d_trunk_critic
            dz_trunk = d_trunk * (1.0 - trunk_h ** 2)
            grad_trunk = np.outer(dz_trunk, feat)
            
            # === Update weights ===
            self.W_actor_out -= (lr / batch_size) * grad_actor_out
            self.b_actor_out -= (lr / batch_size) * dlogits
            self.W_actor -= (lr / batch_size) * grad_actor
            self.b_actor -= (lr / batch_size) * dz_actor
            
            self.W_critic_out -= (lr / batch_size) * grad_critic_out_accum
            self.b_critic_out -= (lr / batch_size) * dv
            self.W_critic -= (lr / batch_size) * grad_critic
            self.b_critic -= (lr / batch_size) * dz_critic
            
            self.W_trunk -= (lr / batch_size) * grad_trunk
            self.b_trunk -= (lr / batch_size) * dz_trunk
        
        return actor_loss / batch_size, critic_loss / batch_size
