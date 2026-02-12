from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from collections import deque
from tqdm import trange

from env.snake_env import SnakeEnv, Action
from agents.dqn_extended_v2_agent import DQNExtendedV2Agent


def run_episode(env: SnakeEnv, agent: DQNExtendedV2Agent, epsilon: float, max_steps: int = 10000) -> Tuple[float, int, bool, List]:
    """Run one episode and collect trajectory."""
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    traj = []
    
    while steps < max_steps:
        obs['legal_actions'] = env.legal_actions()
        a = agent.act(obs, epsilon=epsilon)
        res = env.step(a)
        next_obs = res.obs
        
        # Reward shaping: same as baseline for fair comparison
        base_reward = float(res.reward if hasattr(res, 'reward') else 0.0)
        
        try:
            prev_dist = abs(obs['snake'][-1][0] - obs['food'][0]) + abs(obs['snake'][-1][1] - obs['food'][1])
            next_dist = abs(next_obs['snake'][-1][0] - next_obs['food'][0]) + abs(next_obs['snake'][-1][1] - next_obs['food'][1])
        except Exception:
            prev_dist = next_dist = 0.0
        
        # Enhanced reward: food is primary, approach is strong, survival bonus
        shaped = base_reward  # +1 for food
        shaped += 0.5 * (prev_dist - next_dist)  # approach bonus
        shaped += 0.02  # survival bonus (strongly encourage not dying)
        
        # Penalty for moving away
        if not res.terminated and not res.truncated and next_dist > prev_dist:
            shaped -= 0.1
        
        traj.append((obs, int(a), float(shaped), next_obs, bool(res.terminated or res.truncated)))
        total_reward += shaped
        steps += 1
        obs = next_obs
        
        if res.terminated or res.truncated:
            break

    return total_reward, steps, res.truncated, traj


def evaluate_agent(env: SnakeEnv, agent: DQNExtendedV2Agent, num_episodes: int = 5, seed: int = 0) -> dict:
    """Evaluate agent deterministically (no epsilon)."""
    rewards = []
    for ep_seed in range(seed, seed + num_episodes):
        env_reset_seed = ep_seed
        obs, _ = env.reset()
        obs['seed'] = env_reset_seed
        total_reward = 0.0
        while True:
            obs['legal_actions'] = env.legal_actions()
            a = agent.act(obs, epsilon=0.0)
            res = env.step(a)
            obs = res.obs
            base_reward = res.reward if hasattr(res, 'reward') else 0.0
            total_reward += base_reward
            if res.terminated or res.truncated:
                break
        rewards.append(total_reward)
    
    return {
        'mean': float(np.mean(rewards)),
        'median': float(np.median(rewards)),
        'max': float(np.max(rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description='DQN Extended v2 trainer (21 features with safety)')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--eval_every', type=int, default=200, help='Evaluation frequency')
    parser.add_argument('--eval_seeds', type=int, default=5, help='Evaluation episodes per eval')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--epsilon_start', type=float, default=0.8, help='Initial epsilon')
    parser.add_argument('--seed0', type=int, default=0, help='Seed for eval')
    
    args = parser.parse_args()
    
    env = SnakeEnv()
    agent = DQNExtendedV2Agent()
    target_agent = DQNExtendedV2Agent(W1=agent.W1.copy(), b1=agent.b1.copy(), 
                                      W2=agent.W2.copy(), b2=agent.b2.copy())
    
    print(f"Starting DQN-Extended-V2 training for {args.episodes} episodes")
    print(f"LR={args.lr}, epsilon_start={args.epsilon_start}, features=21 (with safety checks)")
    
    buffer = deque(maxlen=50000)
    best_eval_score = -float('inf')
    warmup_steps = 500
    target_update_steps = 2000
    total_steps = 0
    
    pbar = trange(args.episodes, desc='DQN-Ext-V2 Episodes')
    for ep in pbar:
        epsilon = args.epsilon_start * (0.99 ** (ep / args.episodes))
        
        train_reward, ep_steps, truncated, traj = run_episode(env, agent, epsilon)
        buffer.extend(traj)
        total_steps += ep_steps
        
        if total_steps > warmup_steps and len(buffer) > 64:
            for _ in range(5):
                batch_idx = np.random.choice(len(buffer), 64, replace=False)
                batch = [buffer[i] for i in batch_idx]
                agent.train_on_batch(batch, lr=args.lr, gamma=0.99, target_net=target_agent)
        
        if total_steps % target_update_steps == 0:
            target_agent.W1 = agent.W1.copy()
            target_agent.b1 = agent.b1.copy()
            target_agent.W2 = agent.W2.copy()
            target_agent.b2 = agent.b2.copy()
        
        if (ep + 1) % args.eval_every == 0:
            eval_result = evaluate_agent(env, agent, num_episodes=args.eval_seeds, seed=args.seed0)
            eval_score = eval_result['mean']
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                os.makedirs('models', exist_ok=True)
                agent.save('models/dqn_extended_v2_best.npz')
            
            pbar.write(f"Ep {ep+1:5d}  train_reward={train_reward:7.2f}  eval_score={eval_score:7.2f}  best={best_eval_score:7.2f}  eps={epsilon:.4f}")


if __name__ == '__main__':
    main()
