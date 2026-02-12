from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from collections import deque
from tqdm import trange

from env.snake_env import SnakeEnv, Action
from agents.dqn_extended_agent import DQNExtendedAgent
from env.features_extended import extended_features


def run_episode(env: SnakeEnv, agent: DQNExtendedAgent, epsilon: float, max_steps: int = 10000) -> Tuple[float, int, bool, List]:
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
        
        # Reward shaping: strong food reward + approach bonus + survival bonus
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


def evaluate_agent(env: SnakeEnv, agent: DQNExtendedAgent, seeds: List[int]) -> float:
    """Evaluate agent on fixed seeds."""
    scores = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        obs['legal_actions'] = env.legal_actions()
        steps = 0
        while steps < 10000:
            a = agent.act(obs, epsilon=0.0)
            res = env.step(a)
            obs = res.obs
            steps += 1
            if res.terminated or res.truncated:
                break
        scores.append(obs['score'])
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=5000, help='Total episodes')
    ap.add_argument('--eval_every', type=int, default=200, help='Evaluate every N episodes')
    ap.add_argument('--eval_seeds', type=int, default=5, help='Seeds per eval')
    ap.add_argument('--lr', type=float, default=2e-3, help='Learning rate (higher for extended features)')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    ap.add_argument('--epsilon_start', type=float, default=0.8, help='Initial epsilon')
    ap.add_argument('--epsilon_end', type=float, default=0.05, help='Final epsilon')
    ap.add_argument('--epsilon_decay', type=float, default=0.9995, help='Epsilon decay per episode')
    ap.add_argument('--save_dir', type=str, default='models')
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--buffer_size', type=int, default=50000, help='Replay buffer size')
    ap.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    ap.add_argument('--warmup', type=int, default=500, help='Steps before training starts')
    ap.add_argument('--target_update_steps', type=int, default=2000, help='Update target network every N steps')
    ap.add_argument('--stall_multiplier', type=int, default=50)
    ap.add_argument('--hard_cap_multiplier', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed0)
    
    env = SnakeEnv(16, 16, seed=args.seed0, 
                   stall_multiplier=args.stall_multiplier, 
                   hard_cap_multiplier=args.hard_cap_multiplier)
    
    agent = DQNExtendedAgent()
    target_net = DQNExtendedAgent(W1=agent.W1.copy(), b1=agent.b1.copy(), 
                                  W2=agent.W2.copy(), b2=agent.b2.copy())

    replay = deque(maxlen=args.buffer_size)

    best_score = -1.0
    best_episode = 0
    epsilon = args.epsilon_start
    steps_total = 0
    steps_since_target_update = 0
    training_history = []

    print(f"Starting DQN-Extended training for {args.episodes} episodes")
    print(f"LR={args.lr}, epsilon_start={args.epsilon_start}, buffer_size={args.buffer_size}")
    print(f"Features: 21 (vs 11 in standard DQN)")
    
    for ep in trange(args.episodes, desc='DQN-Ext Episodes'):
        # Run episode
        total_reward, steps, truncated, traj = run_episode(env, agent, epsilon)

        # Push to replay buffer
        for t in traj:
            replay.append(t)

        steps_total += steps
        steps_since_target_update += steps

        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # Train: 4-6 batches per episode
        if steps_total >= args.warmup and len(replay) >= args.batch_size:
            for _ in range(5):
                batch_idx = np.random.choice(len(replay), size=args.batch_size, replace=False)
                batch = [replay[i] for i in batch_idx]
                agent.train_on_batch(batch, lr=args.lr, gamma=args.gamma, target_net=target_net)

        # Update target network
        if steps_since_target_update >= args.target_update_steps:
            target_net.W1 = agent.W1.copy()
            target_net.b1 = agent.b1.copy()
            target_net.W2 = agent.W2.copy()
            target_net.b2 = agent.b2.copy()
            steps_since_target_update = 0

        # Evaluate periodically
        if (ep + 1) % args.eval_every == 0:
            eval_seeds = [args.seed0 + 1000 + i for i in range(args.eval_seeds)]
            mean_score = evaluate_agent(env, agent, eval_seeds)
            training_history.append({
                'episode': ep + 1,
                'train_reward': total_reward,
                'train_steps': steps,
                'eval_score': mean_score,
                'epsilon': epsilon
            })
            
            print(f"Ep {ep+1:4d}  train_reward={total_reward:7.2f}  eval_score={mean_score:6.2f}  best={best_score:6.2f}  eps={epsilon:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_episode = ep + 1
                best_path = os.path.join(args.save_dir, 'dqn_extended_best.npz')
                agent.save(best_path)
                
                meta = {
                    'episode': ep + 1,
                    'best_score': best_score,
                    'eval_seeds': args.eval_seeds,
                    'features': 21,
                    'hidden': 128,
                }
                meta_path = os.path.join(args.save_dir, 'dqn_extended_best_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)

    print("\n" + "="*60)
    print(f"DQN-Extended Training Complete")
    print(f"Best score: {best_score:.2f} at episode {best_episode}")
    print(f"Model saved to: {os.path.join(args.save_dir, 'dqn_extended_best.npz')}")
    print("="*60)
    
    hist_path = os.path.join(args.save_dir, 'dqn_extended_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(training_history, f, indent=2)


if __name__ == '__main__':
    main()
