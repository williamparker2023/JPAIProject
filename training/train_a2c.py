from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from tqdm import trange

from env.snake_env import SnakeEnv, Action
from agents.a2c_agent import A2CAgent


def run_episode(env: SnakeEnv, agent: A2CAgent, training: bool = True, max_steps: int = 10000) -> Tuple[float, int, bool, List]:
    """
    Run one episode and collect trajectory.
    Returns: (total_reward, steps, truncated, rollout)
    rollout: list of (obs, action, reward, next_obs, done)
    """
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    rollout = []
    
    while steps < max_steps:
        obs['legal_actions'] = env.legal_actions()
        
        # Act
        action = agent.act(obs, training=training)
        res = env.step(action)
        next_obs = res.obs
        
        # Reward shaping
        base_reward = float(res.reward if hasattr(res, 'reward') else 0.0)
        
        try:
            prev_dist = abs(obs['snake'][-1][0] - obs['food'][0]) + abs(obs['snake'][-1][1] - obs['food'][1])
            next_dist = abs(next_obs['snake'][-1][0] - next_obs['food'][0]) + abs(next_obs['snake'][-1][1] - next_obs['food'][1])
        except Exception:
            prev_dist = next_dist = 0.0
        
        done = bool(res.terminated or res.truncated)
        
        # Shaped reward: food is primary, then approach bonus, small survival bonus
        shaped = base_reward  # +1 for eating food
        shaped += 0.5 * (prev_dist - next_dist)  # stronger approach bonus
        shaped += 0.01  # tiny survival bonus each step (encourages not dying)
        
        # Only penalize if moving away from food significantly without eating
        if not done and next_dist > prev_dist:
            shaped -= 0.05  # penalty for moving away
        
        rollout.append((obs.copy(), int(action), float(shaped), next_obs.copy(), done))
        
        total_reward += shaped
        steps += 1
        obs = next_obs
        
        if done:
            break
    
    return total_reward, steps, res.truncated, rollout


def evaluate_agent(env: SnakeEnv, agent: A2CAgent, seeds: List[int]) -> float:
    """Evaluate agent on fixed seeds, no exploration."""
    scores = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        obs['legal_actions'] = env.legal_actions()
        steps = 0
        while steps < 10000:
            action = agent.act(obs, training=False)
            res = env.step(action)
            obs = res.obs
            steps += 1
            if res.terminated or res.truncated:
                break
        scores.append(obs['score'])
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3000, help='Total episodes to train')
    ap.add_argument('--eval_every', type=int, default=100, help='Evaluate every N episodes')
    ap.add_argument('--eval_seeds', type=int, default=5, help='Seeds per eval')
    ap.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3 for faster A2C learning)')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    ap.add_argument('--entropy_beta', type=float, default=0.05, help='Entropy regularization weight (higher = more exploration)')
    ap.add_argument('--save_dir', type=str, default='models', help='Directory to save model')
    ap.add_argument('--seed0', type=int, default=42, help='Random seed')
    ap.add_argument('--stall_multiplier', type=int, default=50)
    ap.add_argument('--hard_cap_multiplier', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize
    np.random.seed(args.seed0)
    env = SnakeEnv(16, 16, seed=args.seed0, 
                   stall_multiplier=args.stall_multiplier, 
                   hard_cap_multiplier=args.hard_cap_multiplier)
    agent = A2CAgent()
    
    best_score = -1.0
    best_episode = 0
    training_history = []
    
    print(f"Starting A2C training for {args.episodes} episodes")
    print(f"LR={args.lr}, entropy_beta={args.entropy_beta}, gamma={args.gamma}")
    
    for ep in trange(args.episodes, desc='A2C Episodes'):
        # Run episode
        total_reward, steps, truncated, rollout = run_episode(env, agent, training=True)
        
        # Train on rollout
        if rollout:
            actor_loss, critic_loss = agent.train_on_rollout(
                rollout,
                lr=args.lr,
                gamma=args.gamma,
                entropy_beta=args.entropy_beta
            )
        
        # Evaluate periodically
        if (ep + 1) % args.eval_every == 0:
            eval_seeds = [args.seed0 + 1000 + i for i in range(args.eval_seeds)]
            mean_score = evaluate_agent(env, agent, eval_seeds)
            training_history.append({
                'episode': ep + 1,
                'train_reward': total_reward,
                'train_steps': steps,
                'eval_score': mean_score
            })
            
            print(f"Ep {ep+1:4d}  train_reward={total_reward:7.2f}  eval_score={mean_score:6.2f}  best={best_score:6.2f}")
            
            # Save if best
            if mean_score > best_score:
                best_score = mean_score
                best_episode = ep + 1
                best_path = os.path.join(args.save_dir, 'a2c_best.npz')
                agent.save(best_path)
                
                # Save metadata
                meta = {
                    'episode': ep + 1,
                    'best_score': best_score,
                    'train_loss_actor': float(actor_loss),
                    'train_loss_critic': float(critic_loss),
                    'eval_seeds': int(args.eval_seeds),
                }
                meta_path = os.path.join(args.save_dir, 'a2c_best_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)

    # Final report
    print("\n" + "="*60)
    print(f"A2C Training Complete")
    print(f"Best score: {best_score:.2f} at episode {best_episode}")
    print(f"Model saved to: {os.path.join(args.save_dir, 'a2c_best.npz')}")
    print(f"Metadata saved to: {os.path.join(args.save_dir, 'a2c_best_meta.json')}")
    print("="*60)
    
    # Save training history
    hist_path = os.path.join(args.save_dir, 'a2c_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {hist_path}")


if __name__ == '__main__':
    main()
