from __future__ import annotations

import argparse
import os
import numpy as np
from typing import List, Dict
from collections import deque
import pickle
from tqdm import trange

from env.snake_env import SnakeEnv, Action
from agents.astar_agent import AStarAgent
from env.features import compact_features


def collect_trajectories(env: SnakeEnv, agent: AStarAgent, num_episodes: int, seed0: int) -> List[Dict]:
    """Collect (observation, action) pairs from A* playing."""
    trajectories = []
    
    for ep in trange(num_episodes, desc="Collecting A* data"):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            obs['legal_actions'] = env.legal_actions()
            action = agent.act(obs)
            
            # Store feature vector and action
            feat = compact_features(obs)
            trajectories.append({
                'features': feat,
                'action': int(action),
            })
            
            res = env.step(action)
            obs = res.obs
            if res.terminated or res.truncated:
                break
    
    return trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--save_dir', type=str, default='models')
    ap.add_argument('--stall_multiplier', type=int, default=50)
    ap.add_argument('--hard_cap_multiplier', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = SnakeEnv(16, 16, seed=args.seed0, stall_multiplier=args.stall_multiplier, hard_cap_multiplier=args.hard_cap_multiplier)
    agent = AStarAgent()
    
    trajectories = collect_trajectories(env, agent, args.episodes, args.seed0)
    
    save_path = os.path.join(args.save_dir, 'astar_trajectories.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"Collected {len(trajectories)} (obs, action) pairs")
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()