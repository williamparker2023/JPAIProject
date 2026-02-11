from __future__ import annotations

import argparse
import os
import numpy as np
from typing import List, Tuple
from collections import deque
import json
from tqdm import trange

from env.snake_env import SnakeEnv, Action
from agents.rl_agent import RLAgent


def run_episode(env: SnakeEnv, agent: RLAgent, epsilon: float, max_steps: int = 10000) -> Tuple[float, int, bool, List]:
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    traj = []
    while True:
        obs['legal_actions'] = env.legal_actions()
        a = agent.act(obs, epsilon=epsilon)
        res = env.step(a)
        next_obs = res.obs
        base_reward = res.reward if hasattr(res, 'reward') else 0.0

        # Reward shaping: balanced guidance
        try:
            head = obs['snake'][-1]
            food = obs['food']
            prev_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
            nhead = next_obs['snake'][-1]
            nfood = next_obs['food']
            next_dist = abs(nhead[0] - nfood[0]) + abs(nhead[1] - nfood[1])
        except Exception:
            prev_dist = next_dist = 0

        shaped = float(base_reward)
        shaped += 0.2 * (prev_dist - next_dist)  # moderate bonus for approaching food
        shaped -= 0.005  # tiny step penalty
        
        traj.append((obs, int(a), float(shaped), next_obs, bool(res.terminated or res.truncated)))
        total_reward += shaped
        steps += 1
        obs = next_obs
        if res.terminated or res.truncated or steps >= max_steps:
            break

    return total_reward, steps, res.truncated, traj


def evaluate_agent(env: SnakeEnv, agent: RLAgent, seeds: List[int]) -> float:
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
    ap.add_argument('--episodes', type=int, default=5000)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--eval_seeds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--epsilon_start', type=float, default=0.7)
    ap.add_argument('--epsilon_end', type=float, default=0.05)
    ap.add_argument('--epsilon_decay', type=float, default=0.9995)
    ap.add_argument('--save_dir', type=str, default='models')
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--buffer_size', type=int, default=50000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--target_update_steps', type=int, default=1000)
    ap.add_argument('--stall_multiplier', type=int, default=50)
    ap.add_argument('--hard_cap_multiplier', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = SnakeEnv(16, 16, seed=args.seed0, stall_multiplier=args.stall_multiplier, hard_cap_multiplier=args.hard_cap_multiplier)
    rng = np.random.default_rng(args.seed0)

    agent = RLAgent()
    target_net = RLAgent(W1=agent.W1.copy(), b1=agent.b1.copy(), W2=agent.W2.copy(), b2=agent.b2.copy())

    replay = deque(maxlen=args.buffer_size)

    best_score = -1.0
    epsilon = args.epsilon_start
    steps_total = 0
    steps_since_target_update = 0

    for ep in trange(args.episodes, desc='RL eps'):
        # run episode
        total_reward, steps, truncated, traj = run_episode(env, agent, epsilon)

        # push transitions into replay
        for t in traj:
            replay.append(t)

        steps_total += steps
        steps_since_target_update += steps

        # decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        # moderate training: 4 batches per episode (not 10)
        if steps_total >= args.warmup and len(replay) >= args.batch_size:
            for _ in range(4):
                batch_inds = rng.choice(len(replay), size=args.batch_size, replace=False)
                batch = [replay[i] for i in batch_inds]
                agent.train_on_batch(batch, lr=args.lr, gamma=args.gamma, target_net=target_net)

        # sync target network
        if steps_since_target_update >= args.target_update_steps:
            target_net.W1 = agent.W1.copy()
            target_net.b1 = agent.b1.copy()
            target_net.W2 = agent.W2.copy()
            target_net.b2 = agent.b2.copy()
            steps_since_target_update = 0

        # evaluate
        if (ep + 1) % args.eval_every == 0:
            eval_seeds = [args.seed0 + 2000 + i for i in range(args.eval_seeds)]
            mean_score = evaluate_agent(env, agent, eval_seeds)
            print(f"Ep {ep+1:5d}  score={mean_score:6.2f}  best={best_score:6.2f}  eps={epsilon:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                agent.save(os.path.join(args.save_dir, 'rl_best.npz'))

    print('Training complete. Best saved to:', os.path.join(args.save_dir, 'rl_best.npz'))


if __name__ == '__main__':
    main()