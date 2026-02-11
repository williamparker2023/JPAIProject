from __future__ import annotations

import argparse
import os
import numpy as np
from typing import List
import pickle
import json
from tqdm import trange

from agents.rl_agent import RLAgent
from env.snake_env import SnakeEnv, Action
from env.features import compact_features


def load_trajectories(path: str) -> List:
    with open(path, 'rb') as f:
        return pickle.load(f)


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
    ap.add_argument('--trajectories', type=str, default='models/astar_trajectories.pkl')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--save_dir', type=str, default='models')
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--eval_seeds', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load trajectories
    print("Loading trajectories...")
    traj = load_trajectories(args.trajectories)
    print(f"Loaded {len(traj)} transitions")
    
    # Convert to numpy arrays
    features = np.array([t['features'] for t in traj], dtype=np.float32)
    actions = np.array([t['action'] for t in traj], dtype=np.int32)
    
    # Create agent
    agent = RLAgent()
    env = SnakeEnv(16, 16)
    rng = np.random.default_rng(args.seed0)
    
    best_score = -1.0
    num_transitions = len(traj)
    
    # Training loop
    for epoch in trange(args.epochs, desc="Imitation Learning"):
        # Shuffle and create batches
        inds = rng.permutation(num_transitions)
        
        for batch_start in range(0, num_transitions, args.batch_size):
            batch_end = min(batch_start + args.batch_size, num_transitions)
            batch_inds = inds[batch_start:batch_end]
            
            batch_feats = features[batch_inds]
            batch_acts = actions[batch_inds]
            
            # Supervised learning: MSE on logits
            total_loss = 0.0
            for feat, act in zip(batch_feats, batch_acts):
                q_pred, _, _ = agent._forward_feat(feat)
                
                # Target: action should have logit = 1, others = 0
                q_target = np.zeros(4, dtype=np.float32)
                q_target[int(act)] = 1.0
                
                # MSE loss
                error = q_pred - q_target
                loss = float(np.sum(error ** 2))
                total_loss += loss
                
                # Gradient: dL/dq = 2 * error
                dq = 2.0 * error
                
                # Backprop (same as DQN train_on_batch)
                _, h, z1 = agent._forward_feat(feat)
                grad_W2 = np.outer(dq, h)
                grad_b2 = dq
                dh = agent.W2.T.dot(dq)
                dz1 = dh * (1.0 - np.tanh(z1) ** 2)
                grad_W1 = np.outer(dz1, feat)
                grad_b1 = dz1
                
                agent.W1 -= args.lr * grad_W1
                agent.b1 -= args.lr * grad_b1
                agent.W2 -= args.lr * grad_W2
                agent.b2 -= args.lr * grad_b2
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            eval_seeds = [args.seed0 + 1000 + i for i in range(args.eval_seeds)]
            mean_score = evaluate_agent(env, agent, eval_seeds)
            print(f"Epoch {epoch+1:3d}  eval_score={mean_score:6.2f}  best={best_score:6.2f}")
            
            if mean_score > best_score:
                best_score = mean_score
                agent.save(os.path.join(args.save_dir, 'rl_best.npz'))
                with open(os.path.join(args.save_dir, 'imitation_meta.json'), 'w') as f:
                    json.dump({"epoch": epoch + 1, "best_score": best_score}, f)

    print(f'Training complete. Best score: {best_score:.2f}')
    print(f'Saved to {os.path.join(args.save_dir, "rl_best.npz")}')


if __name__ == '__main__':
    main()