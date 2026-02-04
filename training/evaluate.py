from __future__ import annotations

import argparse
import statistics
from typing import Callable, Dict, List

import numpy as np

from env.snake_env import SnakeEnv, Action
from env.features import compact_features
from agents.astar_agent import AStarAgent


def run_episode(env: SnakeEnv, policy: Callable[[Dict], Action], seed: int) -> Dict:
    obs, _ = env.reset(seed=seed)
    while True:
        a = policy(obs)
        res = env.step(a)
        obs = res.obs
        if res.terminated or res.truncated:
            return {
                "score": obs["score"],
                "length": len(obs["snake"]),
                "terminated": res.terminated,
                "truncated": res.truncated,
                "info": res.info,
                "total_steps": obs["total_steps"],
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["astar"], default="astar")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    env = SnakeEnv(16, 16)

    if args.agent == "astar":
        agent = AStarAgent()
        policy = lambda obs: agent.act(obs)

    results = []
    for i in range(args.episodes):
        seed = args.seed0 + i
        results.append(run_episode(env, policy, seed))

    scores = [r["score"] for r in results]
    print(f"Agent: {args.agent}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean score:   {statistics.mean(scores):.3f}")
    print(f"Median score: {statistics.median(scores):.3f}")
    print(f"Max score:    {max(scores)}")

    deaths = {}
    for r in results:
        d = r["info"].get("death", "none")
        deaths[d] = deaths.get(d, 0) + 1
    print("Deaths:", deaths)


if __name__ == "__main__":
    main()
