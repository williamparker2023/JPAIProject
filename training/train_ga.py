from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from tqdm import trange

from env.snake_env import SnakeEnv, Action
from agents.ga_agent import GAAgent

# Simple GA trainer producing a best-genome file in models/
def run_episode(env: SnakeEnv, policy, seed: int) -> Tuple[int, int]:
    obs, _ = env.reset(seed=seed)
    # ensure legal actions are accessible in obs for GAAgent
    obs["legal_actions"] = env.legal_actions()
    while True:
        a = policy(obs)
        res = env.step(a)
        obs = res.obs
        if res.terminated or res.truncated:
            return obs["score"], obs["total_steps"]


def evaluate_genome(env: SnakeEnv, genome: np.ndarray, seeds: List[int]) -> Tuple[float, int]:
    agent = GAAgent(genome)
    total_score = 0.0
    total_steps = 0
    for s in seeds:
        score, steps = run_episode(env, agent.act, seed=s)
        total_score += score
        total_steps += steps
    avg_score = total_score / len(seeds)
    avg_steps = total_steps / len(seeds)
    # fitness: primary = score, small bonus for survival (avg steps)
    fitness = avg_score + 0.001 * avg_steps
    return fitness, total_steps


def tournament_select(pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator, k: int) -> int:
    inds = rng.integers(0, len(pop), size=k)
    best = inds[0]
    for i in inds[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop_size", type=int, default=100)
    ap.add_argument("--elitism", type=int, default=2)
    ap.add_argument("--tournament_k", type=int, default=3)
    ap.add_argument("--mut_std", type=float, default=0.1)
    ap.add_argument("--episodes_per_genome", type=int, default=3)  # fixed-seed evals per genome
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="models")
    ap.add_argument("--hidden", type=int, default=16)  # kept for potential compat
    ap.add_argument("--stall_multiplier", type=int, default=100, help="multiplier for stall truncation (100*len)")
    ap.add_argument("--hard_cap_multiplier", type=int, default=50, help="multiplier for hard cap (grid_area * X)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # set GAAgent hidden constant to args.hidden so genome shape matches
    GAAgent.HIDDEN = args.hidden
    env = SnakeEnv(16, 16, stall_multiplier=args.stall_multiplier, hard_cap_multiplier=args.hard_cap_multiplier)
    rng = np.random.default_rng(args.seed0)

    genome_len = GAAgent.genome_size()
    pop = np.array([GAAgent.random_genome(rng) for _ in range(args.pop_size)])
    fitness = np.zeros(len(pop), dtype=np.float32)

    # fixed evaluation seeds (same every generation)
    eval_seeds = [args.seed0 + i for i in range(args.episodes_per_genome)]

    cum_steps = 0
    for gen in trange(args.generations, desc="GA gens"):
        # evaluate
        gen_steps = 0
        for i in range(len(pop)):
            fit, steps = evaluate_genome(env, pop[i], eval_seeds)
            fitness[i] = fit
            gen_steps += steps
        cum_steps += gen_steps

        # report
        best_idx = int(np.argmax(fitness))
        best_fit = float(fitness[best_idx])
        mean_fit = float(np.mean(fitness))
        print(f"Gen {gen:3d}  best={best_fit:.3f}  mean={mean_fit:.3f}  cum_steps={cum_steps}")

        # save best genome
        best_path = os.path.join(args.save_dir, "ga_best.npy")
        np.save(best_path, pop[best_idx])
        meta = {
            "generation": gen,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "cum_steps": cum_steps,
            "genome_size": genome_len,
        }
        with open(os.path.join(args.save_dir, "ga_best_meta.json"), "w") as f:
            json.dump(meta, f)

        # create next generation
        new_pop = []
        # elitism
        elite_inds = np.argsort(-fitness)[: args.elitism]
        for ei in elite_inds:
            new_pop.append(pop[ei].copy())

        while len(new_pop) < len(pop):
            # selection
            a_idx = tournament_select(pop, fitness, rng, args.tournament_k)
            b_idx = tournament_select(pop, fitness, rng, args.tournament_k)
            parent_a = pop[a_idx]
            parent_b = pop[b_idx]
            # crossover: simple uniform average with 50% chance
            if rng.random() < 0.5:
                child = 0.5 * parent_a + 0.5 * parent_b
            else:
                # clone from one parent
                child = parent_a.copy() if rng.random() < 0.5 else parent_b.copy()
            # mutation
            child += rng.normal(0, args.mut_std, size=child.shape)
            new_pop.append(child.astype(np.float32))

        pop = np.array(new_pop)

    print("Training complete. Best genome saved to:", best_path)


if __name__ == "__main__":
    main()