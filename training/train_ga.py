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
def run_episode(env: SnakeEnv, policy, seed: int) -> Tuple[int, int, bool]:
    obs, _ = env.reset(seed=seed)
    # ensure legal actions are accessible in obs for GAAgent
    obs["legal_actions"] = env.legal_actions()
    while True:
        a = policy(obs)
        res = env.step(a)
        obs = res.obs
        if res.terminated or res.truncated:
            return obs["score"], obs["total_steps"], res.truncated


def evaluate_genome(env: SnakeEnv, genome: np.ndarray, seeds: List[int]) -> Tuple[float, int]:
    agent = GAAgent(genome)
    total_score = 0.0
    total_steps = 0
    truncation_count = 0
    for s in seeds:
        score, steps, truncated = run_episode(env, agent.act, seed=s)
        total_score += score
        total_steps += steps
        if truncated:
            truncation_count += 1
    avg_score = total_score / len(seeds)
    avg_steps = total_steps / len(seeds)
    # fitness: primary = score, bonus for survival, penalty for truncation (circling)
    fitness = avg_score + 0.001 * avg_steps - (0.5 * truncation_count)
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
    ap.add_argument("--stall_multiplier", type=int, default=50, help="multiplier for stall truncation (N*len) - lower = stricter")
    ap.add_argument("--hard_cap_multiplier", type=int, default=30, help="multiplier for hard cap (grid_area * N) - lower = stricter")
    ap.add_argument("--init_scale", type=float, default=0.5, help="initial weight scale (smaller = more conservative start)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # set GAAgent hidden constant to args.hidden so genome shape matches
    GAAgent.HIDDEN = args.hidden
    env = SnakeEnv(16, 16, stall_multiplier=args.stall_multiplier, hard_cap_multiplier=args.hard_cap_multiplier)
    rng = np.random.default_rng(args.seed0)

    genome_len = GAAgent.genome_size()
    pop = np.array([GAAgent.random_genome(rng, scale=args.init_scale) for _ in range(args.pop_size)])
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
            
            # multi-point crossover: split genome into random chunks
            child = np.empty_like(parent_a)
            chunk_size = max(1, len(parent_a) // rng.integers(2, 5))  # 2-4 crossover points
            crossover_points = sorted(rng.choice(len(parent_a), size=rng.integers(2, 4), replace=False))
            last = 0
            use_a = rng.random() < 0.5
            for cp in crossover_points:
                if use_a:
                    child[last:cp] = parent_a[last:cp]
                else:
                    child[last:cp] = parent_b[last:cp]
                use_a = not use_a
                last = cp
            # fill remainder
            if use_a:
                child[last:] = parent_a[last:]
            else:
                child[last:] = parent_b[last:]
            
            # adaptive mutation: higher early, lower late
            progress = max(0, (gen - 20) / (args.generations - 20))  # 0 to 1 over last 80% of training
            mut_scale = args.mut_std * (1.0 - 0.5 * progress)  # linearly decay from mut_std to 0.5*mut_std
            child += rng.normal(0, mut_scale, size=child.shape)
            new_pop.append(child.astype(np.float32))

        pop = np.array(new_pop)

    print("Training complete. Best genome saved to:", best_path)


if __name__ == "__main__":
    main()