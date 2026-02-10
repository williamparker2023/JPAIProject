from __future__ import annotations

import argparse
import pygame
import numpy as np
from env.snake_env import SnakeEnv, Action
from agents.ga_agent import GAAgent

CELL = 28
MARGIN = 1
FPS = 20

def run_render(env: SnakeEnv, agent: GAAgent, seed: int):
    obs, _ = env.reset(seed=seed)
    pygame.init()
    screen = pygame.display.set_mode((env.width * CELL, env.height * CELL))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ensure GAAgent can see legal actions
        obs["legal_actions"] = env.legal_actions()
        action = agent.act(obs)
        res = env.step(action)
        obs = res.obs

        # draw
        screen.fill((245, 245, 245))
        fx, fy = obs["food"]
        pygame.draw.rect(screen, (220, 60, 60), (fx * CELL + MARGIN, fy * CELL + MARGIN, CELL - 2*MARGIN, CELL - 2*MARGIN))
        snake = obs["snake"]
        for i, (x, y) in enumerate(snake):
            color = (60, 120, 220) if i == len(snake) - 1 else (80, 160, 80)
            pygame.draw.rect(screen, color, (x * CELL + MARGIN, y * CELL + MARGIN, CELL - 2*MARGIN, CELL - 2*MARGIN))
        text = f"Score: {obs['score']}   Len: {len(snake)}"
        img = font.render(text, True, (20, 20, 20))
        screen.blit(img, (6, 6))

        if res.terminated or res.truncated:
            msg = "WIN!" if res.info.get("win") else ("TRUNC" if res.truncated else f"DEAD: {res.info.get('death','?')}")
            img2 = font.render(msg, True, (0, 0, 0)); screen.blit(img2, (6, 28))
            pygame.display.flip()
            pygame.time.wait(900)
            obs, _ = env.reset()

        pygame.display.flip()
    pygame.quit()

def run_eval(env: SnakeEnv, agent: GAAgent, seeds: list[int]):
    results = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        obs["legal_actions"] = env.legal_actions()
        while True:
            a = agent.act(obs)
            res = env.step(a)
            obs = res.obs
            if res.terminated or res.truncated:
                results.append({"score": obs["score"], "steps": obs["total_steps"], "info": res.info})
                break
    import statistics, json
    scores = [r["score"] for r in results]
    out = {"mean": statistics.mean(scores), "median": statistics.median(scores), "max": max(scores), "runs": results}
    print(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/ga_best.npy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--stall_multiplier", type=int, default=30, help="multiplier for stall truncation (N*len) - prevents infinite loops")
    ap.add_argument("--hard_cap_multiplier", type=int, default=20, help="multiplier for hard cap (grid_area * N)")
    args = ap.parse_args()

    genome = np.load(args.model)
    agent = GAAgent(genome)
    env = SnakeEnv(16, 16, seed=args.seed, stall_multiplier=args.stall_multiplier, hard_cap_multiplier=args.hard_cap_multiplier)

    if args.headless:
        seeds = [args.seed + i for i in range(args.eval_episodes)]
        run_eval(env, agent, seeds)
    else:
        run_render(env, agent, seed=args.seed)

if __name__ == "__main__":
    main()