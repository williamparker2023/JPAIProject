from __future__ import annotations

import argparse
import numpy as np
import pygame
from typing import Optional

from env.snake_env import SnakeEnv
from agents.dqn_baseline_agent import DQNBaselineAgent


CELL = 28
MARGIN = 1
FPS = 20


def run_render(agent: DQNBaselineAgent, seed: int = 0):
    """Render agent gameplay with pygame."""
    env = SnakeEnv()
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

        obs['legal_actions'] = env.legal_actions()
        a = agent.act(obs, epsilon=0.0)
        res = env.step(a)
        obs = res.obs

        # Draw
        screen.fill((245, 245, 245))
        fx, fy = obs['food']
        pygame.draw.rect(screen, (220, 60, 60), (fx * CELL + MARGIN, fy * CELL + MARGIN, CELL - 2*MARGIN, CELL - 2*MARGIN))
        
        snake = obs['snake']
        for i, (x, y) in enumerate(snake):
            color = (60, 120, 220) if i == len(snake) - 1 else (80, 160, 80)
            pygame.draw.rect(screen, color, (x * CELL + MARGIN, y * CELL + MARGIN, CELL - 2*MARGIN, CELL - 2*MARGIN))
        
        text = f"Score: {obs['score']}   Len: {len(snake)}"
        img = font.render(text, True, (20, 20, 20))
        screen.blit(img, (6, 6))

        if res.terminated or res.truncated:
            msg = "WIN!" if res.info.get("win") else ("TRUNC" if res.truncated else f"DEAD: {res.info.get('death','?')}")
            img2 = font.render(msg, True, (0, 0, 0))
            screen.blit(img2, (6, 28))
            pygame.display.flip()
            pygame.time.wait(900)
            obs, _ = env.reset()

        pygame.display.flip()
    
    pygame.quit()


def run_eval(agent: DQNBaselineAgent, num_episodes: int = 20, seeds: list = None):
    """Headless evaluation."""
    if seeds is None:
        seeds = list(range(num_episodes))
    
    env = SnakeEnv()
    rewards = []
    
    for seed in seeds:
        obs, _ = env.reset()
        obs['seed'] = seed
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
    
    print(f"Mean: {np.mean(rewards):.2f}")
    print(f"Median: {np.median(rewards):.2f}")
    print(f"Max: {np.max(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f}")
    print(f"All: {rewards}")


def main():
    parser = argparse.ArgumentParser(description='Watch DQN-Baseline Agent')
    parser.add_argument('--model', type=str, default='models/dqn_baseline_best.npz', help='Model path')
    parser.add_argument('--seed', type=int, default=0, help='Seed for gameplay')
    parser.add_argument('--headless', action='store_true', help='No pygame visualization')
    parser.add_argument('--eval_episodes', type=int, default=20, help='Number of eval episodes')
    
    args = parser.parse_args()
    
    agent = DQNBaselineAgent.load(args.model)
    
    if args.headless:
        run_eval(agent, num_episodes=args.eval_episodes, seeds=list(range(args.eval_episodes)))
    else:
        run_render(agent, seed=args.seed)


if __name__ == '__main__':
    main()
