from __future__ import annotations

import argparse
import pygame

from env.snake_env import SnakeEnv, Action
from agents.astar_agent import AStarAgent

CELL = 28
MARGIN = 1
FPS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = SnakeEnv(16, 16, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    agent = AStarAgent()

    pygame.init()
    screen = pygame.display.set_mode((env.width * CELL, env.height * CELL))
    pygame.display.set_caption("Snake (Watch Agent)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
            img2 = font.render(msg, True, (0, 0, 0))
            screen.blit(img2, (6, 28))
            pygame.display.flip()
            pygame.time.wait(1200)
            obs, _ = env.reset()

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
