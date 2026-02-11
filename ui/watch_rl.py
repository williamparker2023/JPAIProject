from __future__ import annotations

import argparse
import pygame
import numpy as np
from env.snake_env import SnakeEnv, Action
from agents.rl_agent import RLAgent

CELL = 28
MARGIN = 1
FPS = 20


def run_render(env: SnakeEnv, agent: RLAgent, seed: int):
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

        # ensure agent can see legal actions
        obs['legal_actions'] = env.legal_actions()
        action = agent.act(obs, epsilon=0.0)
        res = env.step(action)
        obs = res.obs

        # draw
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
            msg = "WIN!" if res.info.get('win') else ("TRUNC" if res.truncated else f"DEAD: {res.info.get('death','?')}")
            img2 = font.render(msg, True, (0, 0, 0)); screen.blit(img2, (6, 28))
            pygame.display.flip()
            pygame.time.wait(900)
            obs, _ = env.reset()

        pygame.display.flip()
    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='models/rl_best.npz')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.model)
    # support new DQN format (W1,b1,W2,b2) and legacy linear (W,b)
    if 'W1' in data and 'W2' in data:
        agent = RLAgent(W1=data['W1'], b1=data['b1'], W2=data['W2'], b2=data['b2'])
    elif 'W' in data and 'b' in data:
        # legacy linear model: copy into output layer of small DQN
        agent = RLAgent()
        agent.W2 = data['W'].astype(np.float32)
        agent.b2 = data['b'].astype(np.float32)
    else:
        raise RuntimeError('Unsupported model format for RL agent: ' + args.model)
    env = SnakeEnv(16, 16, seed=args.seed)

    run_render(env, agent, seed=args.seed)


if __name__ == '__main__':
    main()
