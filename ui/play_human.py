from __future__ import annotations

import pygame
from env.snake_env import SnakeEnv, Action

CELL = 28
MARGIN = 1
FPS = 12


KEYMAP = {
    pygame.K_UP: Action.UP,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
}


def main():
    env = SnakeEnv(16, 16)
    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((env.width * CELL, env.height * CELL))
    pygame.display.set_caption("Snake (Human)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    pending_action = Action(obs["direction"])

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in KEYMAP:
                    pending_action = KEYMAP[event.key]
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()

        res = env.step(pending_action)
        obs = res.obs

        if res.terminated or res.truncated:
            # show end state briefly; press R to restart
            pass

        # draw
        screen.fill((245, 245, 245))

        # food
        fx, fy = obs["food"]
        pygame.draw.rect(screen, (220, 60, 60), (fx * CELL + MARGIN, fy * CELL + MARGIN, CELL - 2*MARGIN, CELL - 2*MARGIN))

        # snake
        snake = obs["snake"]
        for i, (x, y) in enumerate(snake):
            color = (60, 120, 220) if i == len(snake) - 1 else (80, 160, 80)
            pygame.draw.rect(screen, color, (x * CELL + MARGIN, y * CELL + MARGIN, CELL - 2*MARGIN, CELL - 2*MARGIN))

        text = f"Score: {obs['score']}   Len: {len(snake)}   (R to reset)"
        img = font.render(text, True, (20, 20, 20))
        screen.blit(img, (6, 6))

        if res.terminated:
            msg = "WIN!" if res.info.get("win") else f"DEAD: {res.info.get('death','?')}"
            img2 = font.render(msg, True, (0, 0, 0))
            screen.blit(img2, (6, 28))
        elif res.truncated:
            img2 = font.render("TRUNCATED (stall cap) - press R", True, (0, 0, 0))
            screen.blit(img2, (6, 28))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
