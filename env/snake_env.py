from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Deque, List, Optional, Tuple
from collections import deque
import random

Coord = Tuple[int, int]


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIR_VECS = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}


def opposite(a: Action, b: Action) -> bool:
    return (a == Action.UP and b == Action.DOWN) or \
           (a == Action.DOWN and b == Action.UP) or \
           (a == Action.LEFT and b == Action.RIGHT) or \
           (a == Action.RIGHT and b == Action.LEFT)


@dataclass(frozen=True)
class StepResult:
    obs: object
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class SnakeEnv:
    """
    Minimal Snake environment (gym-like but no gym dependency).

    - Grid: width x height (default 16x16)
    - Walls kill
    - Win when fills board
    - Training cap to prevent stalling:
        truncate if steps_since_food >= 100 * len(snake) OR total_steps >= grid_area * 50
    """

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.grid_area = width * height

        self._rng = random.Random(seed)

        self.snake: Deque[Coord] = deque()
        self.direction: Action = Action.RIGHT
        self.food: Coord = (0, 0)

        self.steps_since_food: int = 0
        self.total_steps: int = 0
        self.score: int = 0

        self._terminated: bool = False
        self._truncated: bool = False

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
        if seed is not None:
            self._rng.seed(seed)

        self._terminated = False
        self._truncated = False
        self.steps_since_food = 0
        self.total_steps = 0
        self.score = 0

        # Start centered, length 3, facing right
        cx, cy = self.width // 2, self.height // 2
        self.direction = Action.RIGHT
        self.snake = deque([(cx - 1, cy), (cx, cy), (cx + 1, cy)])  # tail -> head
        self._spawn_food()

        obs = self._get_state()
        info = {"score": self.score}
        return obs, info

    def legal_actions(self) -> List[Action]:
        # No 180-degree reversal
        acts = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        return [a for a in acts if not opposite(a, self.direction)]

    def step(self, action: Action) -> StepResult:
        if self._terminated or self._truncated:
            raise RuntimeError("step() called on terminated/truncated episode. Call reset().")

        if action not in self.legal_actions():
            # Treat illegal reverse as "keep going" to avoid agents hard-crashing.
            action = self.direction

        self.direction = action
        dx, dy = DIR_VECS[action]

        head_x, head_y = self.snake[-1]
        new_head = (head_x + dx, head_y + dy)

        self.total_steps += 1
        self.steps_since_food += 1

        reward = 0.0
        info = {}

        # Wall collision
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            self._terminated = True
            reward = -1.0
            info["death"] = "wall"
            return StepResult(self._get_state(), reward, True, False, info)

        # Self collision: moving into body (tail moves unless we eat)
        body_set = set(self.snake)
        tail = self.snake[0]
        will_grow = (new_head == self.food)
        if will_grow:
            # tail stays, so any overlap is collision
            if new_head in body_set:
                self._terminated = True
                reward = -1.0
                info["death"] = "self"
                return StepResult(self._get_state(), reward, True, False, info)
        else:
            # tail will move away, so stepping into current tail is allowed
            if new_head in body_set and new_head != tail:
                self._terminated = True
                reward = -1.0
                info["death"] = "self"
                return StepResult(self._get_state(), reward, True, False, info)

        # Move
        self.snake.append(new_head)
        if will_grow:
            self.score += 1
            self.steps_since_food = 0
            reward = +1.0
            self._spawn_food()
        else:
            self.snake.popleft()
            # small step penalty helps RL learn faster; keeps fairness for GA too
            reward = -0.01

        # Win condition
        if len(self.snake) == self.grid_area:
            self._terminated = True
            reward = +10.0
            info["win"] = True
            return StepResult(self._get_state(), reward, True, False, info)

        # Truncation cap to avoid stalling
        stall_cap = 100 * len(self.snake)
        hard_cap = self.grid_area * 50
        if self.steps_since_food >= stall_cap or self.total_steps >= hard_cap:
            self._truncated = True
            info["truncated_reason"] = "stall" if self.steps_since_food >= stall_cap else "hard_cap"
            return StepResult(self._get_state(), 0.0, False, True, info)

        return StepResult(self._get_state(), reward, False, False, info)

    def _spawn_food(self) -> None:
        occupied = set(self.snake)
        empties = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        if not empties:
            # already full (win handled elsewhere)
            self.food = (-1, -1)
            return
        self.food = self._rng.choice(empties)

    def _get_state(self) -> dict:
        # State dict is used by A*, feature extractor, UI.
        return {
            "width": self.width,
            "height": self.height,
            "snake": list(self.snake),  # tail -> head
            "direction": int(self.direction),
            "food": self.food,
            "score": self.score,
            "steps_since_food": self.steps_since_food,
            "total_steps": self.total_steps,
        }
