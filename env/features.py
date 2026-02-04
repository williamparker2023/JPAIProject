from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np

from env.snake_env import Action, DIR_VECS

Coord = Tuple[int, int]


def _turn_left(a: Action) -> Action:
    return Action((int(a) - 1) % 4)


def _turn_right(a: Action) -> Action:
    return Action((int(a) + 1) % 4)


def _danger(state: Dict, next_pos: Coord, will_grow: bool) -> float:
    w, h = state["width"], state["height"]
    if not (0 <= next_pos[0] < w and 0 <= next_pos[1] < h):
        return 1.0

    snake: List[Coord] = state["snake"]
    body = set(snake)
    tail = snake[0]
    food = state["food"]

    if will_grow:
        # tail stays, so any overlap is collision
        return 1.0 if next_pos in body else 0.0
    else:
        # tail moves away, so stepping into tail is allowed
        if next_pos in body and next_pos != tail:
            return 1.0
        return 0.0


def compact_features(state: Dict) -> np.ndarray:
    """
    11 features:
      danger_straight, danger_left, danger_right (3)
      direction one-hot (4)
      food relative (left/right/up/down) (4)

    Returns float32 array shape (11,)
    """
    snake = state["snake"]
    head = snake[-1]
    food = state["food"]
    direction = Action(state["direction"])

    # Next positions
    def next_pos_for(act: Action) -> Coord:
        dx, dy = DIR_VECS[act]
        return (head[0] + dx, head[1] + dy)

    straight = direction
    left = _turn_left(direction)
    right = _turn_right(direction)

    will_grow_straight = (next_pos_for(straight) == food)
    will_grow_left = (next_pos_for(left) == food)
    will_grow_right = (next_pos_for(right) == food)

    danger_straight = _danger(state, next_pos_for(straight), will_grow_straight)
    danger_left = _danger(state, next_pos_for(left), will_grow_left)
    danger_right = _danger(state, next_pos_for(right), will_grow_right)

    dir_onehot = np.zeros(4, dtype=np.float32)
    dir_onehot[int(direction)] = 1.0

    food_left = 1.0 if food[0] < head[0] else 0.0
    food_right = 1.0 if food[0] > head[0] else 0.0
    food_up = 1.0 if food[1] < head[1] else 0.0
    food_down = 1.0 if food[1] > head[1] else 0.0

    feats = np.array(
        [danger_straight, danger_left, danger_right,
         *dir_onehot.tolist(),
         food_left, food_right, food_up, food_down],
        dtype=np.float32
    )
    return feats
