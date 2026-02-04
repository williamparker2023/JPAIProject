from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple, Set

from env.snake_env import Action, DIR_VECS, opposite

Coord = Tuple[int, int]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos: Coord) -> List[Tuple[Action, Coord]]:
    x, y = pos
    out = []
    for a, (dx, dy) in DIR_VECS.items():
        out.append((a, (x + dx, y + dy)))
    return out


@dataclass(frozen=True)
class Plan:
    actions: List[Action]


def astar_path(state: Dict, start: Coord, goal: Coord, blocked: Set[Coord], disallow_reverse_from: Optional[Action]) -> Optional[Plan]:
    w, h = state["width"], state["height"]

    def in_bounds(p: Coord) -> bool:
        return 0 <= p[0] < w and 0 <= p[1] < h

    # A* over positions (not full snake dynamics). Works as baseline.
    pq = []
    heappush(pq, (manhattan(start, goal), 0, start, []))
    seen = set()

    while pq:
        f, g, pos, path = heappop(pq)
        if pos in seen:
            continue
        seen.add(pos)

        if pos == goal:
            return Plan(path)

        for a, nxt in neighbors(pos):
            if len(path) == 0 and disallow_reverse_from is not None and opposite(a, disallow_reverse_from):
                continue
            if not in_bounds(nxt):
                continue
            if nxt in blocked:
                continue
            if nxt in seen:
                continue
            ng = g + 1
            nf = ng + manhattan(nxt, goal)
            heappush(pq, (nf, ng, nxt, path + [a]))

    return None


class AStarAgent:
    """
    Baseline:
    - Try A* to food treating body as obstacles (tail treated as obstacle too)
    - If no path, try A* to tail (survival-ish)
    - If still no path, choose safest move (avoid immediate death, maximize distance to food as tie-break)
    """
    def act(self, obs: Dict) -> Action:
        snake: List[Coord] = obs["snake"]
        head = snake[-1]
        tail = snake[0]
        food = obs["food"]
        direction = Action(obs["direction"])

        body_blocked = set(snake)  # conservative: blocks tail too

        plan_food = astar_path(obs, head, food, body_blocked, disallow_reverse_from=direction)
        if plan_food and plan_food.actions:
            a = plan_food.actions[0]
            if a in self._legal_actions(direction):
                return a

        # fallback: try reach tail (lets it “follow itself” to survive)
        plan_tail = astar_path(obs, head, tail, body_blocked - {tail}, disallow_reverse_from=direction)
        if plan_tail and plan_tail.actions:
            a = plan_tail.actions[0]
            if a in self._legal_actions(direction):
                return a

        # final fallback: pick a safe action
        best = None
        best_score = -10**9
        for a in self._legal_actions(direction):
            nxt = (head[0] + DIR_VECS[a][0], head[1] + DIR_VECS[a][1])
            if self._would_die(obs, nxt):
                continue
            # heuristic: prefer moves that reduce distance to food slightly, but prioritize safety
            score = -manhattan(nxt, food)
            if score > best_score:
                best_score = score
                best = a
        return best if best is not None else direction  # if boxed in, keep going

    def _legal_actions(self, direction: Action) -> List[Action]:
        acts = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        return [a for a in acts if not opposite(a, direction)]

    def _would_die(self, obs: Dict, nxt: Coord) -> bool:
        w, h = obs["width"], obs["height"]
        if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
            return True
        snake = obs["snake"]
        body = set(snake)
        tail = snake[0]
        food = obs["food"]
        will_grow = (nxt == food)
        if will_grow:
            return nxt in body
        else:
            return (nxt in body) and (nxt != tail)
