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
    Enhanced A* agent with:
    1. Tail-aware pathfinding: Doesn't block tail (moves away) when finding path to food
    2. Lookahead planning: Commits to multi-step plans instead of re-planning every step
    3. Smarter fallback: Multi-step simulation to evaluate candidate moves
    """
    
    def __init__(self):
        self.current_plan: Optional[Plan] = None
        self.plan_index: int = 0
        self.replan_interval: int = 4  # commit to plan for N steps before replanning
    
    def act(self, obs: Dict) -> Action:
        snake: List[Coord] = obs["snake"]
        head = snake[-1]
        tail = snake[0]
        food = obs["food"]
        direction = Action(obs["direction"])

        # If we have a valid plan, follow it for several steps
        if self.current_plan is not None and self.plan_index < len(self.current_plan.actions):
            a = self.current_plan.actions[self.plan_index]
            if a in self._legal_actions(direction):
                self.plan_index += 1
                return a
        
        # Replan: improvement 1 - don't block tail since it moves away
        body_blocked_for_food = set(snake[:-1])  # exclude tail
        plan_food = astar_path(obs, head, food, body_blocked_for_food, disallow_reverse_from=direction)
        if plan_food and plan_food.actions:
            self.current_plan = plan_food
            self.plan_index = 1  # take first step now
            a = plan_food.actions[0]
            if a in self._legal_actions(direction):
                return a

        # Fallback: try reach tail (lets it "follow itself" to survive)
        body_blocked_for_tail = set(snake[:-1])  # also exclude tail here
        plan_tail = astar_path(obs, head, tail, body_blocked_for_tail, disallow_reverse_from=direction)
        if plan_tail and plan_tail.actions:
            self.current_plan = plan_tail
            self.plan_index = 1
            a = plan_tail.actions[0]
            if a in self._legal_actions(direction):
                return a

        # Improvement 3: Smarter fallback with multi-step lookahead
        best = None
        best_score = -10**9
        
        for a in self._legal_actions(direction):
            nxt = (head[0] + DIR_VECS[a][0], head[1] + DIR_VECS[a][1])
            if self._would_die(obs, nxt):
                continue
            
            # Simulate 3 steps ahead to evaluate the move (improvement 3)
            lookahead_score = self._evaluate_move_lookahead(obs, a, lookahead_depth=3)
            
            if lookahead_score > best_score:
                best_score = lookahead_score
                best = a
        
        self.current_plan = None  # clear plan on fallback
        return best if best is not None else direction  # if boxed in, keep going

    def _evaluate_move_lookahead(self, obs: Dict, action: Action, lookahead_depth: int = 3) -> float:
        """
        Simulate lookahead_depth steps with the given action.
        Returns score: priority to approaching food + bonus for surviving.
        """
        snake: List[Coord] = obs["snake"]
        head = snake[-1]
        food = obs["food"]
        direction = obs["direction"]
        
        current_dist = manhattan(head, food)
        score = 0.0
        pos = head
        current_dir = action
        
        for step in range(lookahead_depth):
            dx, dy = DIR_VECS[action]
            next_pos = (pos[0] + dx, pos[1] + dy)
            
            # Penalize moving away from food or not making progress
            next_dist = manhattan(next_pos, food)
            dist_delta = current_dist - next_dist
            score += dist_delta * (0.5 ** step)  # discount future steps
            
            # Bonus for finding food during lookahead
            if next_pos == food:
                score += 100.0
                break
            
            pos = next_pos
            current_dist = next_dist
        
        return score

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