from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from env.snake_env import Action


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Any) -> Action:
        raise NotImplementedError
