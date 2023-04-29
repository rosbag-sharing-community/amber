from typing import Any
from abc import ABC, abstractmethod


class Automation(ABC):
    @abstractmethod
    def inference(self) -> Any:
        pass
