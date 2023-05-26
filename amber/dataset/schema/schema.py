from abc import ABC
from abc import ABC, abstractmethod


class Schema(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def schema(self) -> str:
        pass
