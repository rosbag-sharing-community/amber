from typing import Protocol, Any


class Automation(Protocol):
    def inference(self) -> Any:
        ...
