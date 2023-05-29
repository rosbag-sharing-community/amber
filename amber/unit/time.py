from enum import Enum


class TimeUnit(Enum):
    NANOSECOND = 1e-09
    """Type of nanosecond, the corresponding value indicates the number of seconds per unit.
    """
    MICROSECOND = 1e-06
    """Type of microsecond, the corresponding value indicates the number of seconds per unit.
    """
    MILLISECOND = 1e-3
    """Type of millisecond, the corresponding value indicates the number of seconds per unit.
    """
    SECOND = 1
    """Type of second, the corresponding value indicates the number of seconds per unit.
    """


class Time:
    __value: float = 0.0
    __unit: TimeUnit = TimeUnit.SECOND

    def __init__(self, value: float, unit: TimeUnit) -> None:
        self.value = value * unit.value

    def get(self, unit: TimeUnit) -> float:
        return float(self.value) / float(unit.value)
