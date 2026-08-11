from dataclasses import dataclass
from typing import Protocol

from .model import Observation


@dataclass(frozen=True, slots=True)
class Estimate:
    share: float
    call_return: float
    win: float
    tie: float
    standard_error: float
    samples: int
    effective_samples: float


class Equity(Protocol):
    def estimate(
        self,
        obs: Observation,
        ranges: tuple[float, ...],
        pressure: tuple[int, ...],
        samples: int = 40,
        deadline: int | None = None,
        information: tuple[float, ...] | None = None,
    ) -> Estimate: ...
