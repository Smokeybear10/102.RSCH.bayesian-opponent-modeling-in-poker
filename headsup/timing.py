from dataclasses import dataclass
from math import isfinite
from random import Random


def _finite_time(name: str, value: float) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True, slots=True)
class ActionSchedule:
    observed_at: float
    target_at: float
    deadline: float | None
    latest_send_at: float | None

    def __post_init__(self) -> None:
        _finite_time("observed_at", self.observed_at)
        _finite_time("target_at", self.target_at)
        if self.target_at < self.observed_at:
            raise ValueError("target_at cannot precede observed_at")
        if (self.deadline is None) != (self.latest_send_at is None):
            raise ValueError("deadline and latest_send_at must both be present or absent")
        if self.deadline is not None:
            _finite_time("deadline", self.deadline)
            if (type(self.latest_send_at) not in (int, float) or not isfinite(self.latest_send_at)
                    or self.latest_send_at > self.deadline):
                raise ValueError("latest_send_at must be finite and no later than deadline")

    def remaining(self, now: float) -> float | None:
        _finite_time("now", now)
        if now < self.observed_at:
            raise ValueError("now cannot precede observed_at")
        if self.latest_send_at is not None and now >= self.latest_send_at:
            return None
        return max(0.0, self.target_at - now)


@dataclass(frozen=True, slots=True)
class ActionTiming:
    min_delay: float = 0.3
    max_delay: float = 1.1
    reserve: float = 1.0

    def __post_init__(self) -> None:
        for name in ("min_delay", "max_delay", "reserve"):
            _finite_time(name, getattr(self, name))
        if self.min_delay > self.max_delay:
            raise ValueError("min_delay cannot exceed max_delay")

    def plan(self, observed_at: float, rng: Random, deadline: float | None = None) -> ActionSchedule:
        _finite_time("observed_at", observed_at)
        if deadline is not None:
            _finite_time("deadline", deadline)
        latest_send_at = None if deadline is None else deadline - self.reserve
        target_at = observed_at + rng.uniform(self.min_delay, self.max_delay)
        if latest_send_at is not None and target_at >= latest_send_at:
            target_at = observed_at
        return ActionSchedule(observed_at, target_at, deadline, latest_send_at)
