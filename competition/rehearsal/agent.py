"""Rehearsal entry point. Adapt the real starter's Agent class on event day."""

from adapter import DemoAdapter
from sdk import SDKEquity

from competition.bridge import Bridge
from competition.policy import TournamentPolicy


class Agent:
    def __init__(self) -> None:
        self.bridge = Bridge(
            DemoAdapter(), lambda: TournamentPolicy(equity=SDKEquity())
        )

    def act(self, observation: object) -> object:
        return self.bridge.decide(observation)
