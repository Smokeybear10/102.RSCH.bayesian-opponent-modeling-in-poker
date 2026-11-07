from dataclasses import dataclass

from .model import Event


@dataclass(slots=True)
class Tendency:
    faced: int = 0
    folds: int = 0
    actions: int = 0
    raises: int = 0
    preflop_actions: int = 0
    preflop_raises: int = 0
    preflop_continues: int = 0
    postflop_actions: int = 0
    postflop_raises: int = 0
    postflop_faced: int = 0
    postflop_folds: int = 0

    @property
    def fold_rate(self) -> float:
        return (self.postflop_folds + 5) / (self.postflop_faced + 12)


class Opponents:
    def __init__(self, seats: int = 8, learn_continuation: bool = True) -> None:
        self.seats = [Tendency() for _ in range(seats)]
        self.hand = -1
        self.ranges = [1.0] * seats
        self.pressure = [0] * seats
        self.street = -1
        self.learn_continuation = learn_continuation

    def information(self) -> tuple[float, ...]:
        result = []
        for seat, pressure in zip(self.seats, self.pressure):
            confidence = 1.0
            if self.learn_continuation:
                if pressure == 1 and seat.postflop_faced >= 16:
                    confidence = min(1.0, seat.fold_rate / 0.4)
                elif pressure >= 2 and seat.postflop_actions >= 16:
                    confidence = min(
                        1.0,
                        0.25
                        / ((seat.postflop_raises + 2) / (seat.postflop_actions + 10)),
                    )
            result.append(confidence)
        return tuple(result)

    def begin_hand(self, hand: int) -> None:
        if hand != self.hand:
            self.hand = hand
            self.ranges = [1.0] * len(self.seats)
            self.pressure = [0] * len(self.seats)
            self.street = -1

    def begin_street(self, street: int) -> None:
        if street != self.street:
            self.street = street
            self.pressure = [0] * len(self.seats)

    def observe(self, event: Event) -> None:
        self.begin_hand(event.hand)
        self.begin_street(event.street)
        tendency = self.seats[event.seat]
        tendency.actions += 1
        if event.facing_bet:
            tendency.faced += 1
            tendency.folds += event.action.kind == "fold"
        tendency.raises += event.action.kind == "raise"
        if event.street != 0:
            tendency.postflop_actions += 1
            tendency.postflop_raises += event.action.kind == "raise"
            if event.facing_bet:
                tendency.postflop_faced += 1
                tendency.postflop_folds += event.action.kind == "fold"
            if event.action.kind == "raise":
                self.pressure[event.seat] = min(
                    3, max(2, self.pressure[event.seat] + 1)
                )
            elif event.action.kind == "call":
                self.pressure[event.seat] = max(1, self.pressure[event.seat])
            return
        tendency.preflop_actions += 1
        tendency.preflop_raises += event.action.kind == "raise"
        tendency.preflop_continues += event.action.kind in ("call", "raise")
        if event.action.kind == "raise":
            current = self.ranges[event.seat]
            learned = (tendency.preflop_raises + 3) / (tendency.preflop_actions + 10)
            fraction = (
                min(0.65, max(0.15, learned)) if tendency.preflop_actions >= 16 else 0.3
            )
            self.ranges[event.seat] = min(
                current, fraction if current == 1 else fraction * 0.6
            )
        elif event.action.kind == "call":
            fraction = 0.65
            if self.learn_continuation and tendency.preflop_actions >= 16:
                fraction = min(
                    0.95,
                    max(
                        0.25,
                        (tendency.preflop_continues + 6)
                        / (tendency.preflop_actions + 10),
                    ),
                )
            self.ranges[event.seat] = min(self.ranges[event.seat], fraction)
