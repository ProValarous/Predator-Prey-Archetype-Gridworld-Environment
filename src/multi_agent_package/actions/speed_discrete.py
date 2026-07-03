from .discrete_actions import DiscreteActionSpace


class SpeedDiscreteActionSpace(DiscreteActionSpace):
    """
    5-action discrete space with micro-stepping.

    to_moves() returns up to min(speed, stamina) direction vectors so the
    caller can apply each sub-step independently, deducting 1 stamina per
    successful move.  NOOP always returns an empty list (zero stamina cost).
    """

    def to_moves(self, action: int, speed: int, stamina: int) -> list:
        direction = self.to_direction(action)
        if not direction.any():  # NOOP — no movement, no stamina cost
            return []
        n = min(max(speed, 1), max(stamina, 0))
        return [direction] * n
