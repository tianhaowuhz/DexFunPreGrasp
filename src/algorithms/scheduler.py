class BetaScheduler:
    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps

    def __call__(self, step: int) -> float:
        raise NotImplementedError


class FixedBetaScheduler(BetaScheduler):
    def __init__(self, value: float = 0):
        super().__init__(value, value, 0)

    def __call__(self, step: int) -> float:
        assert self.start == self.end
        return self.start


class LinearBetaScheduler(BetaScheduler):
    def __call__(self, step: int) -> float:
        step = max(0, min(step, self.num_steps))
        return self.start + (self.end - self.start) * min(1.0, max(0.0, step / self.num_steps))


class ExponentialBetaScheduler(BetaScheduler):
    def __call__(self, step: int) -> float:
        step = max(0, min(step, self.num_steps))
        return self.start * ((self.end / self.start) ** (step / self.num_steps))
