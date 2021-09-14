"""Module for time-efficient statistics calculation on updating data stream."""

import collections



class BaseRollingWindow:
    def __init__(self, length: int = None, *, enqueueing: bool = True):
        self.length = length
        if enqueueing:
            self._queue = collections.deque(maxlen=self.length)
        self._state = 0

    def append(self, val: float):
        raise NotImplemented()

    def get_state(self):
        return self._state



class Average(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length)

    def append(self, val: float):
        if len(self._queue) >= self.length:
            self._state -= self._queue.popleft() / self.length
        self._queue.append(val)
        self._state += val / self.length



class ExponentialAverage(BaseRollingWindow):
    def __init__(self, *, alpha: float = None, span: float = None):
        super().__init__(enqueueing=False)
        self._alpha = alpha or 2 / (span + 1)

    def append(self, val: float):
        self._state = (1 - self._alpha) * self._state + self._alpha * val



class TripleExponentialAverage(BaseRollingWindow):
    def __init__(self, alpha: float = None, span: float = None):
        super().__init__(enqueueing=False)
        self._alpha = alpha or 2 / (span + 1)
        self._exp_avg = [ExponentialAverage(alpha=self._alpha) for _ in range(3)]

    def append(self, val: float):
        for avg in self._exp_avg:
            avg.append(val)
            val = avg.get_state()
        self._state = val



class Min(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length)
        self._queue.append((-self.length, None))
        self.time = 0

    def append(self, val: float):
        if self._queue[0][0] == self.time - self.length:
            self._queue.popleft()
        while self._queue and self._queue[-1][1] >= val:
            self._queue.pop()
        self._queue.append((self.time, val))
        self._state = self._queue[0][1]
        self.time += 1



class Max(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length)
        self._queue.append((-self.length, None))
        self.time = 0

    def append(self, val: float):
        if self._queue[0][0] == self.time - self.length:
            self._queue.popleft()
        while self._queue and self._queue[-1][1] <= val:
            self._queue.pop()
        self._queue.append((self.time, val))
        self._state = self._queue[0][1]
        self.time += 1



class Sum(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length)

    def append(self, val: float):
        if len(self._queue) >= self.length:
            self._state -= self._queue.popleft()
        self._queue.append(val)
        self._state += val



class Lag(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length)
        self._queue.append(0)

    def append(self, val: float):
        self._state = self._queue[0]
        self._queue.append(val)



class Variance(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length, enqueueing=False)
        self._sum = Sum(self.length)
        self._sq_sum = Sum(self.length)

    def append(self, val: float):
        self._sum.append(val)
        self._sq_sum.append(val ** 2)
        s, sq, n = self._sum.get_state(), self._sq_sum.get_state(), self.length
        self._state = (sq / n - (s / n) ** 2)



class StandardDeviation(BaseRollingWindow):
    def __init__(self, length):
        super().__init__(length, enqueueing=False)
        self._var = Variance(self.length)

    def append(self, val: float):
        self._var.append(val)
        self._state = self._var.get_state() ** .5