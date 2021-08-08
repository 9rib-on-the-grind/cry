import collections



class BaseRollingWindow:
    def __init__(self, length: int = None):
        self.length = length
        self._queue = collections.deque(maxlen=self.length)
        self._state = 0

    def get_state(self):
        return self._state



class Average(BaseRollingWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def append(self, val: float):
        if len(self._queue) >= self.length:
            self._state -= self._queue.popleft() / self.length
        self._state += val / self.length
        self._queue.append(val)



class ExponentialAverage(BaseRollingWindow):
    def __init__(self, alpha: float = None, span: float = None, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha or 2 / (span + 1)

    def append(self, val: float):
        self._state = (1 - self._alpha) * self._state + self._alpha * val



class TripleExponentialAverage(BaseRollingWindow):
    def __init__(self, alpha: float = None, span: float = None, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha or 2 / (span + 1)
        self._emas = [ExponentialAverage(self._alpha) for _ in range(3)]

    def append(self, val: float):
        for ema in self._emas:
            ema.append(val)
            val = ema.get_state()
        self._state = val



class Min(BaseRollingWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._queue.append((-self.length, None))
        self.time = 0

    def append(self, val: float):
        if not val:
            raise SystemError()
        if self._queue[0][0] == self.time - self.length:
            self._queue.popleft()
        while self._queue and self._queue[-1][1] >= val:
            self._queue.pop()
        self._queue.append((self.time, val))
        self._state = self._queue[0][1]
        self.time += 1



class Max(BaseRollingWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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



class Shift(BaseRollingWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._queue.append(0)

    def append(self, val: float):
        self._state = self._queue[0]
        self._queue.append(val)