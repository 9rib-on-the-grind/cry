import collections



class BaseRollingWindow:
    def __init__(self, length: int = None):
        self.length = length
        self._queue = collections.deque(maxlen=self.length)
        self._state = 0

    def get_state(self):
        return self._state



class SimpleMovingAverage(BaseRollingWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def append(self, val: float):
        if len(self._queue) >= self.length:
            self._state -= self._queue.popleft() / self.length
        self._state += val / self.length
        self._queue.append(val)



class ExponentialMovingAverage(BaseRollingWindow):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha

    def append(self, val: float):
        self._state = (1 - self._alpha) * self._state + self._alpha * val