import rolling
import data



class BaseIndicator:
    name = 'Base Indicator'

    def __init__(self):
        self._state = None
        self.update_hash = None

    def set_data(self, data: data.DataMaintainer):
        self._history = data['History']
        self._source = self._history[self.source_name]
        self.init_state()

    def get_parameters(self):
        raise NotImplementedError()

    def init_state(self):
        """Calculate initial state"""
        raise NotImplementedError()

    def is_updated(self):
        return self.update_hash == self._history.update_hash

    def update(self):
        raise NotImplementedError()



class MovingAverageIndicator(BaseIndicator):
    name = 'MA'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source_name = source

    def init_state(self):
        self._sma = rolling.Average(length=self.length)
        for val in self._source:
            self.update(val)

    def get_state(self):
        return self._sma.get_state()
            
    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if val is not None else self._source[-1]
        if not self.is_updated() or initialization:
            self.update_hash = self._history.update_hash
            self._sma.append(val)

    def get_parameters(self):
        return {'length': self.length, 'source': self.source_name}



class RelativeStrengthIndexIndicator(BaseIndicator):
    name = 'RSI'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source_name = source

    def init_state(self):
        self._up_smma = rolling.ExponentialAverage(alpha=1/self.length)
        self._down_smma = rolling.ExponentialAverage(alpha=1/self.length)
        self._prev = 0
        for val in self._source:
            self.update(val)

    def get_state(self):
        rs = self._up_smma.get_state() / self._down_smma.get_state()
        rsi = 100 - 100 / (1 + rs)
        return rsi
            
    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if val is not None else self._source[-1]
        if not self.is_updated() or initialization:
            self.update_hash = self._history.update_hash
            diff = val - self._prev
            up = max(val - self._prev, 0)
            down = max(self._prev - val, 0)
            self._up_smma.append(up)
            self._down_smma.append(down)
            self._prev = val
            
    def get_parameters(self):
        return {'length': self.length, 'source': self.source_name}



class TripleExponentialIndicator(BaseIndicator):
    name = 'TRIX'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source_name = source

    def init_state(self):
        self._tma = rolling.TripleExponentialAverage(span=self.length)
        self._signal_line = rolling.Average(length=self.length)
        self._prev = 1
        for val in self._source:
            self.update(val)

    def get_state(self):
        return 100 * 100 * (self._tma.get_state() - self._prev) / self._prev
            
    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if val is not None else self._source[-1]
        if not self.is_updated() or initialization:
            self.update_hash = self._history.update_hash
            self._prev = self._tma.get_state()
            self._tma.append(val)
            self._signal_line.append(self._tma.get_state())

    def get_parameters(self):
        return {'length': self.length, 'source': self.source_name}