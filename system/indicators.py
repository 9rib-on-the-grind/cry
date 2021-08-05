import rolling
import data



class BaseIndicator:
    """Base Indicator Class
    
    Args:
        _data: DataMaintainer. If needed [_data] object will suply the indicator
            with necessary information, must have [self.name] key.
        _min_history: Int. Minimum number of candlestick required for state maintenance # ????????????
    """

    name = 'Base Indicator'

    def __init__(self):
        self._state = None
        self.update_hash = None

    def set_data(self, data: data.DataMaintainer):
        self._history = data['History']

    def set_name(self):
        return f'{self.name} {self.get_parameters()}'

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
        self.name = self.set_name()

        self._sma = rolling.SimpleMovingAverage(length=self.length)

    def set_data(self, data: data.DataMaintainer):
        super().set_data(data)
        self._source = self._history[self.source_name]
        self.init_state()

    def init_state(self):
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
        return [self.length, self.source_name]



class RelativeStrengthIndexIndicator(BaseIndicator):
    name = 'RSI'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source_name = source
        self.name = self.set_name()

        self._up_smma = rolling.ExponentialMovingAverage(alpha=1/self.length)
        self._down_smma = rolling.ExponentialMovingAverage(alpha=1/self.length)

    def set_data(self, data: data.DataMaintainer):
        super().set_data(data)
        self._source = self._history[self.source_name]
        self.init_state()

    def init_state(self):
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
        return [self.length, self.source_name]