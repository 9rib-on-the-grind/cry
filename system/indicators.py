import rolling
import data



class BaseIndicator:
    name = 'Base Indicator'

    def __init__(self):
        self.update_hash = None

    def set_data(self, data: data.DataMaintainer):
        self._data = data
        self.init_state()

    def get_parameters(self):
        raise NotImplementedError()

    def init_state(self):
        """Calculate initial state"""
        raise NotImplementedError()

    def is_updated(self):
        return self.update_hash == self._data.update_hash

    def update(self):
        raise NotImplementedError()



class MovingAverageIndicator(BaseIndicator):
    name = 'MA'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source = source

    def init_state(self):
        self._sma = rolling.Mean(length=self.length)
        for val in self._data['Init', self.source]:
            self.update(val)

    def get_state(self):
        return self._sma.get_state()
            
    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if initialization else self._data[self.source]
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash
            self._sma.append(val)

    def get_parameters(self):
        return {'length': self.length, 'source': self.source}



class RelativeStrengthIndexIndicator(BaseIndicator):
    name = 'RSI'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source = source

    def init_state(self):
        self._up_smma = rolling.ExponentialAverage(alpha=1/self.length)
        self._down_smma = rolling.ExponentialAverage(alpha=1/self.length)
        self._prev = 0
        for val in self._data['Init', self.source]:
            self.update(val)

    def get_state(self):
        rs = self._up_smma.get_state() / self._down_smma.get_state()
        rsi = 100 - 100 / (1 + rs)
        return rsi
            
    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if initialization else self._data[self.source]
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash
            diff = val - self._prev
            up = max(val - self._prev, 0)
            down = max(self._prev - val, 0)
            self._up_smma.append(up)
            self._down_smma.append(down)
            self._prev = val
            
    def get_parameters(self):
        return {'length': self.length, 'source': self.source}



class TripleExponentialIndicator(BaseIndicator):
    name = 'TRIX'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source = source

    def init_state(self):
        self._tema = rolling.TripleExponentialAverage(span=self.length)
        self._signal_line = rolling.Mean(length=self.length)
        self._prev = 1
        for val in self._data['Init', self.source]:
            self.update(val)

    def get_state(self):
        return 100 * 100 * (self._tema.get_state() - self._prev) / self._prev
            
    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if initialization else self._data[self.source]
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash
            self._prev = self._tema.get_state()
            self._tema.append(val)
            self._signal_line.append(self._tema.get_state())

    def get_parameters(self):
        return {'length': self.length, 'source': self.source}



class IchimokuKinkoHyoIndicator(BaseIndicator):
    name = 'Ichimoku'

    def __init__(self, short: int, long: int, **kwargs):
        super().__init__(**kwargs)

        self.short = short
        self.mid = long // 2
        self.long = long

    def init_state(self):
        self.min_short = rolling.Min(length=self.short)
        self.max_short = rolling.Max(length=self.short)
        self.min_mid = rolling.Min(length=self.mid)
        self.max_mid = rolling.Max(length=self.mid)
        self.min_long = rolling.Min(length=self.long)
        self.max_long = rolling.Max(length=self.long)
        self.senkouA = rolling.Lag(length=self.mid)
        self.senkouB = rolling.Lag(length=self.mid)
        for low, high in zip(self._data['Init', 'Low'], self._data['Init', 'High']):
            self.update(low, high)

    def get_state(self):
        return self.tenkan, self.kijun, self.senkouA.get_state(), self.senkouB.get_state()

    def update(self, low: float = None, high: float = None):
        initialization = (low is not None)
        low, high = (low, high) if initialization else (self._data['Low'], self._data['High'])
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash

            for min in (self.min_short, self.min_mid, self.min_long):
                min.append(low)
            for max in (self.max_short, self.max_mid, self.max_long):
                max.append(high)
            self.tenkan = (self.max_short.get_state() + self.min_short.get_state()) / 2
            self.kijun = (self.max_mid.get_state() + self.min_mid.get_state()) / 2
            self.senkouA.append((self.tenkan + self.kijun) / 2)
            self.senkouB.append((self.max_long.get_state() + self.min_long.get_state()) / 2)

    def get_parameters(self):
        return {'short': self.short, 'long': self.long}



class BollingerBandsIndicator(BaseIndicator):
    name = 'BB'

    def __init__(self, length: int, mult: float, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.mult = mult
        self.source = source

    def init_state(self):
        self._sma = rolling.Mean(length=self.length)
        self._mstd = rolling.StandardDeviation(length=self.length)
        for val in self._data['Init', self.source]:
            self.update(val)

    def get_state(self):
        median, std = self._sma.get_state(), self._mstd.get_state()
        return median, median - self.mult*std, median + self.mult*std

    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if initialization else self._data[self.source]
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash
            self._sma.append(val)
            self._mstd.append(val)

    def get_parameters(self):
        return {'length': self.length, 'mult': self.mult}