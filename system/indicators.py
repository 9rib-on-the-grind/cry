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



class PriceIndicator(BaseIndicator):
    name = 'Price'

    def __init__(self, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.source = source

    def init_state(self):
        pass

    def get_state(self):
        return self._data[self.source]

    def update(self):
        pass

    def get_parameters(self):
        return {'source': self.source}



class MovingAverageIndicator(BaseIndicator):
    name = 'MA'

    def __init__(self, length: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.source = source

    def init_state(self):
        self._sma = rolling.Average(length=self.length)
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
        self._up = rolling.ExponentialAverage(alpha=1/self.length)
        self._down = rolling.ExponentialAverage(alpha=1/self.length)
        self._prev = 0
        for val in self._data['Init', self.source]:
            self.update(val)

    def get_state(self):
        up, down = self._up.get_state(), self._down.get_state()
        rs = up / down if down != 0 else float('inf')
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
            self._up.append(up)
            self._down.append(down)
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
        self._signal_line = rolling.Average(length=self.length)
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
        if not self.short < self.mid < self.long:
            raise ValueError('Invalid short/long period combination')

    def init_state(self):
        self.min_short = rolling.Min(length=self.short)
        self.max_short = rolling.Max(length=self.short)
        self.min_mid = rolling.Min(length=self.mid)
        self.max_mid = rolling.Max(length=self.mid)
        self.min_long = rolling.Min(length=self.long)
        self.max_long = rolling.Max(length=self.long)
        self.senkouA_lag = rolling.Lag(length=self.mid)
        self.senkouB_lag = rolling.Lag(length=self.mid)
        self.senkouA_last = None
        self.senkouB_last = None
        self.close_lag = rolling.Lag(length=self.mid)
        for high, low, close in zip(self._data['Init', 'High'], self._data['Init', 'Low'], self._data['Init', 'Close']):
            self.update(high, low, close)

    def get_state(self):
        return (self.tenkan, self.kijun, self.senkouA_last, self.senkouB_last,
                       self.senkouA_lag.get_state(), self.senkouB_lag.get_state(), self.close_lag.get_state())

    def update(self, high: float = None, low: float = None, close: float = None):
        initialization = (high is not None)
        high, low, close = (high, low, close) if initialization else (self._data['High'], self._data['Low'], self._data['Close'])
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash

            for min in (self.min_short, self.min_mid, self.min_long):
                min.append(low)
            for max in (self.max_short, self.max_mid, self.max_long):
                max.append(high)
            self.tenkan = (self.max_short.get_state() + self.min_short.get_state()) / 2
            self.kijun = (self.max_mid.get_state() + self.min_mid.get_state()) / 2
            self.senkouA_last = (self.tenkan + self.kijun) / 2
            self.senkouB_last = (self.max_long.get_state() + self.min_long.get_state()) / 2
            self.senkouA_lag.append(self.senkouA_last)
            self.senkouB_lag.append(self.senkouB_last)
            self.close_lag.append(close)

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
        self._sma = rolling.Average(length=self.length)
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



class MovingAverageConvergenceDivergenceIndicator(BaseIndicator):
    name = 'MACD'

    def __init__(self, long: int, signal: int, source: str = 'Close', **kwargs):
        super().__init__(**kwargs)

        self.long = long
        self.short = self.long // 2
        self.signal = signal
        self.source = source
        if not self.signal <= self.short:
            raise ValueError('Invalid short/long/signal period combination')

    def init_state(self):
        self._short_ema = rolling.ExponentialAverage(span=self.short)
        self._long_ema = rolling.ExponentialAverage(span=self.long)
        self._macd = None
        self._signal_line = rolling.ExponentialAverage(span=self.signal)
        for val in self._data['Init', self.source]:
            self.update(val)

    def get_state(self):
        return self._macd, self._signal_line.get_state()

    def update(self, val: float = None):
        initialization = (val is not None)
        val = val if initialization else self._data[self.source]
        if not self.is_updated() or initialization:
            self.update_hash = self._data.update_hash
            self._short_ema.append(val)
            self._long_ema.append(val)
            self._macd = self._short_ema.get_state() - self._long_ema.get_state()
            self._signal_line.append(self._macd)

    def get_parameters(self):
        return {'long': self.long, 'signal': self.signal, 'source': self.source}
