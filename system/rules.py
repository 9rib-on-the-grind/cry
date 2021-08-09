import indicators
from decision import Decision



class CrossoverState:
    """Helping class for tracking relative position between two lines.
    This class tracks which of two line is dominating (higher) and 
    how many steps ago they changed dominance (crossed).
    """

    def __init__(self):
        self._a = self._b = float('inf')
    
    def update(self, a: float, b: float):
        """Update relative positions of two lines, return a and b dominance"""
        if a > b:
            self._a += 1
            self._b = 0
        else:
            self._b += 1
            self._a = 0
        return self._a, self._b



class DirectionState:
    """Helping class for tracking length and direction of change."""

    def __init__(self):
        self._length = 1
        self._dir = 0
    
    def update(self, change: float):
        if change > 0 and self._dir > 0 or change < 0 and self._dir < 0:
            self._length += 1
        else:
            self._dir = change
            self._length = 1
        return self._dir > 0, self._length



class BaseRule:
    name = 'Base Rule'
    
    def __init__(self, patience: int = 1):
        self._state = None
        self._patience = patience

    def update(self):
        raise NotImplementedError()
    
    def decide(self):
        raise NotImplementedError()

    def get_parameters(self):
        return {'patience': self._patience}

    def signal(self, buy: int, sell: int, instant: bool = True):
        condition = lambda x: x == self._patience if instant else lambda x: x >= self._patience
        if condition(buy):
            return Decision.BUY
        elif condition(sell):
            return Decision.SELL
        else:
            return Decision.WAIT




class BaseCrossoverRule(BaseRule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class BaseTrasholdRule(BaseRule):
    def __init__(self, lower: float, upper: float, **kwargs):
        self._upper = upper
        self._lower = lower
        super().__init__(**kwargs)



class BaseDirectionChangeRule(BaseRule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class MovingAverageCrossoverRule(BaseCrossoverRule):
    name = 'MACrossover'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cross = CrossoverState()

    def decide(self, slow: indicators.MovingAverageIndicator, 
                     fast: indicators.MovingAverageIndicator):
        buy, sell = self._cross.update(fast.get_state(), slow.get_state())
        return self.signal(buy, sell)



class RelativeStrengthIndexTrasholdRule(BaseTrasholdRule):
    name = 'RSITrashold'

    def __init__(self, lower: float, upper: float, **kwargs):
        """
        Args:
            offset. Float in (0, 50). Trashold levels are defined as 50 +- offset"""
        super().__init__(lower, upper, **kwargs)
        self._lower_cross = CrossoverState()
        self._upper_cross = CrossoverState()

    def decide(self, rsi: indicators.RelativeStrengthIndexIndicator):
        val = rsi.get_state()
        buy, _ = self._lower_cross.update(self._lower, val)
        sell, _ = self._upper_cross.update(val, self._upper)
        return self.signal(buy, sell, instant=False)

    def get_parameters(self):
        return {'lower': self._lower, 
                'upper': self._upper, 
                'patience': self._patience}



class TripleExponentialDirectionChangeRule(BaseDirectionChangeRule):
    name = 'TRIXDirectionChange'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dir = DirectionState()
        self._prev = 0

    def decide(self, trix: indicators.TripleExponentialIndicator):
        inc, length = self._dir.update(trix.get_state() - self._prev)
        self._prev = trix.get_state()
        buy, sell = (length, 0) if inc else (0, length)
        return self.signal(buy, sell)



class IchimokuKinkoHyoTenkanKijunCrossoverRule(BaseCrossoverRule):
    name = 'IchimokuTenkanKijunCrossover'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cross = CrossoverState()

    def decide(self, ichimoku: indicators.IchimokuKinkoHyoIndicator):
        tenkan, kijun, _, _ = ichimoku.get_state()
        buy, sell = self._cross.update(tenkan, kijun)
        return self.signal(buy, sell)



class BollingerBandsLowerUpperCrossoverRule(BaseCrossoverRule):
    name = 'BBLowerUpperCrossover'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lower_cross = CrossoverState()
        self._upper_cross = CrossoverState()

    def decide(self, bb: indicators.BollingerBandsIndicator,
                     ma: indicators.MovingAverageIndicator):
        mid, lower, upper = bb.get_state()
        val = ma.get_state()
        buy, _ = self._lower_cross.update(lower, val)
        sell, _ = self._upper_cross.update(val, upper)
        return self.signal(buy, sell, instant=False)



class BollingerBandsLowerMidCrossoverRule(BaseCrossoverRule):
    name = 'BBLowerMidCrossover'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lower_cross = CrossoverState()
        self._mid_cross = CrossoverState()

    def decide(self, bb: indicators.BollingerBandsIndicator,
                     ma: indicators.MovingAverageIndicator):
        mid, lower, upper = bb.get_state()
        val = ma.get_state()
        buy, _ = self._lower_cross.update(lower, val)
        sell, _ = self._mid_cross.update(val, mid)
        return self.signal(buy, sell)



class BollingerBandsUpperMidCrossoverRule(BaseCrossoverRule):
    name = 'BBUpperMidCrossover'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._upper_cross = CrossoverState()
        self._mid_cross = CrossoverState()

    def decide(self, bb: indicators.BollingerBandsIndicator,
                     ma: indicators.MovingAverageIndicator):
        mid, lower, upper = bb.get_state()
        val = ma.get_state()
        buy, _ = self._upper_cross.update(val, upper)
        sell, _ = self._mid_cross.update(mid, val)
        return self.signal(buy, sell)