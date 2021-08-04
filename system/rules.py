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



class BaseRule:
    name = 'Base Rule'
    
    def __init__(self, patience: int = 1):
        self._state = None
        self._patience = patience

    def update(self):
        raise NotImplementedError()
    
    def decide(self):
        raise NotImplementedError()



class BaseCrossoverRule(BaseRule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class BaseTrasholdRule(BaseRule):
    def __init__(self, lower: float, upper: float, **kwargs):
        super().__init__(**kwargs)
        self._upper = upper
        self._lower = lower



class MovingAverageCrossoverRule(BaseCrossoverRule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cross = CrossoverState()

    def decide(self, slow: indicators.MovingAverageIndicator, 
                     fast: indicators.MovingAverageIndicator):
        buy, sell = self.cross.update(fast.get_state(), slow.get_state())
        if buy == self._patience:
            return Decision.BUY
        elif sell == self._patience:
            return Decision.SELL
        else:
            return Decision.WAIT



class RelativeStrengthIndexTrasholdRule(BaseTrasholdRule):
    def __init__(self, lower: float, upper: float, **kwargs):
        """
        Args:
            offset. Float in (0, 50). Trashold levels are defined as 50 +- offset"""
        super().__init__(lower, upper, **kwargs)
        self.lower_cross = CrossoverState()
        self.upper_cross = CrossoverState()

    def decide(self, rsi: indicators.RelativeStrengthIndexIndicator):
        val = rsi.get_state()
        buy, _ = self.lower_cross.update(self._lower, val)
        sell, _ = self.upper_cross.update(val, self._upper)
        if buy >= self._patience:
            return Decision.BUY
        elif sell >= self._patience:
            return Decision.SELL
        else:
            return Decision.WAIT