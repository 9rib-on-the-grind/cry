from collections.abc import Sequence

import numpy as np

import indicators
import rules



class BaseExpert:
    """Base Expert class for decision making."""

    def __init__(self):
        self.name = 'Base Expert'
        self._inner_experts = None
        self._weights = np.ones((len(self._inner_experts), 1))

    def set_experts(self, experts: Sequence):
        self._inner_experts = experts
    
    def estimate(self):
        estimations = np.array([expert.estimate() for expert in self._inner_experts])
        # return estimations @ self._weights
        return np.mean(estimations)
    
    def update(self):
        for expert in self._inner_experts:
            expert.update()

    def show(self, indentation=0):
        print(' ' * indentation + self.name)
        for expert in self._inner_experts:
            expert.show(indentation + 10)


class PairExpert(BaseExpert):
    """Expert class for handling specific trading pair.

    Args:
        base: String. Name of base currency.
        quote: String. Name of quote currency.
    """

    def __init__(self, base: str, quote: str):
        self.name = f'{base}/{quote} Expert'



class TimeFrameExpert(BaseExpert):
    """Expert class for handling specific timeframe.

    Args:
        timeframe: String. Name of timeframe (Example: '1h').
    """

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.name = f'{timeframe} Expert'



class RuleExpert(BaseExpert):
    """Expert class for handling specific trading rule.

    Args:
        indicators: List of BaseIndicator. Indicators to which rule is applied.
        rule: BaseRule. Trading rule that applies to indicators.
    """

    def __init__(self, indicators: Sequence[indicators.BaseIndicator], rule: rules.BaseRule):
        self._indicators = indicators
        self._rule = rule
        self.name = f'[{self._rule.name}] {str([indicator.name for indicator in self._indicators])}'

    def set_experts(self):
        raise SystemError('Do not call this method')

    def estimate(self):
        return self._rule.decide(*self._indicators)

    def update(self):
        for indicator in self._indicators:
            indicator.update()

    def show(self, indentation):
        print(' ' * indentation + self.name)