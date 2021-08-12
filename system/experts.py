import collections
from collections.abc import Sequence

import numpy as np
from scipy.special import softmax

import indicators
import rules
import data



class BaseExpert:
    """Base Expert class for decision making."""

    def __init__(self):
        self.name = 'BaseExpert'
        self._inner_experts = None
        self._weights = self._original_weights = np.array([])
        self._estimated_profit = None
        self._estimated_ntrades = None

    def set_experts(self, experts: Sequence):
        self._inner_experts = experts

    def set_data(self, data: data.DataMaintainer):
        raise NotImplementedError()

    def set_weights(self, weights: list['weights', list['inner weights']] = None):
        if hasattr(self, '_inner_experts'):
            if weights is not None:
                self._weights, inner = weights
            else:
                self._weights = np.random.normal(size=len(self._inner_experts))
                inner = [None] * len(self._inner_experts)
            for expert, weights in zip(self._inner_experts, inner):
                expert.set_weights(weights)
            self.normalize_weights()

    def get_weights(self) -> list['weights', list['inner weights']]:
        inner_weights = [expert.get_weights() for expert in self._inner_experts
                                              if hasattr(expert, '_inner_experts')]
        return [self._original_weights, inner_weights]

    def normalize_weights(self):
        if self._weights.size > 0:
            self._original_weights, self._weights = self._weights, softmax(self._weights)

    def get_parameters(self):
        raise NotImplementedError()
    
    def estimate(self):
        estimations = np.array([expert.estimate() for expert in self._inner_experts])
        return estimations @ self._weights
    
    def update(self):
        for expert in self._inner_experts:
            expert.update()

    def show(self, indentation=0, overview=True):
        print(' ' * indentation + self.name)
        if overview and hasattr(self, 'summary'):
            self.summary(indentation + 10)
        else:
            for expert in self._inner_experts:
                expert.show(indentation + 10, overview=overview)



class PairExpert(BaseExpert):
    """Expert class for handling specific trading pair.

    Args:
        base: String. Name of base currency.
        quote: String. Name of quote currency.
    """

    def __init__(self, base: str, quote: str):
        super().__init__()
        self.base, self.quote = base, quote
        self.name = f'PairExpert [{base}/{quote}]'

    def set_data(self, data: data.DataMaintainer):
        for expert in self._inner_experts:
            expert.set_data(data[expert.timeframe])

    def get_parameters(self):
        return {'base': self.base, 'quote': self.quote}



class TimeFrameExpert(BaseExpert):
    """Expert class for handling specific timeframe.

    Args:
        timeframe: String. Name of timeframe (Example: '1h').
    """

    def __init__(self, timeframe: str):
        super().__init__()
        self.timeframe = timeframe
        self.name = f'TimeFrameExpert [{timeframe}]'

    def set_data(self, data: data.DataMaintainer):
        for expert in self._inner_experts:
            expert.set_data(data)

    def get_parameters(self):
        return {'timeframe': self.timeframe}

    def summary(self, indentation):
        count = collections.Counter(expert.name for expert in self._inner_experts)
        count['Total'] = len(self._inner_experts)
        for name, count in count.items():
            print(f'{" " * indentation}{count:>5} {name}')




class RuleExpert(BaseExpert):
    """Expert class for handling specific trading rule.

    Args:
        _rule: BaseRule. Trading rule that applies to indicators.
        _indicators: Sequence of BaseIndicator. Indicators to which rule is applied.
    """

    def __init__(self, rule: rules.BaseRule, indicators: Sequence[indicators.BaseIndicator]):
        super().__init__()
        self._rule = rule
        self._indicators = indicators
        if not self._rule.compatible(*self._indicators):
            raise ValueError('Rule is incompatible with indicators')
        indicator_names = [indicator.name for indicator in self._indicators]
        self.name = f'RuleExpert [{self._rule.name}, {str(indicator_names)}]'
        del self._inner_experts
        del self._weights

    def set_experts(self):
        raise SystemError('Do not call this method')

    def set_data(self, data: data.DataMaintainer):
        for indicator in self._indicators:
            indicator.set_data(data)

    def get_parameters(self):
        return {}

    def estimate(self):
        return self._rule.decide(*self._indicators)

    def update(self):
        for indicator in self._indicators:
            indicator.update()

    def show(self, indentation=10, overview=True):
        if overview:
            name = self.name
        else:
            rule_name = f'{self._rule.name}, {self._rule.get_parameters()}'
            inds_names = f'{str([(ind.name, ind.get_parameters()) for ind in self._indicators])}'
            ntrades = f'{self._estimated_ntrades:.2f}' if self._estimated_ntrades is not None else 'unknown'
            profit = f'{self._estimated_profit:.2f} %' if self._estimated_profit is not None else 'unknown'
            name = f'{rule_name:<80} {inds_names:<100} {ntrades:>10} {profit:>10}'
        print(' ' * indentation + name)