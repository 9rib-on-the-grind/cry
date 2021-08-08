from collections.abc import Sequence

import numpy as np

import indicators
import rules
import data



class BaseExpert:
    """Base Expert class for decision making."""

    def __init__(self):
        self.name = 'BaseExpert'
        self._inner_experts = None
        self._weights = None
        self._estimated_profit = None
        self._estimated_ntrades = None

    def set_experts(self, experts: Sequence):
        self._inner_experts = experts

    def set_data(self, data: data.DataMaintainer):
        raise NotImplementedError()

    def init_weights(self):
        raise NotImplementedError()

    def get_weights(self) -> list['weights', list['inner weights']]:
        inner_weights = [expert.get_weights() for expert in self._inner_experts 
                                              if hasattr(expert, '_inner_experts')]
        return [self._weights, inner_weights]

    def set_weights(self, weights: list['weights', list['inner weights']]):
        self._weights, inner = weights
        for expert, weights in zip(self._inner_experts, inner):
            expert.set_weights(weights)
        self.normalize_weights(recursively=True)

    def normalize_weights(self, recursively=False):
        if hasattr(self, '_weights'):
            # weights should not be divided in place because of trainer fit function
            self._weights = self._weights / np.sum(self._weights)
            if recursively:
                for expert in self._inner_experts:
                    expert.normalize_weights(recursively=True)

    def get_parameters(self):
        raise NotImplementedError()
    
    def estimate(self):
        estimations = np.array([expert.estimate() for expert in self._inner_experts])
        return estimations @ self._weights
    
    def update(self):
        for expert in self._inner_experts:
            expert.update()

    def show(self, indentation=0, detailed=True):
        print(' ' * indentation + self.name)
        for expert in self._inner_experts:
            expert.show(indentation + 10, detailed=detailed)



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

    def init_weights(self):
        self._weights = np.random.uniform(0, 1, size=len(self._inner_experts))
        self.normalize_weights()
        for expert in self._inner_experts:
            expert.init_weights()

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

    def init_weights(self):
        if self._inner_experts and self._inner_experts[0]._estimated_profit is not None:
            profit = [expert._estimated_profit for expert in self._inner_experts]
            self._weights = np.array(profit)
        else:
            self._weights = np.random.uniform(0, 1, size=len(self._inner_experts))
        self.normalize_weights()

    def get_parameters(self):
        return {'timeframe': self.timeframe}



class RuleExpert(BaseExpert):
    """Expert class for handling specific trading rule.

    Args:
        indicators: List of BaseIndicator. Indicators to which rule is applied.
        rule: BaseRule. Trading rule that applies to indicators.
    """

    def __init__(self, rule: rules.BaseRule, indicators: Sequence[indicators.BaseIndicator]):
        super().__init__()
        self._rule = rule
        self._indicators = indicators
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

    def show(self, indentation=10, detailed=True):
        if detailed:
            rule_name = f'{self._rule.name}, {self._rule.get_parameters()}'
            inds_names = f'{str([(ind.name, ind.get_parameters()) for ind in self._indicators])}'
            profit = f'{self._estimated_profit:.2f} %' if self._estimated_profit is not None else 'unknown'
            name = f'{rule_name:<80} {inds_names:<100} {profit}'
        else:
            name = self.name
        print(' ' * indentation + name)