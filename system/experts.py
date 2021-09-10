import collections
from collections.abc import Sequence
import pickle

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
        for expert in self._inner_experts:
            expert.set_data(data)

    def set_weights(self, weights: np.array = None, recursive=False):
        if hasattr(self, '_inner_experts'):
            if weights is not None:
                self._weights = weights
            else:
                self._weights = np.ones(len(self._inner_experts))
            if recursive:
                for expert in self._inner_experts:
                    expert.set_weights(None, recursive=True)
            self.normalize_weights()

    def get_weights(self):
        return self._original_weights

    def normalize_weights(self):
        if self._weights.size > 0:
            self._original_weights, self._weights = self._weights, softmax(self._weights)

    def save_weights(self, filename='weights', *, write=True):
        if hasattr(self, '_weights'):
            if isinstance(self, RuleClassExpert):
                inner = []
            else:
                inner = [exp.save_weights(write=False) for exp in self._inner_experts]
            weights = [self.get_weights(), inner]
            if write:
                pickle.dump(weights, open(filename, 'wb'))
            return weights

    def load_weights(self, filename='weights', *, weights=None):
        if weights is None:
            weights, inner = pickle.load(open(filename, 'rb'))
        else:
            weights, inner = weights
        self.set_weights(weights)
        for exp, weights in zip(self._inner_experts, inner):
            exp.load_weights(weights=weights)

    def estimate(self):
        estimations = np.array([expert.estimate() for expert in self._inner_experts])
        return estimations @ self._weights

    def update(self):
        for expert in self._inner_experts:
            expert.update()

    def show(self, indentation=0, overview=True):
        total = self.count_total_inner_experts()
        if overview:
            print(f'{" " * indentation} {total:>5}] {self.name}')
            if not isinstance(self, RuleClassExpert):
                for expert in self._inner_experts:
                    expert.show(indentation + 10, overview=overview)
            return total
        else:
            print(f'{" " * indentation} {total:>5}] {self.name}')
            for expert in self._inner_experts:
                expert.show(indentation + 10, overview=overview)

    def count_total_inner_experts(self):
        if hasattr(self, '_inner_experts'):
            return sum(expert.count_total_inner_experts() for expert in self._inner_experts)
        else:
            return 1



class PairExpert(BaseExpert):
    """Expert class for handling specific trading pair.

    Args:
        base: String. Name of base currency.
        quote: String. Name of quote currency.
    """

    def __init__(self, base: str, quote: str):
        super().__init__()
        self.base, self.quote = base, quote
        self.pair = f'{base}/{quote}'
        self.name = f'PairExpert [{self.pair}]'

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

    def get_parameters(self):
        return {'timeframe': self.timeframe}



class RuleClassExpert(BaseExpert):
    """Expert class for handling specific trading rule.

    Args:
        rule: String. Name of the rule (Example: 'MovingAverageCrossoverRule').
    """

    def __init__(self, rule: str):
        super().__init__()
        self.rule = rule
        self.name = f'RuleClassExpert [{rule}]'

    def get_parameters(self):
        return {'rule': self.rule}



class RuleExpert(BaseExpert):
    """Expert class for handling specific trading rule with specific parameters.

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
        self.name = f'RuleExpert [{self._rule.name}]'
        del self._inner_experts
        del self._weights

    def set_experts(self):
        raise AttributeError('RuleExpert cannot have inner experts')

    def set_data(self, data: data.DataMaintainer):
        self._data = data
        self._estimation_hash = None
        self._signals = []
        for indicator in self._indicators:
            indicator.set_data(data)

    def get_parameters(self):
        return {}

    def estimate(self):
        if self._estimation_hash == self._data.update_hash:
            estimation = self._last_estimation
        else:
            estimation = self._rule.decide(*self._indicators)
            self._last_estimation = estimation
            self._estimation_hash = self._data.update_hash
        self._signals.append(estimation)
        return estimation

    def update(self):
        for indicator in self._indicators:
            indicator.update()

    def show(self, indentation=0, overview=True):
        if overview:
            info = self.name
        else:
            rule_name = f'{self._rule.name}, {self._rule.get_parameters()}'
            inds_names = f'{str([(ind.name, ind.get_parameters()) for ind in self._indicators])}'
            ntrades = f'{self._estimated_ntrades:>5} trades' if self._estimated_ntrades is not None else 'unknown'
            profit = f'{self._estimated_profit:.2f} %' if self._estimated_profit is not None else 'unknown'
            info = f'{rule_name:<80} {inds_names:<100} {ntrades:>10} {profit:>10}'
        print(f'{" " * indentation} {info}')