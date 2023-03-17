from dataset import Dataset
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


class Rule:
    def __init__(
        self,
        model,
        hardcode,
        indicators,
        config,
        product,
        getOrdersFn
    ):
        self.model = model
        self.model.load_params(hardcode)

        self.dataset = Dataset(indicators, **config)

        self.product = product
        self.getOrdersFn = getOrdersFn

    def __call__(self, states):
        input = self.dataset.compute_single(states)
        output = self.model.predict(input)

        return self.product, self.getOrdersFn(output)

class Trader:
    def __init__(
        self,
        rules
    ):
        self.states = []
        self.rules = {} # hardcode rule for each product

    def run(self, state: TradingState):
        self.states.append(state)

        result = {}
        for product in state.order_depths.keys():
            rule = self.rules[product]
            orders = rule(self.states)
            result[product] = orders

        return result

