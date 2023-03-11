from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def __init__(self):
        self.iteration_num: int = 0
        self.states: List[TradingState] = []

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
    
        result = {}

        self.iteration_num += 1
        self.states.append(state)

        if self.iteration_num % 500 == 0:
            print(self.states[self.iteration_num - 1].order_depths['PEARLS'].buy_orders)
            print(self.states[self.iteration_num - 1].order_depths['PEARLS'].sell_orders)

        return result