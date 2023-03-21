import json
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""

class Trader:

    def __init__(self):
        self.logger = Logger()

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Version 0 attempt of submitting an algorithm for the practice round.
        Does very basic market making within the given position limits.
        Can play around with what position size tolerance and order quantity
        for a given time step is optimal.
        """
        result = {}

       


        for product in state.order_depths.keys():
            if product.upper() != "PEARLS":
                continue
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []
            if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = min(order_depth.buy_orders.keys())
                position = state.position.get(product, 0)

                spread = best_ask - best_bid
                
                if position >= -17:
                    orders.append(Order(product, best_ask-1, -3))
                if position <= 17:
                    orders.append(Order(product, best_bid+1, 3))

                
                result[product] = orders

        self.logger.flush(state, result)
        return result


