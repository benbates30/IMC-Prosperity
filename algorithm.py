import json
import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

"""
Right now we may lose out on money when the best bid (ask) has a volume less than the position limit.
In that case, we could look at the next best bid (ask).
If it is still above our fair price, then we can place an order for that bid (ask).
Then check the third best bid (ask) and so on until the position limit is hit.

In order to do this we can hard code the position 
"""
class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:

        result = {}

        for product in state.order_depths.keys():

            if product == 'BANANAS' and not (450 < state.timestamp // 100 < 1150) :

                if 0 <= state.timestamp // 100 <= 450:
                    fair_price = 4950
                elif 1150 <= state.timestamp // 100 <= 2000:
                    fair_price = 4930

                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []

                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if best_ask < fair_price:
                        print('BUY', str(-best_ask_volume) + 'x', best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > fair_price:
                        print('SELL', str(best_bid_volume) + 'x', best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))
                
                result[product] = orders
            
            if product == 'PEARLS':
                
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []

                fair_price = 10000

                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if best_ask < fair_price:
                        print('BUY', str(-best_ask_volume) + 'x', best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > fair_price:
                        print('SELL', str(best_bid_volume) + 'x', best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                result[product] = orders

        return result