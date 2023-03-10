from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Version 0 attempt of submitting an algorithm for the practice round.

        Does very basic market making within the given position limits.
        Can play around with what position size tolerance and order quantity
        for a given time step is optimal.

        """
        result = {}

       


        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []
            if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = min(order_depth.buy_orders.keys())
                position = state.position.get(product, 0)

                spread = best_ask - best_bid
                if spread > 2:
                    our_ask = best_ask -0.1
                    our_bid = best_bid+0.1
                
                    # our_ask = best_ask*0.9999

                    if abs(position) < 6:
                        print("SELL 3", product + "x", our_ask)
                        orders.append(Order(product, our_ask, -3))

                        print("BUY 3", product + "x", our_bid)
                        orders.append(Order(product, our_bid, 3))
                    elif abs(position) < 8:
                        print("SELL 2", product + "x", our_ask)
                        orders.append(Order(product, our_ask, -2))

                        print("BUY 2", product + "x", our_bid)
                        orders.append(Order(product, our_bid, 2))
                        
                    else:
                        if position >= -11:
                            print("SELL", product + "x", our_ask)
                            orders.append(Order(product, our_ask, -1))
                        # our_bid = best_bid*1.0001
                        if position <= 11:
                            print("BUY", product + "x", our_bid)
                            orders.append(Order(product, our_bid, 1))

                result[product] = orders







            

        return result
            
