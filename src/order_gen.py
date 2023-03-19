from datamodel import TradingState, Trade, Order
from typing import List, Dict

class OrderGen():
    r"""
    Class to create orders for a product given inputs on buy/sell
    willingess and trading state info (e.g. position size)

    """
    def __init__(self, product):
        self.product = product

    def __call__(self, state: TradingState) -> List[Order]:
        pass
