import numpy as np
import pandas as pd
from typing import Dict, List
from datamodel import TradingState

def criterion(data, product, window, profit, shortsell=False):
    """
    Creates a ground truth value for a single type of product 
    over a windowed period if it can be bought/sold for profit 

    return 1 if the product can be bought/sold now to make atleast 
    profit amound within the next window number of trading states

    return 0 otherwise
    """
    if len(data) < window:
        return 0
    wdata = data[:window]

class Dataset:
    def __init__(
        self,
        indicators,
        **config,
    ):
        self.indicators = indicators
        self.config = config

    def compute_single(self, data : List[TradingState]):
        ret = []
        for indicator in self.indicators:
            ret.append(indicator(data, self.config))
        return ret

    def compute_many(self, data : List[TradingState]):
        ret = []
        for i in range(len(data)):
            wdata = data[:i]
            ret.append(self.compute_single(wdata))
        return ret
    
    def compute_gt(
        self, 
        data : List[TradingState],
        product, 
        profit,
        window,
        shortsell=False
    ):
        ret = []
        for i in range(len(data)):
            wdata = data[i:]
            ret.append(
                criterion(
                    wdata,
                    product,
                    window,
                    profit,
                    shortsell=shortsell
                )
            )
        return ret