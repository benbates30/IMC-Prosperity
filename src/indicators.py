import statistics
from typing import Dict, List
from datamodel import TradingState

class Average:
    def __init__(
        self, 
        windowsize : int,

    ):
        self.windowsize = windowsize

    def __call__(self, data : List[TradingState], config) -> List[float]:
        if len(data) < self.windowsize:
            return 0

        window = data[-self.windowsize:]

        