'''
Training Workflow:
1. Convert csv data to list of trading state data
2. Convert list of trading state data to training data
    xs -> compute list of indicators on different window size
        window sizes: 1, 5, 10, 50, 100, 300
        Cache values based on window sizes when possible
    ys -> whether the per item price will rise pass a profit per item
3. Train a model on this data and save the parameters as json in a file

Prediction Workflow:
1. Taking the json hardcoded values in file, hard code them into the
    the system
2. Feed the current new state data to compute the new features for 
    current iteration
3. Use the models to make predictions on the updated features
4. Based on the prediction output, implement the appropriate
    purchasing rules
'''

import numpy as np

def avg_price_indicator(states, window, products):
    ret = []
    wdata = states[-window:]
    for product in products:
        pass
    return []

class Features:
    def __init__(self, products):
        self.products = products

    def compute_features(self, states, indicators, windows):
        x = []
        for indicator in indicators:
            for window in windows:
                x.extend(indicator(states, window, self.products))

        return x
    
    def compute_gt(self, states, window, profit, shortsell=False):
        pass

class Rule:
    def __init__(
        self,
        model,
        hardcode,
        product,
        getOrdersFn,
        profit,
        fwindow,
        shortsell = False
    ):
        self.model = model
        if hardcode:
            self.model.load_params(hardcode)

        self.product = product
        self.getOrdersFn = getOrdersFn
        self.profit = profit
        self.fwindow = fwindow
        self.shortsell = shortsell

    def train(self, xs, ys):
        self.model.train(xs, ys)
        
        print("Accuracy...")
        preds = self.model.predict(xs)
        v = np.logical_and(preds, ys)
        print(np.sum(v)/len(ys))

        return self.model.get_params()
    
    def __call__(self, states):
        input = self.dataset.compute_single(states)
        output = self.model.predict(input)

        return self.product, output

class Trader:
    def __init__(
        self,
        rules
    ):
        self.states = []
        self.rules = rules

    def run(self, state):
        self.states.append(state)

        # Create the features here

        # Make predictions on whether to buy a product

        # Make predictions on whether to sell or shortsell a product

        # Market making here

        result = {}
        for rule in self.rules:
            product, orders = rule(self.states)
            result[product] = orders

        return result

