from dataset import Dataset

class Rule:
    def __init__(
        self,
        model,
        hardcode,
        indicators,
        config,
        product,
        getOrdersFn,
        profit,
        fwindow,
        shortsell = False
    ):
        self.model = model

        if hardcode:
            self.model.load_params(hardcode)

        self.dataset = Dataset(indicators, **config)

        self.product = product
        self.getOrdersFn = getOrdersFn
        self.profit = profit
        self.fwindow = fwindow
        self.shortsell = shortsell

    def train(self, states):
        xs = self.dataset.compute_many(states)
        ys = self.dataset.compute_gt(
            states,
            self.product,
            self.profit,
            self.fwindow,
            shortsell=self.shortsell
        )

        self.model.train(xs, ys)
        return self.model.get_params()



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
        self.rules = rules

    def run(self, state):
        self.states.append(state)

        result = {}
        for rule in self.rules:
            product, orders = rule(self.states)
            result[product] = orders

        return result

