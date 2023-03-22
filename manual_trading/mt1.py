#brute force, keep track of highest trade and path
'''
idea: to maximize profits, want to trade all of your currency in each trade.
Why: since trade rates are stationary (same for every iter), if we have a trading strategy and use
x < y dollars on this strat, we could've gotten trading_strat(y-x) more dollars had we spent all y dollars
'''

class ManualTrade1:
  def __init__(self):
    self.graph = {
      "pizza": {
        "wasabi": 0.5,
        "snowball": 1.45,
        "shells": 0.75
      },
      "wasabi": {
        "pizza": 1.95,
        "snowball": 3.1,
        "shells": 1.49
      },
      "snowball": {
        "pizza": 0.67,
        "wasabi": 0.31,
        "shells": 0.48
      },
      "shells": {
        "pizza": 1.34,
        "wasabi": 0.64,
        "snowball": 1.98
      }
    }
    self.starting_amt = 2e6
    self.max_earnings = self.starting_amt
    self.best_trades = []

  def dfs(self, path, cash):
    last_cur = path[-1]   #current currency
    if len(path) == 5:
      #on last trade, must trade back to shells
      if last_cur != "shells":
        new_cash = cash * self.graph[last_cur]["shells"]
        path.append("shells")
        if new_cash > self.max_earnings:
          self.max_earnings = new_cash
          self.best_trades = path
    else:
      #try all options for trades, check for optimization condition
      for cur in self.graph[last_cur].keys():
        new_cash = cash * self.graph[last_cur][cur]
        new_path = path + [cur]
        if cur == "shells" and new_cash > self.max_earnings:
          self.max_earnings = new_cash
          self.best_trades = new_path
        #branch on this trade path
        self.dfs(new_path, new_cash)

  def find_optimal_trade(self):
    #execute dfs
    self.dfs(["shells"], self.starting_amt)
    #print results
    print("Maximum obtainable amount after trades: ", self.max_earnings)
    print("Optimal Currency Path: ", self.best_trades)
    print("\nTrading Strategy: ")
    for i in range(len(self.best_trades)-1):
      c1, c2 = self.best_trades[i], self.best_trades[i+1]
      print(c1, " -> ", c2)


if __name__ == '__main__':
  mt1 = ManualTrade1()
  mt1.find_optimal_trade()
