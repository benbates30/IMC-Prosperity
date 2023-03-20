import numpy as np

def wavg_bid_price(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    wsum = 0
    vsum = 0
    avg = 0
    for wi, (_, d) in enumerate(ds[::-1]):
        for p, v in d.order_depths[product].buy_orders.items():
            if not np.isnan(p):
                wsum += p*v
                vsum += v
        if wi+1 in windows:
            ret.append(wsum/vsum)
            avg += wsum/vsum

    while len(ret) < len(windows):
        ret.append(avg/len(windows))
    return ret

def wavg_ask_price(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    wsum = 0
    vsum = 0
    avg = 0
    for wi, (_, d) in enumerate(ds[::-1]):
        for p, v in d.order_depths[product].sell_orders.items():
            if not np.isnan(p):
                wsum += p*v
                vsum += v
        if wi+1 in windows:
            ret.append(wsum/vsum)
            avg += wsum/vsum
    while len(ret) < len(windows):
        ret.append(avg/len(windows))
    return ret

def avg_mid_price(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    wsum = 0
    vsum = 0
    avg = 0
    for wi, (_, d) in enumerate(ds[::-1]):
        wsum += d.order_depths[product].mid_price
        vsum += 1
        if wi+1 in windows:
            ret.append(wsum/vsum)
            avg += wsum/vsum
    while len(ret) < len(windows):
        ret.append(avg/len(windows))
    return ret

def volume_diff(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    vsum = 0
    avg = 0
    for wi, (_, d) in enumerate(ds[::-1]):
        for p, v in d.order_depths[product].buy_orders.items():
            if not np.isnan(p):
                vsum -= v
        for p, v in d.order_depths[product].sell_orders.items():
            if not np.isnan(p):
                vsum += v
        if wi+1 in windows:
            ret.append(vsum)
            avg = vsum
    while len(ret) < len(windows):
        ret.append(0)
    return ret

def best_prices(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    asks = []
    bids = []
    for wi, (_, d) in enumerate(ds[::-1]):
        for p, _ in d.order_depths[product].buy_orders.items():
            if not np.isnan(p):
                bids.append(p)
        for p, _ in d.order_depths[product].sell_orders.items():
            if not np.isnan(p):
                asks.append(p)
        if wi+1 in windows:
            ret.append(max(bids))
            ret.append(min(asks))
    while len(ret) < len(windows)*2:
        ret.append(0)

    return ret