from datamodel import TradingState, Listing, OrderDepth

def generate_states(rawdata):
    trading_states_d = {}
    for d in rawdata:
        timestamp = d['timestamp']
        product = d['product']
        buy_orders = {}
        sell_orders = {}
        # bids
        for i in range(1,4):
            price = d['bid_price_'+str(i)]
            volume = d['bid_volume_'+str(i)]
            
            buy_orders[price] = volume
        # asks
        for i in range(1,4):
            price = d['ask_price_'+str(i)]
            volume = d['ask_volume_'+str(i)]
            sell_orders[price] = volume
        
        listing = Listing(
            product,
            product,
            product
        )

        order_depth = OrderDepth(
            buy_orders,
            sell_orders,
            d['mid_price']
        )

        if timestamp not in trading_states_d:
            trading_states_d[timestamp] = TradingState(
                timestamp,
                {},
                {},
                {},
                {},
                {},
                {}
            )

        trading_states_d[timestamp].listings[product] = listing
        trading_states_d[timestamp].order_depths[product] = order_depth
    return trading_states_d
