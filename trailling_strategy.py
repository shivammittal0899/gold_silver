from strategy import *
# def log2(msg):
#     with open("static/logs.txt", "a") as f:
#         f.write(f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")
TRAILLING_STOP = False
def run_trailling_strategy():
    global kite, TRAILLING_STOP
    TRAILLING_STOP = False     # reset when starting

def stop_trailling_strategy():
    global TRAILLING_STOP
    TRAILLING_STOP = True
    log("⛔ Strategy STOP requested.")


def wait_until_next_time(tf):
    """Wait until next given-minute candle + 15 seconds mark."""
    now = datetime.now() + timedelta(hours=5, minutes=30)
    log(f"{now} -- wait for {tf} minutes")
    # now = datetime.now()
    log(now)
    # Find the next minute multiple
    waittime = 0
    if tf == "5m":
        waittime = 5
    elif tf == "15m":
        waittime = 15
    elif tf == "30m":
        waittime = 30
    elif tf == "1hr":
        waittime = 60
    
    min = waittime
    next_minute = (now.minute // min + 1) * min
    # log(next_minute)
    next_time = now.replace(minute=0, second=15, microsecond=0) + timedelta(minutes=next_minute)
    # log(next_time)
    if next_minute >= 60:
        next_time = now.replace(hour=(now.hour + 1) % 24, minute=0, second=15, microsecond=0)
    wait_seconds = (next_time - now).total_seconds()
    # log(wait_seconds)
    if wait_seconds < 0:
        wait_seconds += (60*min)  # just in case of rounding errors
    log(f"⏳ Waiting {int(wait_seconds)} sec until next candle time {next_time.strftime('%H:%M:%S')}...")

    time.sleep(wait_seconds)


def get_stoploss_value(df, symbol, indicator, min_val, multiplier, max_val, position):
    df = compute_adx(df)
    df = compute_ichimoku(df, 9, 26, 52)
    df['ATR'] = ATR(df, 14)
    # Calculate RSI (14-period default)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    exchange = "MCX"
    if indicator == "highlow":
        if position == 1:
            df['refline'] = df['Low']
        elif position == -1:
            df['refline'] = df['High']
    else:
        df['refline'] = df[indicator]

    
    stoploss_value = df['refline'].iat[-1]
    
    atr = df['ATR'].iat[-1]*multiplier
    if (atr > min_val) and (atr > max_val):
        margin_val = max_val
    elif (atr > min_val):
        margin_val = atr
    else:
        margin_val = min_val
    if position == 1:
        stoploss_value = stoploss_value - margin_val
    elif position == -1:
        stoploss_value = stoploss_value + margin_val
    log(stoploss_value)
    price = df['Close'].iat[-1]
            # if stoploss_value 
    if (position == 1):
        if (sl_orderid != None):
            try:
                log(f"MSL placed: {sl_orderid} {stoploss_value}")
                modify_sl_order(sl_orderid, stoploss_value)
            except Exception as e: 
                log(f"MSL order error {e}")
                if "Trigger price" in e:
                    cancel_order(sl_orderid)
                    sl_orderid = None
                    buy_sell = "SELL"
                    quantity = qty
                    kite_app_buy_sell(exchange, symbol, buy_sell, quantity)
                    log(f"Error occured so MSL order canceled and exit long position")
        elif(sl_orderid == None) and (price > stoploss_value):
            quantity = qty
            log(f"SL placed: {sl_orderid} {stoploss_value}")
            sl_orderid = place_sl_order(symbol, "SELL", quantity, stoploss_value)
            log("SL Placed")
        elif (sl_orderid != None) and (stoploss_value == 0):
            try:
                cancel_order(sl_orderid)
            except Exception as e: 
                log(f"Stoploss cancel error {e}")
            log("SL Canceled")
            sl_orderid = None
        else:
            sl_orderid = None
    if position == -1:
        if (sl_orderid != None) and (stoploss_value != 0):
            try:
                log(f"MSL placed: {sl_orderid} {stoploss_value} start")
                modify_sl_order(sl_orderid, stoploss_value)
                log(f"SLM placed: {sl_orderid} {stoploss_value}")
            except Exception as e:
                log(f"Error - {e}")
                if "Trigger price" in e:
                    cancel_order(sl_orderid)
                    sl_orderid = None
                    buy_sell = "BUY"
                    quantity = qty
                    kite_app_buy_sell(exchange, symbol, buy_sell, quantity)
                    log(f"Error occured so SL order canceled and exit from short position")
            
        elif (sl_orderid == None) and (stoploss_value != 0):
            quantity = qty
            sl_orderid = place_sl_order(symbol , "BUY", quantity, stoploss_value)
            log(f"SL placed: {sl_orderid} {stoploss_value}")

        elif (sl_orderid != None) and (stoploss_value == 0):
            try:
                cancel_order(sl_orderid)
            except Exception as e: 
                log(f"Stoploss cancel error {e}")
            sl_orderid = None
        else:
            sl_orderid = None
    if position == 0:
        log("No positions")
    return stoploss_value