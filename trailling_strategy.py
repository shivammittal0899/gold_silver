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


def get_stoploss_value(df, symbol, indicator, min_val, multiplier, max_val):
    df = compute_adx(df)
    df = compute_ichimoku(df, 9, 26, 52)
    df['ATR'] = ATR(df, 14)
    # Calculate RSI (14-period default)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    position = 0
    positions = kite.positions()
    pos = next((p for p in positions["net"] if p["tradingsymbol"] == symbol), None)
    if pos:
        log(f"🟢 Symbol: {pos['tradingsymbol']} | 📊 Quantity: {pos['quantity']} | 💰 Avg Price: {pos['average_price']} | 📈 P&L: {pos['pnl']}")
        if pos['quantity'] > 0:
            position = 1
        elif pos['quantity'] < 0:
            position = -1
        else:
            position = 0
    
    if indicator == "highlow":
        if position == 1:
            df['refline'] = df['Low']
        elif position == -1:
            df['refline'] = df['High']
    else:
        df['refline'] = df[indicator]

    
    stoploss_value = df['refline'].iat[-1]
    log(stoploss_value)
    
    return stoploss_value