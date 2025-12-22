"""
Ichimoku + Volume + OI Participation Strategy with Capital Backtest
- Uses starting capital = ₹2,000,000
- Outputs: trades CSV, performance metrics with equity curve in INR
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
from strategy_fun import *
from kiteconnect import KiteConnect
import time


STOP_FLAG = False      # global controller
kite = None

# Place market sell (to exit/short) — change transaction_type to "SELL"
def kite_app_buy_sell(exchange, tradingsymbol, buy_sell, quantity):
    # Place market buy
    # print(time.now())
    order_id = kite.place_order(
        variety = kite.VARIETY_REGULAR,      # regular order
        exchange = exchange,
        tradingsymbol = tradingsymbol,
        transaction_type = buy_sell,            # or "SELL"
        quantity = quantity,
        order_type = "MARKET",
        product = "NRML",                    # NRML for futures (or MIS for intraday margin intraday)
        validity = "DAY"
    )
    print(f"Placed {buy_sell} of {tradingsymbol} quantity is {quantity} order id: {order_id}")
    return order_id

def fetch_with_retry(symbol, interval, retries=3, delay=5):
    for attempt in range(retries):
        try:
            time_correction = timedelta(hours=5, minutes=30)
            # time_correction = 0
            time_now = datetime.now() + time_correction
            time_delay = time_now - timedelta(days=5)
            print(time_delay, time_now)
            instrument = kite.ltp(f"MCX:{symbol}")[f"MCX:{symbol}"]['instrument_token']
            data = kite.historical_data(
                instrument_token=instrument,
                from_date=time_delay,
                to_date=time_now,
                interval=interval,
                oi = True
            )
            df = pd.DataFrame(data)
            print(df['date'].iloc[-1])
            return df
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise



def backtest_with_capital(p):
    # ==============================
    # 🔧 C`ONFIGURATION
    # ==============================

    capital = p['starting_capital']
    trades = []
    position = 0
    entry_price = None
    entry_time = None
    position_size = 0
    total_diff = 0
    # exchange = "MCX"
    symbol = "GOLDM26JANFUT"   # example instrument
    tradingsymbol = "GOLDM26JANFUT"
    interval = "15minute"
    days = 5
    qty = p['quantity']
    print()
    log(qty)
    sl_orderid = None
    while  not STOP_FLAG:
        print(STOP_FLAG)
        # try:
        now = datetime.now() + timedelta(hours=5, minutes=30)
        # now = datetime.now() 
        log(f'Present Time: {now}')
        market_open  = (now.hour > 9) or (now.hour == 9 and now.minute >= 9)
        # market_open  = (now.hour >= 8)
        market_close = (now.hour > 23) or (now.hour == 23 and now.minute >= 55)


        if not (market_open and not market_close):
            print("🕘 MCX Market Closed — sleeping...")
            log("🕘 MCX Market Closed — sleeping...")
            wait_until_next_15min_plus30()
            # time.sleep(600)
            continue

        df = fetch_with_retry(symbol, interval)

        df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','oi':'OI'}, inplace=True)
        log(f"✅ Data fetched: {len(df)} bars | Last candle at {df['date'].iloc[-1]}")


        df = normalize(df)
        log_df(df.tail(5), title="Last 5 Candles")

        df = compute_ichimoku(df, p['tenkan'], p['kijun'], p['senkou_b'])
        df = compute_adx(df)
        df['ATR'] = ATR(df, 14)
        # Calculate RSI (14-period default)
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        log(df.columns)
        print(f"✅ Data fetched: {len(df)} bars | Last candle at {df.index[-1]}")
        df = generate_signals(df, p)

        i = -2
        cur = df.iloc[i]
        nxt = df.iloc[i+1]
        t_next = df.index[i+1]
        exchange = "MCX"
        # Fetch all open positions
        positions = kite.positions()

        # Filter the position list
        pos = next((p for p in positions["net"] if p["tradingsymbol"] == symbol), None)
        position1 = position
        if pos:
            log(f"🟢 Symbol: {pos['tradingsymbol']}")
            log(f"📊 Quantity: {pos['quantity']}")
            log(f"💰 Avg Price: {pos['average_price']}")
            log(f"📈 P&L: {pos['pnl']}")
            if pos['quantity'] > 0:
                position1 = 1
                entry_price = pos['average_price']
            elif pos['quantity'] < 0:
                position1 = -1
                entry_price = pos['average_price']
            else:
                position1 = 0

            if position1 != position:
                position = position1
        else:
            log(f"⚪ No open position in {symbol}")
            # position = 0
        log(position)
        if position == 0:
            sl_orderid = None
    # df['entry_long'] | df['entry_pullback_long'] | df['entry_kumo_break_long'] | df['entry_sideways_break_long'] | df['entry_gap_long'] | df['entry_long_price'] | df['entry_long_price_cloud']
    # df['entry_short'] | df['entry_pullback_short'] | df['entry_kumo_break_short'] | df['entry_sideways_break_short'] | df['entry_gap_short'] | df['entry_short_price'] | df['entry_short_price_cloud']
    # df['exit_long'] | df['exit_long_below_cloud'] | df['exit_long_tkcross'] | df['exit_long_price_cloud'] | df['exit_long_tenkan'] | df['exit_long_kijun'] | df['trailing_stop_long']
    # df['exit_short'] | df['exit_short_above_cloud'] | df['exit_short_tkcross'] | df['exit_short_price_cloud'] | df['exit_short_tenkan'] | df['exit_short_kijun'] | df['trailing_stop_short']
        print(f"Time: {df['datetime'].iloc[i]}, Market Regime: {df['Market_Regime'].iloc[i]}, open: {df['Open'].iloc[i]}, close: {df['Close'].iloc[i]}")

        # print(f"{cur['entry_long']} | {cur['entry_pullback_long']} | {cur['entry_kumo_break_long']} | {cur['entry_sideways_break_long']} | {cur['entry_gap_long']} | {cur['entry_long_price']} | {cur['entry_long_price_cloud']}")
        # print(f"{cur['entry_short']} | {cur['entry_pullback_short']} | {cur['entry_kumo_break_short']} | {cur['entry_sideways_break_short']} | {cur['entry_gap_short']} | {cur['entry_short_price']} | {cur['entry_short_price_cloud']}")
        # print(f"{cur['exit_long']} | {cur['exit_long_below_cloud']} | {cur['exit_long_tkcross']} | {cur['exit_long_price_cloud']} | {cur['exit_long_tenkan']} | {cur['exit_long_kijun']} | {cur['trailing_stop_long']}")
        # print(f"{cur['exit_short']} | {cur['exit_short_above_cloud']} | {cur['exit_short_tkcross']} | {cur['exit_short_price_cloud']} | {cur['exit_short_tenkan']} | {cur['exit_short_kijun']} | {cur['trailing_stop_short']}")
        
        
        log(f"Time: {df['datetime'].iloc[i]}, Market Regime: {df['Market_Regime'].iloc[i]}, open: {df['Open'].iloc[i]}, close: {df['Close'].iloc[i]}")

        # log(f"{cur['entry_long']} | {cur['entry_pullback_long']} | {cur['entry_kumo_break_long']} | {cur['entry_sideways_break_long']} | {cur['entry_gap_long']} | {cur['entry_long_price']} | {cur['entry_long_price_cloud']}")
        # log(f"{cur['entry_short']} | {cur['entry_pullback_short']} | {cur['entry_kumo_break_short']} | {cur['entry_sideways_break_short']} | {cur['entry_gap_short']} | {cur['entry_short_price']} | {cur['entry_short_price_cloud']}")
        # log(f"{cur['exit_long']} | {cur['exit_long_below_cloud']} | {cur['exit_long_tkcross']} | {cur['exit_long_price_cloud']} | {cur['exit_long_tenkan']} | {cur['exit_long_kijun']} | {cur['trailing_stop_long']}")
        # log(f"{cur['exit_short']} | {cur['exit_short_above_cloud']} | {cur['exit_short_tkcross']} | {cur['exit_short_price_cloud']} | {cur['exit_short_tenkan']} | {cur['exit_short_kijun']} | {cur['trailing_stop_short']}")

        # if position == 1 and cur['exit_long_final']:
        if position == 1 and (cur['exit_long_final']):
            exit_price = nxt['Open'] - p['slippage']
            price_diff = exit_price - entry_price
            total_diff +=price_diff
            pnl_price_units = price_diff * position_size
            pnl_inr = pnl_price_units * p['contract_value'] - p['commission']
            capital += pnl_inr
            trades.append({
                'entry_time': entry_time, 'exit_time': t_next, 'side': 'long',
                'entry_price': entry_price, 'exit_price': exit_price, 'difference': price_diff, 'total_diff': total_diff,
                'contracts': position_size, 'pnl_inr': pnl_inr, 'capital': capital
            })
            position = 0
            buy_sell = "SELL"
            quantity = qty
            kite_app_buy_sell(exchange, tradingsymbol, buy_sell, quantity)
            log(f"Exit long: (Entry price - {entry_price}), (Exit price - {exit_price}), (PnL diff -- {price_diff})")

        # elif position == -1 and (cur['final_exit_short']):
        elif position == -1 and (cur['final_exit_short']):
            exit_price = nxt['Open'] + p['slippage']
            price_diff = entry_price - exit_price
            total_diff +=price_diff
            pnl_price_units = price_diff * position_size
            pnl_inr = pnl_price_units * p['contract_value'] - p['commission']
            capital += pnl_inr
            trades.append({
                'entry_time': entry_time, 'exit_time': t_next, 'side': 'short',
                'entry_price': entry_price, 'exit_price': exit_price, 'difference': price_diff, 'total_diff': total_diff,
                'contracts': position_size, 'pnl_inr': pnl_inr, 'capital': capital
            })
            position = 0
            buy_sell = "BUY"
            quantity = qty
            kite_app_buy_sell(exchange, tradingsymbol, buy_sell, quantity)
            log(f"Exit short: (Entry price - {entry_price}), (Exit price - {exit_price}), (PnL diff -- {price_diff})")

            # print(capital)
        # Entry logic
        if position == 0:
            if cur['final_entry_long']:
                # print(capital)
                risk_amount = p['risk_per_trade'] * capital
                # position_size = math.floor(risk_amount / (df['Close']))
                position_size = np.floor(risk_amount / (nxt['Open']))
                if position_size <= 0: position_size = 1
                entry_price = nxt['Open'] + p['slippage']
                entry_time = t_next
                position = 1
                buy_sell = "BUY"
                quantity = qty
                kite_app_buy_sell(exchange, tradingsymbol, buy_sell, quantity)
                print("Buy price. ",entry_time, entry_price)
                log(f"Buy price. ,{entry_time}, {entry_price}")
                # print(position_size)
            elif cur['final_entry_short']:
                # print(capital)
                risk_amount = p['risk_per_trade'] * capital
                # position_size = math.floor(risk_amount / (df['Close'].std() or 1))
                position_size = np.floor(risk_amount / (nxt['Open']))
                if position_size <= 0: position_size = 1
                entry_price = nxt['Open'] - p['slippage']
                entry_time = t_next
                position = -1
                buy_sell = "SELL"
                quantity = qty
                kite_app_buy_sell(exchange, tradingsymbol, buy_sell, quantity)
                print("Sell price. ",entry_time, entry_price)
                log(f"Sell price. ,{entry_time}, {entry_price}")
        if position == 1:
            stoploss_val = stopless_point(cur, position)
            if (sl_orderid != None) and (stoploss_val != 0):
                kite.modify_order(
                                variety=kite.VARIETY_REGULAR,
                                order_id=sl_orderid,
                                trigger_price=stoploss_val                         # new SL trigger
                            )
                
                log(f"SL placed: {order_id} {stoploss_val}")
            elif (sl_orderid == None) and (stoploss_val != 0):
                quantity = qty
                order_id = kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=kite.EXCHANGE_MCX,                # 🔴 MCX
                                tradingsymbol=tradingsymbol,           # ⚠️ Correct MCX symbol
                                transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity=quantity,                                # MCX lot size
                                product=kite.PRODUCT_NRML,                 # 🔴 NRML for futures
                                order_type=kite.ORDER_TYPE_SLM,
                                trigger_price=stoploss_val,                        # initial SL trigger
                                validity=kite.VALIDITY_DAY
                            )
                sl_orderid = order_id
                log(f"SL placed: {order_id} {stoploss_val}")
            elif (sl_orderid != None) and (stoploss_val == 0):
                kite.cancel_order(
                                variety=kite.VARIETY_REGULAR,
                                order_id=sl_orderid
                            )
                sl_orderid = None
            else:
                sl_orderid = None


        elif position == -1:
            stoploss_val = stopless_point_short(cur, position)
            if (sl_orderid != None) and (stoploss_val != 0):
                kite.modify_order(
                                variety=kite.VARIETY_REGULAR,
                                order_id=sl_orderid,
                                trigger_price=stoploss_val                         # new SL trigger
                            )
                
                log(f"SL placed: {order_id} {stoploss_val}")
            elif (sl_orderid == None) and (stoploss_val != 0):
                quantity = qty
                order_id = kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=kite.EXCHANGE_MCX,                # 🔴 MCX
                                tradingsymbol=tradingsymbol,           # ⚠️ Correct MCX symbol
                                transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity=quantity,                                # MCX lot size
                                product=kite.PRODUCT_NRML,                 # 🔴 NRML for futures
                                order_type=kite.ORDER_TYPE_SLM,
                                trigger_price=stoploss_val,                        # initial SL trigger
                                validity=kite.VALIDITY_DAY
                            )
                sl_orderid = order_id
                log(f"SL placed: {order_id} {stoploss_val}")
            elif (sl_orderid != None) and (stoploss_val == 0):
                kite.cancel_order(
                                variety=kite.VARIETY_REGULAR,
                                order_id=sl_orderid
                            )
                sl_orderid = None
            else:
                sl_orderid = None
        else: 
            if sl_orderid != None:
                kite.cancel_order(
                                variety=kite.VARIETY_REGULAR,
                                order_id=sl_orderid
                            )
            sl_orderid = None
        # except Exception as e:
        #     print("⚠️ Error:", e)
        wait_until_next_15min_plus30()


def log(msg):
    with open("static/logs.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")

def log_df(df, title="DataFrame"):
    text = df.to_string()
    log(f"\n----- {title} -----\n{text}\n---------------------\n")

def stop_strategy():
    global STOP_FLAG
    STOP_FLAG = True
    log("⛔ Strategy STOP requested.")

def run_strategy(API_KEY, ACCESS_TOKEN, params):
    """
    params = {
        'quantity': 10,
        'other_params': ...
    }
    """
    global kite, STOP_FLAG
    STOP_FLAG = False     # reset when starting

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    quantity = params["quantity"]
    log(f"🚀 Strategy Started | Qty = {quantity}")
    log(f"🚀 Strategy Started | Param = {params}")

    # while not STOP_FLAG:
    try:
        # 🔥 your real function:
        backtest_with_capital(params)    # includes your logic
    except Exception as e:
        log(f"⚠ Error: {str(e)}")
    # time.sleep(15)

    log("🟢 Strategy stopped successfully.")