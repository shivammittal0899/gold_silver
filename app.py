from flask import Flask, redirect, request, render_template, url_for, jsonify
from kiteconnect import KiteConnect
import os
import threading
import time
# from strategy import run_strategy, stop_strategy
from strategy import *
# from trailling_strategy import run_trailling_strategy, stop_trailling_strategy
import uuid
from datetime import datetime, timedelta
from trailling_strategy import *
from technical_analysis import *
from fund_analysis import *
import pandas as pd
import yfinance as yf
from threading import Lock
import yfinance as yf
app = Flask(__name__)
# api_key = "0qw10pvn638g9jid"
# api_secret = "8bbev51ab3ov4jfkq0ddhmsw1itviexc"
API_KEY = "0qw10pvn638g9jid"
API_SECRET = "6przxtyeeoi9jtvyx76qga4hrv7q86qr"
REDIRECT_URL = "http://localhost:8000/callback"

kite = KiteConnect(api_key=API_KEY)
LOGIN_URL = kite.login_url()
STRATEGY_RUNNING = False
TRAILLING_STRATEGY = False
TRAILING_CONFIGS = []


import sqlite3
def init_watchlist_db():
    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlist_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        watchlist_id INTEGER,
        symbol TEXT,
        UNIQUE(watchlist_id, symbol),
        FOREIGN KEY(watchlist_id) REFERENCES watchlists(id)
    )
    """)

    conn.commit()
    conn.close()
# ---------------------- SAVE TOKEN ----------------------
def save_access_token(token):
    with open("access_token.txt", "w") as f:
        f.write(token)

# ---------------------- READ TOKEN ----------------------
def read_access_token():
    if os.path.exists("access_token.txt"):
        with open("access_token.txt") as f:
            return f.read().strip()
    return None

# ---------------------- LOGIN PAGE ----------------------
@app.route("/")
def index():
    return render_template("login.html", login_url=LOGIN_URL)

# ---------------------- CALLBACK FROM KITE ----------------------
@app.route("/callback")
def login_callback():
    request_token = request.args.get("request_token")
    if not request_token:
        return "Error: Missing request_token."

    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        kite.set_access_token(access_token)
        save_access_token(access_token)
        return redirect(url_for("dashboard"))
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------- DASHBOARD PAGE ----------------------
@app.route("/dashboard")
def dashboard():
    token = read_access_token()
    quantity = read_quantity("quantity")
    indicator = read_quantity("indicator")
    min_val = read_quantity("min_val")
    max_val = read_quantity("max_val")
    multiplier = read_quantity("multiplier")
    return render_template("dashboard.html", token=token, quantity=quantity, indicator=indicator, min_val=min_val, max_val=max_val, multiplier=multiplier, trailing_configs=TRAILING_CONFIGS)

# # ---------------------- STRATEGY EXECUTOR ----------------------
# def run_my_strategy(options):
#     print("Strategy Started with options:", options)
#     while True:
#         # your trading logic here
#         print("Running strategy loop...")
#         time.sleep(5)

# ---------------------- START BUTTON ENDPOINT ----------------------
# ----------------------------
# START STRATEGY
# ----------------------------
@app.route("/start", methods=["POST"])
def start():
    global STRATEGY_RUNNING
    access_token = read_access_token()
    quantity = int(request.form.get("quantity"))
    save_quantity(quantity, "quantity")    # <---- SAVE IT HERE
    params = {
        # Ichimoku
        'tenkan': 9,
        'kijun': 26,
        'senkou_b': 52,
        # Participation filters
        'vol_ma_window': 50,
        'oi_ma_window': 50,
        # Capital & risk
        'starting_capital': 5000000,   # ₹ 20 lakh
        'risk_per_trade': 1,          # 2% of capital per trade
        'contract_value': 10,           # ₹ per 1 price point (change this if 1 tick = ₹100)
        # Execution / costs
        'slippage': 0.0,
        'commission': 20,                # ₹50 per side per trade
        'verbose': True,
        "quantity": quantity
    }
    # mark running
    STRATEGY_RUNNING = True
    

    thread = threading.Thread(target=run_strategy, args=(API_KEY, access_token, params))
    thread.daemon = True
    thread.start()

    return redirect("/dashboard")

@app.route("/recent_logs")
def recent_logs():
    try:
        with open("static/logs.txt") as f:
            lines = f.readlines()
            return "<br>".join(lines[-50:])  # only last 20 lines
    except:
        return "No logs available."


# ----------------------------
# STOP STRATEGY
# ----------------------------
@app.route("/stop")
def stop():
    global STRATEGY_RUNNING
    stop_strategy()
    STRATEGY_RUNNING = False
    return redirect("/dashboard")

def log1(msg):
    with open("static/logs.txt", "a") as f:
        f.write(f"{msg}\n")

@app.route('/start_trailing', methods=['POST'])
def start_trailing():
    global TRAILLING_STRATEGY
    indicator = request.form.get('indicator')
    min_val = int(request.form.get('min'))
    multiplier = float(request.form.get('multiplier'))
    max_val = int(request.form.get('max'))
    save_quantity(indicator, "indicator") 
    save_quantity(min_val, "min_val") 
    save_quantity(multiplier, "multiplier") 
    save_quantity(max_val, "max_val") 
    log1(f"{indicator}, {min_val}, {multiplier}, {max_val}")
    TRAILLING_STRATEGY = True
    access_token = read_access_token()
    thread_t = threading.Thread(target=run_trailling_strategy, args=(API_KEY, access_token))
    thread_t.daemon = True
    thread_t.start()
    # Your trailing logic here

    return redirect('/dashboard')
    

@app.route('/stop_trailing')
def stop_trailing():
    log1("Trailing stopped")
    
    # Stop logic here

    return redirect('/dashboard')





import threading
import time

import json


def init_db():
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS trailing (
        id TEXT PRIMARY KEY,
        instrument TEXT,
        indicator TEXT,
        timeframe TEXT,
        qty INTEGER,
        min INTEGER,
        multiplier REAL,
        max INTEGER,
        running INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()
def init_live_sl_db():
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS live_sl (
        task_id TEXT PRIMARY KEY,
        symbol TEXT,
        stoploss REAL,
        position INTEGER,
        qty INTEGER,
        updated_at TEXT
    )
    """)

    conn.commit()
    conn.close()

init_live_sl_db()

def save_live_sl(task_id, symbol, stoploss, position, qty):
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO live_sl
        (task_id, symbol, stoploss, position, qty, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        task_id,
        symbol,
        stoploss,
        position,
        qty,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def delete_live_sl(task_id):
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("DELETE FROM live_sl WHERE task_id=?", (task_id,))
    conn.commit()
    conn.close()


def load_live_sl_from_db():
    global LIVE_SL

    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    rows = c.execute("SELECT * FROM live_sl").fetchall()
    conn.close()

    with sl_lock:
        for r in rows:
            LIVE_SL[r[0]] = {
                "symbol": r[1],
                "stoploss": r[2],
                "position": r[3],
                "qty": r[4]
            }

    log1(f"🔁 Restored {len(rows)} SL from DB")




@app.route('/download_instruments')
def download_instruments():
    try:
        access_token = read_access_token()
        kite.set_access_token(access_token)

        # instruments = kite.instruments("NSE")
        instruments = kite.instruments("MCX")
        data = [
            {
                "tradingsymbol": i["tradingsymbol"],
                "name": i["name"],
                "instrument_token": i["instrument_token"]
            }
            for i in instruments
            if (i["name"] in ["GOLDM", "SILVERM"] and
                i["instrument_type"] == "FUT")
        ]
        log1(f"Saved {len(data)} instruments")
        # Save to file
        with open("instruments.json", "w") as f:
            json.dump(data, f)

        return jsonify({"status": "saved", "count": len(data)})

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/get_saved_instruments')
def get_saved_instruments():
    try:
        with open("instruments.json") as f:
            data = json.load(f)

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)})

import json


INSTRUMENT_MAP = {}
TRAILING_THREADS = {}
SL_ORDERS = {}   # task_id → order_id
LIVE_SL = {}

WS_RUNNING = False
WS_THREAD = None
KWS = None
ws_lock = Lock()
threads_lock = Lock()
orders_lock = Lock()

def load_instruments_once():
    global INSTRUMENT_MAP
    with open("instruments.json") as f:
        data = json.load(f)

    INSTRUMENT_MAP = {
        i["tradingsymbol"]: i["instrument_token"]
        for i in data
    }

load_instruments_once()

def stop_task(task_id, reason):
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "UPDATE trailing SET running=0 WHERE id=?",
        (task_id,)
    )
    conn.commit()
    conn.close()
    log1(f"[{task_id}] STOPPED: {reason}")

def fetch_with_retry_token(symbol, token, interval, kite, period = 30, retries=3, delay=5):
    for attempt in range(retries):
        try:
            time_correction = timedelta(hours=5, minutes=30)
            # time_correction = 0
            time_now = datetime.now() + time_correction
            time_delay = time_now - timedelta(days=period)
            data = kite.historical_data(
                instrument_token=token,
                from_date=time_delay,
                to_date=time_now,
                interval=interval,
                oi = True
            )
            df = pd.DataFrame(data)
            # log1(df['date'].iloc[-1])
            return df
        except Exception as e:
            log1(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

from kiteconnect import KiteTicker

def start_ws_if_needed():
    global WS_RUNNING, WS_THREAD
    log1("ws will start here")
    with ws_lock:
        if WS_RUNNING:
            return
        WS_RUNNING = True
        access_token = read_access_token()

        def run_ws():
            try:
                kite_local = KiteConnect(api_key=API_KEY)
                kite_local.set_access_token(access_token)

                start_ws(API_KEY, access_token, kite_local)
                log1("ws will started here ------------")
            except Exception as e:
                log1(f"WS error: {e}")
            finally:
                WS_RUNNING = False

        WS_THREAD = threading.Thread(target=run_ws)
        WS_THREAD.daemon = True
        WS_THREAD.start()

        log1("🚀 WebSocket started")


def stop_ws_if_idle():
    global WS_RUNNING, KWS

    with ws_lock, threads_lock:
        active = [
            t for t in TRAILING_THREADS.values()
            if t.is_alive()
        ]

        if len(active) == 0:
            if KWS:
                try:
                    KWS.close()   # ✅ ACTUAL STOP
                    log1("🛑 WebSocket closed")
                except Exception as e:
                    log1(f"WS close error: {e}")

            WS_RUNNING = False
            KWS = None
            log1("🛑 WebSocket stopping (no active threads)")

def start_ws(api_key, access_token, kite):

    global KWS

    kws = KiteTicker(api_key, access_token)
    KWS = kws   # ✅ STORE INSTANCE

    def on_connect(ws, response):
        print("✅ WebSocket Connected")

        tokens = list(set(INSTRUMENT_MAP.values()))
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_LTP, tokens)

    def on_ticks(ws, ticks):
        # log1("in on_ticks loop")
        for tick in ticks:
            token = tick["instrument_token"]
            ltp = tick["last_price"]

            with sl_lock:
                items = list(LIVE_SL.items())  # copy for safe iteration

            for task_id, data in items:
                if INSTRUMENT_MAP.get(data["symbol"]) != token:
                    continue

                sl = data["stoploss"]
                position = data["position"]
                qty = data["qty"]
                symbol = data["symbol"]
                # log1(f"{data} tick data ------------- {ltp}")

                try:
                    # 🚨 EXIT LOGIC
                    if position == 1 and ltp <= sl:
                        log1(f"[{task_id}] 🔥 SL HIT LONG {symbol} at {ltp}")

                        exit_market(kite, symbol, qty, "SELL")

                        with sl_lock:
                            LIVE_SL.pop(task_id, None)

                        delete_live_sl(task_id)
                        stop_task(task_id, "SL Hit") 

                    elif position == -1 and ltp >= sl:
                        log1(f"[{task_id}] 🔥 SL HIT SHORT {symbol} at {ltp}")

                        exit_market(kite, symbol, qty, "BUY")

                        with sl_lock:
                            LIVE_SL.pop(task_id, None)

                        delete_live_sl(task_id)
                        stop_task(task_id, "SL Hit") 

                except Exception as e:
                    log1(f"[{task_id}] Exit error: {e}")
    def on_close(ws, code, reason):
        global WS_RUNNING, KWS
        log1(f"🔌 WS Closed: {reason}")
        WS_RUNNING = False
        KWS = None

    kws.on_close = on_close
    kws.on_connect = on_connect
    kws.on_ticks = on_ticks

    kws.connect(threaded=True)
def exit_market(kite, symbol, qty, side):
    log1(f"in exit condition -- {symbol} -- {side} -- {qty}")
    try:
        txn_type = kite.TRANSACTION_TYPE_SELL if side == "SELL" else kite.TRANSACTION_TYPE_BUY

        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_MCX,
            tradingsymbol=symbol,
            transaction_type=txn_type,
            quantity=qty,
            product=kite.PRODUCT_NRML,
            order_type=kite.ORDER_TYPE_MARKET,
            market_protection=2,
            validity=kite.VALIDITY_DAY
        )

        log1(f"✅ EXITED {symbol}, Order ID: {order_id}")

    except Exception as e:
        log1(f"Exit error: {e}")



sl_lock = Lock()

def trailing_worker(task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val):
    
    try:
        
        # 🚀 YOUR STRATEGY LOGIC
        log1(f"[{task_id}] | {instrument} | Running {indicator} | min={min_val} max={max_val} | Quantity= {qty}")
        # log1(f"{timeframe} -- {sleeptime}")
        instrument_token = INSTRUMENT_MAP.get(instrument)

        log1(instrument_token)

        if not instrument_token:
            log1(f"❌ Token not found for {instrument}")

            conn = sqlite3.connect("trailing.db", check_same_thread=False)
            c = conn.cursor()
            c.execute("UPDATE trailing SET running=0 WHERE id=?", (task_id,))
            conn.commit()
            conn.close()

            return
        sleeptime = 0
        sleep_map = {
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600
        }

        sleeptime = sleep_map.get(timeframe, 300)
        interval_map = {
            "5m": "5minute",
            "15m": "15minute",
            "30m": "30minute",
            "1h": "60minute"
        }
        
        kite_interval = interval_map.get(timeframe, "5minute")
        log1(f"[{task_id}] Worker started")
        access_token = read_access_token()

        kite_local = KiteConnect(api_key=API_KEY)
        kite_local.set_access_token(access_token)
        
        exchange = "MCX"
        sl_orderid = None
        stoploss_valo = 0
        while True:
            conn = sqlite3.connect("trailing.db", check_same_thread=False)
            c = conn.cursor()

            status = c.execute(
                "SELECT running FROM trailing WHERE id=?",
                (task_id,)
            ).fetchone()

            conn.close()

            if not status or status[0] == 0:
                log1(f"[{task_id}] Stopped normally")
                break
            
            
            now = datetime.now() + timedelta(hours=5, minutes=30)
            # now = datetime.now() 
            log1(f'Present Time: {now}')

            market_open  = (now.hour > 9) or (now.hour == 9 and now.minute >= 10)
            # market_open  = (now.hour >= 8)
            market_close = (now.hour > 23) or (now.hour == 23 and now.minute >= 30)
            time23 = (now.hour >= 23)

            if not (market_open and not market_close):
                print("🕘 MCX Market Closed — sleeping...")
                log1("🕘 MCX Market Closed — sleeping...")
                wait_until_next_time(timeframe)
                # time.sleep(600)
                continue
            # log1("Fetching data")
            df = fetch_with_retry_token(instrument, instrument_token, kite_interval, kite_local)
            # log1("Fetching data complete")
            # log1(df.tail(3))
            df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','oi':'OI'}, inplace=True)
            log1(f"✅ Data fetched: {len(df)} bars | Last candle at {df['date'].iloc[-1]}")
            log1("Fetching position")
            position = 0
            positions = kite_local.positions()
            pos = next((p for p in positions["net"] if p["tradingsymbol"] == instrument), None)
            if pos:
                log1(f"🟢 Symbol: {pos['tradingsymbol']} | 📊 Quantity: {pos['quantity']} | 💰 Avg Price: {pos['average_price']} | 📈 P&L: {pos['pnl']}")
                if pos['quantity'] > 0:
                    position = 1
                elif pos['quantity'] < 0:
                    position = -1
                else:
                    position = 0
            if position != 0:
                stoploss_val = get_stoploss_value(df, instrument, indicator, min_val, multiplier, max_val, position)
                stoploss_val = int(stoploss_val)
            else:
                
                if sl_orderid != None:
                    try:
                        cancel_order(sl_orderid, kite_local)
                    except Exception as e: 
                        log(f"Stoploss cancel error {e}")
                SL_ORDERS.pop(task_id, None)
                sl_orderid = None
                stop_task(task_id, "No Position")
                break
            
            # 🔥 STORE SL IN MEMORY INSTEAD OF ORDER
            if position != 0:
                with sl_lock:
                    prev_sl = LIVE_SL.get(task_id, {}).get("stoploss")
                    if prev_sl:
                        if position == 1:  # LONG
                            stoploss_val = max(prev_sl, stoploss_val)
                        elif position == -1:  # SHORT
                            stoploss_val = min(prev_sl, stoploss_val)
                    log1(f"{prev_sl}, {stoploss_val}")
                    LIVE_SL[task_id] = {
                        "symbol": instrument,
                        "stoploss": stoploss_val,
                        "position": position,
                        "qty": qty
                    }

                if prev_sl != stoploss_val:
                    save_live_sl(task_id, instrument, stoploss_val, position, qty)

                log1(f"[{task_id}] SL Updated in memory: {stoploss_val}")

            else:
                # 🚨 No position → remove SL tracking
                with sl_lock:
                    LIVE_SL.pop(task_id, None)

                delete_live_sl(task_id)

                stop_task(task_id, "No Position")
                break


            if position == 0:
                log1("No positions")
            # time.sleep(sleeptime)
            wait_until_next_time(timeframe)
        sl_orderid_c = SL_ORDERS.get(task_id)
        if (sl_orderid != None) and (sl_orderid_c != sl_orderid):
            SL_ORDERS[task_id] = sl_orderid
        elif (sl_orderid_c is not None) and sl_orderid == None:
            SL_ORDERS.pop(task_id, None)
    except Exception as e:
        log1(f"[{task_id}] ERROR: {str(e)}")

        # ❌ Mark as stopped due to error
        conn = sqlite3.connect("trailing.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("UPDATE trailing SET running=0 WHERE id=?", (task_id,))
        conn.commit()
        conn.close()

    finally:
        # 🧹 CLEAN THREAD FROM MEMORY
        TRAILING_THREADS.pop(task_id, None)
        stop_ws_if_idle() 
        log1(f"[{task_id}] Thread cleaned up")
    

# import uuid
# from flask import request

@app.route('/start_trailing_row', methods=['POST'])
def start_trailing_row():
    data = request.json
    task_id = data.get("id")
    instrument = data.get('instrument')
    indicator = data['indicator']
    timeframe = data.get('timeframe', '5m')
    qty = int(data['qty']) if data.get('qty') else 1
    min_val = int(data['min'])
    multiplier = float(data['multiplier'])
    max_val = int(data['max']) if data.get('max') else 0

    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()
    # 🔥 ✅ 1. IF ID EXISTS → UPDATE SAME ROW
    if task_id:
        log1(f"🔄 Updating existing row {task_id}")

        c.execute("""
            UPDATE trailing 
            SET instrument=?, indicator=?, timeframe=?, qty=?, min=?, multiplier=?, max=?, running=1
            WHERE id=?
        """, (instrument, indicator, timeframe, qty, min_val, multiplier, max_val, task_id))

        conn.commit()
        conn.close()

        # 🛑 Prevent duplicate thread
        if task_id in TRAILING_THREADS and TRAILING_THREADS[task_id].is_alive():
            log1(f"⚠️ Thread already running {task_id}")
            return jsonify({"id": task_id})

        # 🚀 Restart thread
        thread = threading.Thread(
            target=trailing_worker,
            args=(task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val)
        )
        thread.daemon = True
        thread.start()

        with threads_lock:
            TRAILING_THREADS[task_id] = thread
        start_ws_if_needed()

        return jsonify({"id": task_id})
    
    
    
    task_id = str(uuid.uuid4())

    log1(f"🆕 Creating new row {task_id}")

    c.execute("""
        INSERT INTO trailing (id, instrument, indicator, timeframe, qty, min, multiplier, max, running)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val, 1))

    conn.commit()
    conn.close()

    # 🚀 Start thread
    thread = threading.Thread(
        target=trailing_worker,
        args=(task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val)
    )
    thread.daemon = True
    thread.start()
    with threads_lock:
        TRAILING_THREADS[task_id] = thread
    start_ws_if_needed()
    return jsonify({"id": task_id})


@app.route('/stop_trailing_row', methods=['POST'])
def stop_trailing_row():
    access_token = read_access_token()

    kite_local = KiteConnect(api_key=API_KEY)
    kite_local.set_access_token(access_token)
    data = request.json
    task_id = data['id']

    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()
    sl_orderid = SL_ORDERS.get(task_id)

    if sl_orderid:
        try:
            cancel_order(sl_orderid, kite_local)
            log1(f"✅ SL canceled before stop: {sl_orderid}")
        except Exception as e:
            log1(f"❌ SL cancel error: {e}")

        SL_ORDERS.pop(task_id, None)
    log1(f"Trailling stop button {task_id}")
    c.execute("UPDATE trailing SET running=0 WHERE id=?", (task_id,))

    conn.commit()
    conn.close()
    stop_ws_if_idle()
    return {"status": "stopped"}

@app.route('/get_trailing')
def get_trailing():
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    rows = c.execute("SELECT * FROM trailing").fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "id": r[0],
            "instrument": r[1],
            "indicator": r[2],
            "timeframe": r[3] or "5m",
            "qty": r[4],
            "min": r[5],
            "multiplier": r[6],
            "max": r[7],
            "running": r[8],
            "status": "running" if r[8] == 1 else "stopped"
        })

    return jsonify(data)


@app.route('/delete_trailing_row', methods=['POST'])
def delete_trailing_row():
    data = request.json
    task_id = data.get('id')
    access_token = read_access_token()

    kite_local = KiteConnect(api_key=API_KEY)
    kite_local.set_access_token(access_token)
    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()
    sl_orderid = SL_ORDERS.get(task_id)

    if sl_orderid:
        try:
            cancel_order(sl_orderid, kite_local)
            log1(f"✅ SL canceled before delete: {sl_orderid}")
        except Exception as e:
            log1(f"❌ SL cancel error: {e}")

        SL_ORDERS.pop(task_id, None)
    c.execute("DELETE FROM trailing WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    stop_ws_if_idle()

    log1(f"Deleted row {task_id}")

    return jsonify({"status": "deleted"})





@app.route("/strategy_status")
def strategy_status():
    global STRATEGY_RUNNING
    return "running" if STRATEGY_RUNNING else "stopped"


def save_quantity(value, file):
    if file == "quantity":
        with open("quantity.txt", "w") as f:
            f.write(str(value))
    elif file == "indicator":
        with open("indicator.txt", "w") as f:
            f.write(str(value))
    elif file == "min_val":
        with open("minval.txt", "w") as f:
            f.write(str(value))
    elif file == "max_val":
        with open("maxval.txt", "w") as f:
            f.write(str(value))
    elif file == "multiplier":
        with open("multiplier.txt", "w") as f:
            f.write(str(value))
    
    # with open("quantity.txt", "w") as f:
    #     f.write(str(qty))

def read_quantity(file):
    # def read_value(file):
    config = {
        "quantity": ("quantity.txt", int, 10),
        "indicator": ("indicator.txt", str, "senkou_a"),
        "min_val": ("minval.txt", int, 0),
        "max_val": ("maxval.txt", int, 0),
        "multiplier": ("multiplier.txt", float, 1.0),
    }

    filename, dtype, default = config[file]

    try:
        return dtype(open(filename).read().strip())
    except:
        return default


def init_analysis_db():
    conn = sqlite3.connect("analysis.db", check_same_thread=False)
    c = conn.cursor()

    # 🔥 State table
    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis_state (
        id INTEGER PRIMARY KEY,
        running INTEGER
    )
    """)

    # 🔥 Data table (IMPORTANT)
    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis_data (
        timeframe TEXT PRIMARY KEY,
        price REAL,
        ret6 REAL,
        ret12 REAL,
        trend TEXT,
        l_high TEXT,
        l_low REAL,
        highlow TEXT,
        atr_val REAL,
        volatility_regime TEXT,
        volatility_per REAL,   
        tenkan_kijun TEXT,   
        price_tenkan TEXT,
        cloud_trend TEXT,      
        rsi REAL,      
        vwap TEXT,      
        signal TEXT,
        updated_at TEXT
    )
    """)

    c.execute("INSERT OR IGNORE INTO analysis_state (id, running) VALUES (1, 0)")

    conn.commit()
    conn.close()

# -------------------- GLOBAL STATE --------------------
# ANALYSIS_RUNNING = False
ANALYSIS_THREADS = {}

# ANALYSIS_DATA = {
#     "5m": {},
#     "15m": {},
#     "30m": {}
# }
ANALYSIS_DATA = {
    "5m": {"price": "", "ret6": "", "ret12": ""},
    "15m": {"price": "", "ret6": "", "ret12": ""},
    "30m": {"price": "", "ret6": "", "ret12": ""}
}

# -------------------- WORKER --------------------
def analysis_worker(tf, instrument, instrument_token):
    # global ANALYSIS_RUNNING

    ensure_analysis_db() 
    interval_map = {
        "5m": "5minute",
        "15m": "15minute",
        "30m": "30minute",
        "1h": "60minute"
    }
    
    kite_interval = interval_map.get(tf, "5minute")
    
    access_token = read_access_token()

    kite_local = KiteConnect(api_key=API_KEY)
    kite_local.set_access_token(access_token)
    
    exchange = "MCX"
    
    while True:
        # ✅ CHECK DB STATE (IMPORTANT)
        conn = sqlite3.connect("analysis.db", check_same_thread=False)
        c = conn.cursor()

        running = c.execute(
            "SELECT running FROM analysis_state WHERE id=1"
        ).fetchone()[0]

        conn.close()
        log1(f"{tf} DB running = {running}")
        if running == 0:
            log1(f"{tf} stopped")
            break


        try:
            now = datetime.now() + timedelta(hours=5, minutes=30)
            # now = datetime.now() 
            log1(f'Present Time: {now} --- {datetime.now()}')

            market_open  = (now.hour > 9) or (now.hour == 9 and now.minute >= 10)
            # market_open  = (now.hour >= 8)
            market_close = (now.hour > 23) or (now.hour == 23 and now.minute >= 30)
            time23 = (now.hour >= 23)

            if not (market_open and not market_close):
                print("🕘 MCX Market Closed — sleeping...")
                log1("🕘 MCX Market Closed — sleeping...")
                wait_until_next_time(tf)
                # time.sleep(600)
                continue
            # log1("Fetching data")
            df = fetch_with_retry_token(instrument, instrument_token, kite_interval, kite_local)
            if len(df) < 20:
                log1("Not enough data")
                continue
            log1(df.tail(2))
            df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','oi':'OI'}, inplace=True)
            data = data_analysis(df, tf)
            log1(data)
            ensure_analysis_db() 
            l_high_json = json.dumps(data.get("l_high"))
# l_low_json  = json.dumps(data.get("l_low"))
            #  ✅ STORE IN DB (VERY IMPORTANT)
            conn = sqlite3.connect("analysis.db", check_same_thread=False)
            c = conn.cursor()

            c.execute("""
                INSERT OR REPLACE INTO analysis_data
                (timeframe, price, ret6, ret12, trend, l_high, l_low, highlow, atr_val, volatility_regime, volatility_per, tenkan_kijun, price_tenkan, cloud_trend, rsi, vwap, signal, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tf,
                data.get("price"),
                data.get("ret6"),
                data.get("ret12"),
                data.get("trend"),
                # data.get("l_high"),
                l_high_json,
                data.get("l_low"),
                data.get("highlow"),
                data.get("atr_val"),
                data.get("volatility_regime"),
                data.get("volatility_per"),
                data.get("tenkan_kijun"),
                data.get("price_tenkan"),
                data.get("cloud_trend"),
                data.get("rsi"),
                data.get("vwap"),
                data.get("signal"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

            conn.commit()
            conn.close()
            
            log1(f"{tf} stored in DB")

            # ANALYSIS_DATA[tf] = data
            # log1(f"{tf} updated")
            wait_until_next_time(tf)

        except Exception as e:
            log1(f"{tf} error: {e}")
            time.sleep(5)

        # ✅ Smart sleep (instant stop support)
        # for _ in range(sleep_map[tf]):
        #     if not ANALYSIS_RUNNING:
        #         break
        #     time.sleep(1)

# -------------------- START --------------------
def ensure_analysis_db():
    conn = sqlite3.connect("analysis.db", check_same_thread=False)
    c = conn.cursor()

    # Create table if not exists
    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis_state (
        id INTEGER PRIMARY KEY,
        running INTEGER
    )
    """)
    # 🔥 DATA TABLE (MISSING ONE)
    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis_data (
        timeframe TEXT PRIMARY KEY,
        price REAL,
        ret6 REAL,
        ret12 REAL,
        trend TEXT,
        l_high TEXT,
        l_low REAL,
        highlow TEXT,
        atr_val REAL,
        volatility_regime TEXT,
        volatility_per REAL,   
        tenkan_kijun TEXT,   
        price_tenkan TEXT,
        cloud_trend TEXT,    
        rsi REAL,      
        vwap TEXT,  
        signal TEXT,
        updated_at TEXT
    )
    """)
    # Ensure row exists
    c.execute("INSERT OR IGNORE INTO analysis_state (id, running) VALUES (1, 0)")

    conn.commit()
    conn.close()
    
def start_analysis_internal():
    log1("🚀 Starting analysis internal")
    global ANALYSIS_THREADS
    instrument = "GOLDM26MAYFUT"
    instrument_token = 124881671   # ✅ FIXED (int)

    for tf in ["5m", "15m", "30m"]:
        log1(f"Starting thread {tf}")
        if tf in ANALYSIS_THREADS and ANALYSIS_THREADS[tf].is_alive():
            continue  # already running
        t = threading.Thread(
            target=analysis_worker,
            args=(tf, instrument, instrument_token)
        )
        t.daemon = True
        t.start()
        ANALYSIS_THREADS[tf] = t

def is_analysis_running():
    conn = sqlite3.connect("analysis.db")
    c = conn.cursor()

    # running = c.execute(
    #     "SELECT running FROM analysis_state WHERE id=1"
    # ).fetchone()[0]
    row = c.execute(
        "SELECT running FROM analysis_state WHERE id=1"
    ).fetchone()

    conn.close()
    return row[0] if row else 0

@app.route('/start_analysis')
def start_analysis():
    if is_analysis_running() == 1:
        return {"status": "already running"}
    
    ensure_analysis_db()   # 🔥 ADD THIS
    conn = sqlite3.connect("analysis.db", check_same_thread=False)
    c = conn.cursor()

    # 🔥 Now update safely
    c.execute("UPDATE analysis_state SET running=1 WHERE id=1")

    conn.commit()
    conn.close()
    start_analysis_internal()
    log1("✅ Analysis started")
    return jsonify({"status": "started"})

# -------------------- STOP --------------------
@app.route('/stop_analysis')
def stop_analysis():
    conn = sqlite3.connect("analysis.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis_state (
        id INTEGER PRIMARY KEY,
        running INTEGER
    )
    """)

    c.execute("INSERT OR IGNORE INTO analysis_state (id, running) VALUES (1, 0)")

    c.execute("UPDATE analysis_state SET running=0 WHERE id=1")

    conn.commit()
    conn.close()

    log1("⛔ Analysis stopped")

    return jsonify({"status": "stopped"})

# -------------------- GET DATA --------------------
@app.route('/get_analysis')
def get_analysis():
    conn = sqlite3.connect("analysis.db", check_same_thread=False)
    c = conn.cursor()

    rows = c.execute("SELECT * FROM analysis_data").fetchall()
    conn.close()

    result = {}

    for r in rows:
        tf = r[0]
        result[tf] = {
            "price": r[1],
            "ret6": r[2],
            "ret12": r[3],
            "trend": r[4],
            "l_high": json.loads(r[5]) if r[5] else [],
            "l_low": r[6],
            "highlow": r[7],
            "atr_val": r[8],
            "volatility_regime": r[9],
            "volatility_per": r[10],
            "tenkan_kijun": r[11],
            "price_tenkan": r[12],
            "cloud_trend": r[13],
            "rsi": r[14],
            "vwap": r[15],
            "signal": r[16]
        }

    return jsonify(result)

@app.route('/analysis_status')
def analysis_status():
    try:
        conn = sqlite3.connect("analysis.db", check_same_thread=False)
        c = conn.cursor()

        # 🔥 Ensure table exists
        c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_state (
            id INTEGER PRIMARY KEY,
            running INTEGER
        )
        """)

        # 🔥 Ensure row exists
        c.execute("INSERT OR IGNORE INTO analysis_state (id, running) VALUES (1, 0)")

        row = c.execute(
            "SELECT running FROM analysis_state WHERE id=1"
        ).fetchone()

        conn.commit()
        conn.close()

        return jsonify({"running": row[0] if row else 0})

    except Exception as e:
        log1(f"analysis_status error: {str(e)}")
        return jsonify({"running": 0})

def restart_analysis():
    conn = sqlite3.connect("analysis.db", check_same_thread=False)
    c = conn.cursor()

    running = c.execute(
        "SELECT running FROM analysis_state WHERE id=1"
    ).fetchone()[0]

    conn.close()

    if running == 1:
        log1("🔄 Restarting analysis...")
        start_analysis_internal()


@app.route('/search_symbols')
def search_symbols():
    query = request.args.get('q', '')

    conn = sqlite3.connect("instruments.db")
    c = conn.cursor()

    rows = c.execute("""
        SELECT tradingsymbol FROM instruments
        WHERE segment='EQ' AND tradingsymbol LIKE ?
        LIMIT 50
    """, (f"%{query}%",)).fetchall()
    

    conn.close()

    results = [{"id": r[0], "text": r[0]} for r in rows]

    return jsonify({"results": results})

@app.route('/searchfno_symbols')
def search_symbols():
    query = request.args.get('q', '')

    conn = sqlite3.connect("instruments.db")
    c = conn.cursor()

    rows = c.execute("""
        SELECT tradingsymbol FROM instruments
        WHERE segment='NFO-FUT' AND tradingsymbol LIKE ?
        LIMIT 50
    """, (f"%{query}%",)).fetchall()
    

    conn.close()

    results = [{"id": r[0], "text": r[0]} for r in rows]

    return jsonify({"results": results})

@app.route('/get_watchlist_items')
def get_watchlist_items():
    wid = request.args.get("id")

    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    rows = c.execute("""
        SELECT symbol FROM watchlist_items
        WHERE watchlist_id=?
    """, (wid,)).fetchall()

    conn.close()

    return jsonify([r[0] for r in rows])

@app.route('/stocks_analysis')
def stocks_analysis():
    init_watchlist_db()

    # 🔥 ENSURE TABLE EXISTS BEFORE QUERY

    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    watchlists = c.execute("SELECT * FROM watchlists").fetchall()
    conn.close()

    # fetch symbols
    conn = sqlite3.connect("instruments.db")
    c = conn.cursor()
    symbols = c.execute("""
        SELECT tradingsymbol FROM instruments 
        WHERE segment='EQ'
        ORDER BY tradingsymbol
    """).fetchall()
    conn.close()

    symbols = [s[0] for s in symbols]

    return render_template(
        "stocks_analysis.html",
        watchlists=watchlists,
        symbols=symbols
    )

@app.route('/create_watchlist', methods=['POST'])
def create_watchlist():
    init_watchlist_db()
    name = request.json.get("name")

    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    c.execute("INSERT OR IGNORE INTO watchlists (name) VALUES (?)", (name,))
    conn.commit()

    # 🔥 Get ID
    c.execute("SELECT id FROM watchlists WHERE name=?", (name,))
    wid = c.fetchone()[0]

    conn.close()

    return {"status": "created", "id": wid}


@app.route('/delete_watchlist', methods=['POST'])
def delete_watchlist():
    wid = request.json.get("id")

    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    c.execute("DELETE FROM watchlists WHERE id=?", (wid,))
    c.execute("DELETE FROM watchlist_items WHERE watchlist_id=?", (wid,))

    conn.commit()
    conn.close()

    return {"status": "deleted"}

@app.route('/validate_symbols', methods=['POST'])
def validate_symbols():
    data = request.json
    symbols = data.get("symbols", [])

    # 🔥 Handle empty input (IMPORTANT)
    if not symbols:
        return jsonify({"valid": [], "invalid": []})

    conn = sqlite3.connect("instruments.db")
    c = conn.cursor()

    # 🔥 Create placeholders (?, ?, ?, ...)
    placeholders = ",".join(["?"] * len(symbols))

    query = f"""
        SELECT tradingsymbol FROM instruments
        WHERE tradingsymbol IN ({placeholders})
        AND segment='EQ'
    """

    rows = c.execute(query, symbols).fetchall()
    conn.close()

    # 🔥 Convert DB result to set
    valid_set = set(r[0] for r in rows)

    # 🔥 Split valid & invalid
    valid = [s for s in symbols if s in valid_set]
    invalid = [s for s in symbols if s not in valid_set]

    return jsonify({
        "valid": valid,
        "invalid": invalid
    })
@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.json
    wid = data.get("watchlist_id")
    stocks = data.get("stocks", [])

    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    for s in stocks:
        c.execute("""
            INSERT OR IGNORE INTO watchlist_items (watchlist_id, symbol)
            VALUES (?, ?)
        """, (wid, s))

    conn.commit()
    conn.close()

    return {"status": "added"}
@app.route("/delete_stocks_from_watchlist", methods=["POST"])
def delete_stocks_from_watchlist():
    data = request.json
    wid = data.get("watchlist_id")
    stocks = data.get("stocks", [])
    # stocks = [s.upper() for s in stocks]
    conn = sqlite3.connect("stocks_analysis.db")  # ✅ FIXED
    c = conn.cursor()

    c.executemany("""
        DELETE FROM watchlist_items
        WHERE watchlist_id = ? AND symbol = ?
    """, [(wid, s) for s in stocks])

    conn.commit()
    conn.close()

    return jsonify({"status": "success"})


from instruments import *
from concurrent.futures import ThreadPoolExecutor

# @app.route('/instruments_dashboard')
# def instruments_dashboard():
#     data = get_stock_with_futures()
#     return render_template("instruments_dashboard.html", data=data)


@app.route('/instruments_dashboard')
def instruments_dashboard():
    access_token = read_access_token()
    kite.set_access_token(access_token)
    ensure_instruments_data(kite)   # 🔥 ADD THIS

    data = get_stock_with_futures()
    return render_template("instruments_dashboard.html", data=data)

@app.route('/reload_instruments', methods=['POST'])
def reload_instruments_route():
    access_token = read_access_token()
    kite.set_access_token(access_token)
    reload_instruments(kite)
    return redirect('/instruments_dashboard')

def get_instrument_token(symbol):
    conn = sqlite3.connect("instruments.db")
    c = conn.cursor()

    row = c.execute("""
        SELECT instrument_token 
        FROM instruments
        WHERE tradingsymbol=? AND segment='EQ'
    """, (symbol,)).fetchone()

    conn.close()

    return row[0] if row else None
def empty_stock_result(symbol, e):
    return {
        "symbol": symbol,
        "price": None,
        "ret5": None,
        "ret15": None,
        "ret30": None,
        "ret90": None,
        "trend_30m": None,
        "trend_60m": None,
        "trend_1d": None,
        "vwap_30m": None,
        "vwap_60m": None,
        "vwap_1d": None,
        "rsi_30m": None,
        "rsi_60m": None,
        "rsi_1d": None,
        "tenkan_kijun_30m": None,
        "tenkan_kijun_60m": None,
        "tenkan_kijun_1d": None,
        "price_tenkan_30m": None,
        "price_tenkan_60m": None,
        "price_tenkan_1d": None,
        "cloud_trend_30m": None,
        "cloud_trend_60m": None,
        "cloud_trend_1d": None,
        "signal_30m": None,
        "signal_60m": None,
        "signal_1d": None,
    }
def analyze_one_stock(symbol, access_token):
    try:
        kite_local = KiteConnect(api_key=API_KEY)
        kite_local.set_access_token(access_token)
        token = get_instrument_token(symbol)

        if not token:
            return empty_stock_result(symbol, "token issue")

        df1 = fetch_with_retry_token(symbol, token, "30minute", kite_local, period = 60)
        df2 = fetch_with_retry_token(symbol, token, "60minute", kite_local, period = 90)
        df3 = fetch_with_retry_token(symbol, token, "day", kite_local, period = 360)
        
        if df1 is None or len(df1) < 120:
            return empty_stock_result(symbol, "df1")

        df1.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'oi': 'OI'
        }, inplace=True)
        
        if df2 is None or len(df2) < 120:
            return empty_stock_result(symbol, "df2")

        df2.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'oi': 'OI'
        }, inplace=True)
        if df3 is None or len(df3) < 120:
            return empty_stock_result(symbol, "df3")

        df3.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'oi': 'OI'
        }, inplace=True)

        df_yf = yf.Ticker(symbol+".NS")
        info = df_yf.get_info()
        
        result1, df1 = stock_data_analysis(df1, "30m")
        result2, df2 = stock_data_analysis(df2, "60m")
        result3, df3 = stock_data_analysis(df3, "1d")
        result_ret = stock_data_analysis_common(df3)
        result = {
            # 'symbol': result1['symbol'],
            'price': result1['price'],
            'ret1': result_ret['ret1'],
            'ret5': result_ret['ret5'],
            'ret15': result_ret['ret15'],
            'ret30': result_ret['ret30'],
            'ret90': result_ret['ret90'],
            'trend_30m': result1['trend'],
            'trend_60m': result2['trend'],
            'trend_1d': result3['trend'],
            'vwap_30m': result1['vwap'],
            'vwap_60m': result2['vwap'],
            'vwap_1d': result3['vwap'],
            'rsi_30m': result1['rsi'],
            'rsi_60m': result2['rsi'],
            'rsi_1d': result3['rsi'],
            'tenkan_kijun_30m': result1['tenkan_kijun'],
            'tenkan_kijun_60m': result2['tenkan_kijun'],
            'tenkan_kijun_1d': result3['tenkan_kijun'],
            'price_tenkan_30m': result1['price_tenkan'],
            'price_tenkan_60m': result2['price_tenkan'],
            'price_tenkan_1d': result3['price_tenkan'],
            'cloud_trend_30m': result1['cloud_trend'],
            'cloud_trend_60m': result2['cloud_trend'],
            'cloud_trend_1d': result3['cloud_trend'],
            'signal_30m': result1['signal'],
            'signal_60m': result2['signal'],
            'signal_1d': result3['signal'],
            'industry': info.get("industry", None),
            'sector': info.get("sector", None),
            'business': info.get("longBusinessSummary", None),
            'dividendYield': info.get("dividendYield", None),
            'payoutRatio': info.get("payoutRatio", None),
            'beta': info.get("beta", None),
            'trailingPE': info.get("trailingPE", None),
            'forwardPE': info.get("forwardPE", None),
            'trailingEPS': info.get("trailingEps", None),
            'forwardEPS': info.get("forwardEps", None),
            'epsTrailingTwelveMonths': info.get("epsTrailingTwelveMonths", None),
            'epsForward': info.get("epsForward", None),
            'epsCurrentYear': info.get("epsCurrentYear", None),
            'pegRatio': info.get("pegRatio", None),
            'marketCap': info.get("marketCap", None),   #
            'enterpriseValue': info.get("enterpriseValue", None), #
            'profitMargins': info.get("profitMargins", None), ##
            'bookValue': info.get("bookValue", None),
            'priceToBook': info.get("priceToBook", None),
            'earningsQuarterlyGrowth': info.get("earningsQuarterlyGrowth", None),
            'enterpriseToRevenue': info.get("enterpriseToRevenue", None),
            'enterpriseToEbitda': info.get("enterpriseToEbitda", None),
            'targetHighPrice': info.get("targetHighPrice", None), #
            'targetLowPrice': info.get("targetLowPrice", None), ###
            'targetMeanPrice': info.get("targetMeanPrice", None),  ##
            'recommendationKey': info.get("recommendationKey", None),  ##
            'totalCashPerShare': info.get("totalCashPerShare", None),
            'ebitda': info.get("ebitda", None),
            'totalRevenue': info.get("totalRevenue", None),
            'totalDebt': info.get("totalDebt", None),
            'quickRatio': info.get("quickRatio", None),  ##
            'currentRatio': info.get("currentRatio", None),  ##
            'debtToEquity': info.get("debtToEquity", None),  ##
            'revenuePerShare': info.get("revenuePerShare", None),
            'returnOnAssets': info.get("returnOnAssets", None),
            'returnOnEquity': info.get("returnOnEquity", None),
            'grossProfits': info.get("grossProfits", None),  
            'freeCashflow': info.get("freeCashflow", None),
            'operatingCashflow': info.get("operatingCashflow", None),
            'earningsGrowth': info.get("earningsGrowth", None),
            'revenueGrowth': info.get("revenueGrowth", None),
            'grossMargins': info.get("grossMargins", None),  ##
            'ebitdaMargins': info.get("ebitdaMargins", None),  ##
            'operatingMargins': info.get("operatingMargins", None),  ##
            'customPriceAlertConfidence': info.get("customPriceAlertConfidence", None),
            'fiftyTwoWeekRange': info.get("fiftyTwoWeekRange", None),
        }
        valu = valuation_analysis(result)
        # log1(val)
        growth = growth_analysis(result)
        # log1(growth)
        prof = profitability_analysis(result)
        # log1(prof)
        risk = financial_health_analysis(result)
        # log1(risk)
        sent = sentiment_analysis(result, result1['price'])
        # log1(sent)
        result.update(valu if isinstance(valu, dict) else {})
        result.update(growth if isinstance(growth, dict) else {})
        result.update(prof if isinstance(prof, dict) else {})
        result.update(risk if isinstance(risk, dict) else {})
        result.update(sent if isinstance(sent, dict) else {})

        composite = composite_score(result)
        result.update(composite if isinstance(composite, dict) else {})
        log1(result)
        return {"symbol": symbol, **result}

    except Exception as e:
        log1(f"Error in {symbol}: {e}")
        return empty_stock_result(symbol, "Error")


@app.route('/analyze_stocks', methods=['POST'])
def analyze_stocks():
    # symbols = request.json.get("symbols", [])
    data = request.get_json(silent=True) or {}
    symbols = data.get("symbols", [])

    access_token = read_access_token()

    # kite_local = KiteConnect(api_key=API_KEY)
    # kite_local.set_access_token(access_token)

    with ThreadPoolExecutor(max_workers=10) as executor:
        output = list(executor.map(
            lambda s: analyze_one_stock(s, access_token),
            symbols
        ))
    # log1(output)
    return jsonify([r for r in output if r])
def safe(val):
    if pd.isna(val) or val is None:
        return None
    return float(val)
@app.route("/get_chart_data")
def get_chart_data():
    symbol = request.args.get("symbol")
    interval = request.args.get("interval", "day")

    access_token = read_access_token()
    kite_local = KiteConnect(api_key=API_KEY)
    kite_local.set_access_token(access_token)

    token = get_instrument_token(symbol)

    if not token:
        return jsonify([])
    
    # 🔥 adjust period dynamically
    period_map = {
        "5minute": 7,
        "15minute": 15,
        "30minute": 60,
        "60minute": 60,
        "day": 665
    }
    # log1(f"{interval} -- {period_map.get(interval, 365)}")
    df = fetch_with_retry_token(
        symbol,
        token,
        interval,
        kite_local,
        period=period_map.get(interval, 365)
    )


    if df is None or len(df) < 50:
        return jsonify([])

    df.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df = indicator_values(df)
    df = df[100:]
    data = []
    for _, row in df.iterrows():

        o = safe(row["Open"])
        h = safe(row["High"])
        l = safe(row["Low"])
        c = safe(row["Close"])

        # 🚨 HARD FILTER (MANDATORY)
        if None in (o, h, l, c):
            continue
        # log1(row)
        # if interval == "day":
        #     time_val = row["Date"].strftime("%Y-%m-%d")
        # else:
        time_val = int(pd.Timestamp(row["Date"]).timestamp())
        data.append({
            "time": time_val,
            "open": o,
            "high": h,
            "low": l,
            "close": c,

            # indicators
            "ema": safe(row.get("ema")),
            "vwap": safe(row.get("VWAP")),
            "rsi": safe(row.get("RSI")),
            "tenkan": safe(row.get("tenkan")),
            "kijun": safe(row.get("kijun")),
            "spanA": safe(row.get("senkou_a")),
            "spanB": safe(row.get("senkou_b")),
            "volume": safe(row.get("Volume")),
        })

    return jsonify(data)


# ----------------------------
# Live Logs Page
# ----------------------------
@app.route("/logs")
def logs():
    return render_template("logs.html")
@app.route("/get_logs")
def get_logs():
    try:
        with open("static/logs.txt") as f:
            return "<br>".join(f.readlines()[-300:])
    except:
        return "No logs yet"

import logging
log = logging.getLogger('werkzeug')
log.disabled = True



def init_portfolio_db():
    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    # Portfolio table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )
    """)

    # Holdings table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS holdings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        symbol TEXT,
        quantity REAL,
        buy_price REAL,
        FOREIGN KEY(portfolio_id) REFERENCES portfolios(id)
    )
    """)

    conn.commit()
    conn.close()
init_portfolio_db()

@app.route('/get-symbols')
def get_symbols():
    conn = sqlite3.connect('instruments.db')  # or your table
    cur = conn.cursor()

    # Fetch EQ + FUT
    cur.execute("""
        SELECT tradingsymbol, exchange
        FROM instruments
        WHERE exchange IN ('NSE', 'NFO')
        LIMIT 1000
    """)

    data = cur.fetchall()
    conn.close()

    symbols = [
        {
            "symbol": row[0],
            "exchange": row[1]
        } for row in data
    ]

    return jsonify(symbols)

# @app.route('/search-symbol')
# def search_symbol():
#     query = request.args.get('q', '').upper()

#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cur = conn.cursor()

#         cur.execute("""
#             SELECT tradingsymbol, exchange
#             FROM instruments
#             WHERE tradingsymbol LIKE ?
#             AND exchange IN ('NSE', 'NFO')
#             AND (
#                 exchange = 'NSE'
#                 OR (exchange = 'NFO' AND instrument_type = 'FUT')
#             )
#             LIMIT 20
#         """, (f"%{query}%",))

#         data = cur.fetchall()
#         conn.close()

#         return jsonify({
#             "results": [
#                 {
#                     "id": r[0],   # required by Select2
#                     "text": f"{r[0]} ({r[1]})"
#                 } for r in data
#             ]
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route('/portfolio')
def portfolio():
    try:
        access_token = read_access_token()
        # kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token)
        positions_data = kite.positions()['net']
        active_positions = [p for p in positions_data if p['quantity'] != 0]

        symbols = []
        for p in active_positions:
            symbols.append(f"{p['exchange']}:{p['tradingsymbol']}")

        ltp_data = kite.ltp(symbols) if symbols else {}

        total_pnl = 0

        for p in active_positions:
            key = f"{p['exchange']}:{p['tradingsymbol']}"
            ltp = ltp_data.get(key, {}).get('last_price', 0)

            p['ltp'] = ltp
            p['pnl'] = (ltp - p['average_price']) * p['quantity'] * p['multiplier']
            p['side'] = 'LONG' if p['quantity'] > 0 else 'SHORT'
            # 👉 Invested value
            invested = p['average_price'] * abs(p['quantity'])

            # 👉 P&L %
            p['pnl_percent'] = (p['pnl'] / invested * 100) if invested != 0 else 0

            # Classification
            if p['exchange'] == 'NSE' and p['product'] in ['CNC', 'MIS']:
                p['type'] = 'Equity'
            elif p['exchange'] == 'NFO':
                p['type'] = 'Futures'
            elif p['exchange'] == 'MCX':
                p['type'] = 'Commodity'
            else:
                p['type'] = 'Other'

            total_pnl += p['pnl']
        
        
        conn = sqlite3.connect('portfolio.db')
        cur = conn.cursor()

        cur.execute("SELECT * FROM portfolios")
        portfolios = cur.fetchall()

        result = []

        for p in portfolios:
            cur.execute("SELECT * FROM holdings WHERE portfolio_id=?", (p[0],))
            holdings = cur.fetchall()

            result.append({
                "id": p[0],
                "name": p[1],
                "holdings": holdings
            })

        conn.close()
        return render_template('portfolio.html',
                        positions=active_positions,
                        total_pnl=total_pnl,
                        portfolios=result)

    except Exception as e:
        return str(e)

@app.route('/portfolio-data')
def portfolio_data():
    access_token = read_access_token()
    # kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)
    positions_data = kite.positions()['net']
    active_positions = [p for p in positions_data if p['quantity'] != 0]

    symbols = [f"{p['exchange']}:{p['tradingsymbol']}" for p in active_positions]
    ltp_data = kite.ltp(symbols) if symbols else {}

    total_pnl = 0

    for p in active_positions:
        log1(p)
        key = f"{p['exchange']}:{p['tradingsymbol']}"
        ltp = ltp_data.get(key, {}).get('last_price', 0)

        p['ltp'] = ltp
        p['pnl'] = (ltp - p['average_price']) * p['quantity'] * p['multiplier']
        p['side'] = 'LONG' if p['quantity'] > 0 else 'SHORT'
        # 👉 Invested value
        invested = p['average_price'] * abs(p['quantity'])

        # 👉 P&L %
        p['pnl_percent'] = (p['pnl'] / invested * 100) if invested != 0 else 0

        if p['exchange'] == 'NSE' and p['product'] in ['CNC', 'MIS']:
            p['type'] = 'Equity'
        elif p['exchange'] == 'NFO':
            p['type'] = 'Futures'
        elif p['exchange'] == 'MCX':
            p['type'] = 'Commodity'
        else:
            p['type'] = 'Other'

        total_pnl += p['pnl']

    return jsonify({
        "positions": active_positions,
        "total_pnl": total_pnl
    })
@app.route('/manual-portfolio')
def manual_portfolio():
    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    cur.execute("SELECT * FROM portfolios")
    portfolios = cur.fetchall()

    result = []

    for p in portfolios:
        cur.execute("SELECT * FROM holdings WHERE portfolio_id=?", (p[0],))
        holdings = cur.fetchall()

        result.append({
            "id": p[0],
            "name": p[1],
            "holdings": holdings
        })

    conn.close()
    return render_template('portfolio.html', portfolios=result)

@app.route('/create-portfolio', methods=['POST'])
def create_portfolio():
    name = request.form['name']

    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    cur.execute("INSERT INTO portfolios (name) VALUES (?)", (name,))

    conn.commit()
    conn.close()

    return redirect('/portfolio')

@app.route('/delete-portfolio/<int:portfolio_id>')
def delete_portfolio(portfolio_id):
    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    # Delete holdings first (important)
    cur.execute("DELETE FROM holdings WHERE portfolio_id=?", (portfolio_id,))
    
    # Then delete portfolio
    cur.execute("DELETE FROM portfolios WHERE id=?", (portfolio_id,))

    conn.commit()
    conn.close()

    return redirect('/portfolio')

@app.route('/add-holding', methods=['POST'])
def add_holding():
    portfolio_id = request.form['portfolio_id']
    symbol = request.form['symbol'].upper()
    quantity = float(request.form['quantity'])
    buy_price = float(request.form['buy_price'])

    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO holdings (portfolio_id, symbol, quantity, buy_price)
        VALUES (?, ?, ?, ?)
    """, (portfolio_id, symbol, quantity, buy_price))

    conn.commit()
    conn.close()

    return redirect('/portfolio')

@app.route('/delete-holding/<int:holding_id>')
def delete_holding(holding_id):
    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    cur.execute("DELETE FROM holdings WHERE id=?", (holding_id,))

    conn.commit()
    conn.close()

    return redirect('/portfolio')

@app.route('/edit-holding/<int:holding_id>')
def edit_holding(holding_id):
    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    cur.execute("SELECT * FROM holdings WHERE id=?", (holding_id,))
    holding = cur.fetchone()

    conn.close()

    return render_template('edit_holding.html', holding=holding)

@app.route('/update-holding/<int:holding_id>', methods=['POST'])
def update_holding(holding_id):
    symbol = request.form['symbol'].upper()
    quantity = float(request.form['quantity'])
    buy_price = float(request.form['buy_price'])

    conn = sqlite3.connect('portfolio.db')
    cur = conn.cursor()

    cur.execute("""
        UPDATE holdings
        SET symbol=?, quantity=?, buy_price=?
        WHERE id=?
    """, (symbol, quantity, buy_price, holding_id))

    conn.commit()
    conn.close()

    return redirect('/manual-portfolio')
# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    init_watchlist_db()   # 🔥 MUST BE FIRST
    init_analysis_db()
    load_live_sl_from_db()
    # restart_analysis()

    try:
        access_token = read_access_token()
        if access_token:
            kite.set_access_token(access_token)
            ensure_instruments_data(kite)
    except Exception as e:
        print(f"Instruments init error: {e}")

    app.run(port=8000, debug=True, use_reloader=False)  # 🔥 IMPORTANT


