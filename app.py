from flask import Flask, redirect, request, render_template, url_for, jsonify
from kiteconnect import KiteConnect
import os
import threading
import time
# from strategy import run_strategy, stop_strategy
from strategy import *
from trailling_strategy import run_trailling_strategy, stop_trailling_strategy
import uuid
from datetime import datetime, timedelta
from trailling_strategy import *


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



# @app.route('/start_trailing_row', methods=['POST'])
# def start_trailing_row():
#     data = request.json

#     config = {
#         "id": str(uuid.uuid4()),
#         "indicator": data['indicator'],
#         "min": int(data['min']),
#         "multiplier": float(data['multiplier']),
#         "max": int(data['max']),
#         "running": True
#     }

#     TRAILING_CONFIGS.append(config)

#     return config


import threading
import time

import json

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

# def load_instrument_map():
#     log1("loading instrument token")
#     with open("instruments.json") as f:
#         instruments = json.load(f)

#     return {
#         i["tradingsymbol"]: i["instrument_token"]
#         for i in instruments
#     }

INSTRUMENT_MAP = {}

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

def fetch_with_retry_token(symbol, token, interval, retries=3, delay=5):
    log1(token)
    for attempt in range(retries):
        try:
            time_correction = timedelta(hours=5, minutes=30)
            # time_correction = 0
            time_now = datetime.now() + time_correction
            time_delay = time_now - timedelta(days=30)
            # print(time_delay, time_now)
            # instrument = kite.ltp(f"MCX:{symbol}")[f"MCX:{symbol}"]['instrument_token']
            data = kite.historical_data(
                instrument_token=token,
                from_date=time_delay,
                to_date=time_now,
                interval=interval,
                oi = True
            )
            df = pd.DataFrame(data)
            log1(df['date'].iloc[-1])
            return df
        except Exception as e:
            log1(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
TRAILING_THREADS = {}

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
        interval_map = {
            "5m": "5minute",
            "15m": "15minute",
            "30m": "30minute",
            "1h": "60minute"
        }
        
        kite_interval = interval_map.get(timeframe, "5minute")
        access_token = read_access_token()
        log1(f"[{task_id}] Worker started")
        global kite    

        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token)
        
        exchange = "MCX"
        sl_orderid = None
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
            sleeptime = 0
            sleep_map = {
                "5m": 300,
                "15m": 900,
                "30m": 1800,
                "1h": 3600
            }

            sleeptime = sleep_map.get(timeframe, 300)
            
            now = datetime.now() + timedelta(hours=5, minutes=30)
            # now = datetime.now() 
            log1(f'Present Time: {now}')

            market_open  = (now.hour > 9) or (now.hour == 9 and now.minute >= 20)
            # market_open  = (now.hour >= 8)
            market_close = (now.hour > 23) or (now.hour == 23 and now.minute >= 30)
            time23 = (now.hour >= 23)

            if not (market_open and not market_close):
                print("🕘 MCX Market Closed — sleeping...")
                log1("🕘 MCX Market Closed — sleeping...")
                wait_until_next_time(timeframe)
                # time.sleep(600)
                continue
            log1("Fetching data")
            df = fetch_with_retry_token(instrument, instrument_token, kite_interval)
            log1("Fetching data complete")
            log1(df.tail(3))
            df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','oi':'OI'}, inplace=True)
            log1(f"✅ Data fetched: {len(df)} bars | Last candle at {df['date'].iloc[-1]}")
            log1("Fetching position")
            position = 0
            positions = kite.positions()
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
            else:
                sl_orderid == None
                if sl_orderid != None:
                    try:
                        cancel_order(sl_orderid)
                    except Exception as e: 
                        log(f"Stoploss cancel error {e}")
                stop_task(task_id, "No Position")
                break
            log1(f"Stoploss value is {stoploss_val}")
            price = df['Close'].iat[-1]
            # if stoploss_value 
            if (position == 1):
                if (sl_orderid != None):
                    try:
                        log1(f"MSL placed: {sl_orderid} {stoploss_val}")
                        modify_sl_order(sl_orderid, stoploss_val)
                    except Exception as e: 
                        log1(f"MSL order error {e}")
                        if "Trigger price" in e:
                            cancel_order(sl_orderid)
                            sl_orderid = None
                            buy_sell = "SELL"
                            quantity = qty
                            kite_app_buy_sell(exchange, instrument, buy_sell, quantity)
                            log1(f"Error occured so MSL order canceled and exit long position")
                elif(sl_orderid == None) and (price > stoploss_val):
                    quantity = qty
                    log1(f"SL placed: {sl_orderid} {stoploss_val}")
                    sl_orderid = place_sl_order(instrument, "SELL", quantity, stoploss_val)
                    log1("SL Placed")
                elif (sl_orderid != None) and (stoploss_val == 0):
                    try:
                        cancel_order(sl_orderid)
                    except Exception as e: 
                        log1(f"Stoploss cancel error {e}")
                    log1("SL Canceled")
                    sl_orderid = None
                else:
                    sl_orderid = None
            if position == -1:
                if (sl_orderid != None) and (stoploss_val != 0):
                    try:
                        log1(f"MSL placed: {sl_orderid} {stoploss_val} start")
                        modify_sl_order(sl_orderid, stoploss_val)
                        log1(f"SLM placed: {sl_orderid} {stoploss_val}")
                    except Exception as e:
                        log1(f"Error - {e}")
                        if "Trigger price" in e:
                            cancel_order(sl_orderid)
                            sl_orderid = None
                            buy_sell = "BUY"
                            quantity = qty
                            kite_app_buy_sell(exchange, instrument, buy_sell, quantity)
                            log1(f"Error occured so SL order canceled and exit from short position")
                    
                elif (sl_orderid == None) and (stoploss_val != 0):
                    quantity = qty
                    sl_orderid = place_sl_order(instrument , "BUY", quantity, stoploss_val)
                    log1(f"SL placed: {sl_orderid} {stoploss_val}")

                elif (sl_orderid != None) and (stoploss_val == 0):
                    try:
                        cancel_order(sl_orderid)
                    except Exception as e: 
                        log1(f"Stoploss cancel error {e}")
                    sl_orderid = None
                else:
                    sl_orderid = None
            if position == 0:
                log1("No positions")
            # time.sleep(sleeptime)
            wait_until_next_time(timeframe)

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
        log1(f"[{task_id}] Thread cleaned up")


# import uuid
# from flask import request

@app.route('/start_trailing_row', methods=['POST'])
def start_trailing_row():
    data = request.json

    instrument = data.get('instrument')
    indicator = data['indicator']
    timeframe = data.get('timeframe', '5m')
    qty = int(data['qty']) if data.get('qty') else 1
    min_val = int(data['min'])
    multiplier = float(data['multiplier'])
    max_val = int(data['max']) if data.get('max') else 0

    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    # 🔍 CHECK IF SAME CONFIG EXISTS
    existing = c.execute("""
        SELECT id FROM trailing 
        WHERE instrument=? AND indicator=? AND timeframe=? AND qty=? AND min=? AND multiplier=? AND max=?
    """, (instrument, indicator, timeframe, qty, min_val, multiplier, max_val)).fetchone()

    # 🔁 RESTART EXISTING
    if existing:
        task_id = existing[0]

        # 🛑 Prevent duplicate thread
        if task_id in TRAILING_THREADS and TRAILING_THREADS[task_id].is_alive():
            log1(f"Thread already running {task_id}")
            conn.close()
            return jsonify({"id": task_id})

        # 🔄 Restart
        c.execute("UPDATE trailing SET running=1 WHERE id=?", (task_id,))
        conn.commit()
        conn.close()

        log1(f"Restarting Trailing {indicator} | ID: {task_id}")

        thread = threading.Thread(
            target=trailing_worker,
            args=(task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val)
        )
        thread.daemon = True
        thread.start()

        TRAILING_THREADS[task_id] = thread

        return jsonify({"id": task_id})

    # 🆕 NEW TASK
    task_id = str(uuid.uuid4())

    log1(f"Going to start Trailing {indicator} | ID: {task_id}")

    c.execute("""
    INSERT INTO trailing (id, instrument, indicator, timeframe, qty, min, multiplier, max, running)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val, 1))
    conn.commit()
    conn.close()

    log1(f"Trailing STARTED {task_id}")

    thread = threading.Thread(
        target=trailing_worker,
        args=(task_id, instrument, indicator, timeframe, qty, min_val, multiplier, max_val)
    )
    thread.daemon = True
    thread.start()

    TRAILING_THREADS[task_id] = thread

    return jsonify({"id": task_id})


@app.route('/stop_trailing_row', methods=['POST'])
def stop_trailing_row():
    data = request.json
    task_id = data['id']

    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()
    log1(f"Trailling stop button {task_id}")
    c.execute("UPDATE trailing SET running=0 WHERE id=?", (task_id,))

    conn.commit()
    conn.close()

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

    conn = sqlite3.connect("trailing.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("DELETE FROM trailing WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()

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
# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app.run(port=8000, debug=True)
