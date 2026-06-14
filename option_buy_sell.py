import sqlite3
import logging
import pandas as pd
from technical_analysis import *
from load_once import *
from datetime import datetime, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from kiteconnect import KiteConnect
automation_threads = {}
automation_flags = {}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DB_NAME = "options_automation.db"

def log2(msg):
    with open("static/logs.txt", "a") as f:
        f.write(f"{msg}\n")

def wait_until_next_target_second(target_second):

    now = datetime.now()
    if target_second == 60:
        next_time = (
            now.replace(
                second=0,
                microsecond=0
            )
            + timedelta(minutes=1)
        )

    else:

        next_time = now.replace(
            second=target_second,
            microsecond=0
        )

        if now.second >= target_second:
            next_time += timedelta(minutes=1)

    wait_seconds = (
        next_time - now
    ).total_seconds()

    time.sleep(wait_seconds)
def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def get_option_tokens(kite_local, strikes, index):
    log2("options token")
    instruments = kite_local.instruments("NFO")
    df = pd.DataFrame(instruments)
    # NIFTY OPTIONS ONLY
    df = df[(df["name"] == index)&(df["instrument_type"].isin(["CE", "PE"]))]
    # CURRENT WEEK EXPIRY
    df["expiry"] = pd.to_datetime(df["expiry"])
    current_expiry = sorted(df["expiry"].unique())[0]
    log2(f"this week expiry -- {current_expiry}")
    df = df[df["expiry"] == current_expiry]
    # REQUIRED STRIKES
    df = df[df["strike"].isin(strikes)]
    result = []
    for _, row in df.iterrows():
        result.append({
            "symbol": row["tradingsymbol"],
            "strike": int(row["strike"]),
            "type": row["instrument_type"],
            "token": row["instrument_token"],
            "expiry": str(
                row["expiry"].date()
            )
        })
    return sorted(
        result,
        key=lambda x: (
            x["strike"],
            x["type"]
        )
    )

def fetch_and_analyze(symbol, name, timeframe, kite_local):
    """
    Fetch data and analyze
    """
    try:
        log2("fetch and analysis of index start here")
        # Get historical data
        df = get_historical_data(symbol, name, timeframe, kite_local)
        if df is None or len(df) < 2:
            return {
                'symbol': name,
                'ltp': 1,
                'return': 0,
                'high': 0,
                'low': 0,
                'signal': 'NO_DATA',
                'hl_trend': 'N/A',
                'vwap': 0,
                'rsi': 0,
                'price_tenkan': 0,
            }
        
        # Run analysis
        analysis = {
            'symbol': name,
            'high': float(df['High'].iloc[-1]),
            'low': float(df['Low'].iloc[-1]),
            'open': float(df['Open'].iloc[-1]),
        }
        data, df = stock_data_analysis(df, timeframe)
        analysis.update(data if isinstance(data, dict) else {})
        return analysis
    
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return {
            'symbol': name,
            'ltp': 0,
            'return': 0,
            'high': 0,
            'low': 0,
            'signal': 'ERROR',
            'hl_trend': 'N/A',
            'vwap': 0,
            'rsi': 0,
            'price_tenkan': 0,
        }

def fetch_and_analyze_option(kite_local, item, name, timeframe):
    symbol = item['symbol']
    """
    Fetch data and analyze (for options)
    """
    try:
        
        # log2(f"{symbol} -- {strike} -- {option_type} -- {index} -- {expiry} ")
        df = get_historical_data(item['token'], "option", timeframe, kite_local)
        # log2(df.tail(5))
        
        if df is None or len(df) < 2:
            return None
        # oi_change = round(((float(df['OI'].iloc[-1])/float(df['OI'].iloc[-2])) - 1)*100,2)
        log2(f"option last candle time -- {df["date"].iloc[-1]}")
        if 'OI' not in df.columns:
            oi_val = 0
            oi_change = 0
        else:
            oi_val = float(df['OI'].iloc[-1])
            prev_oi = float(df['OI'].iloc[-2])
            if prev_oi > 0:
                oi_change = round(
                    ((oi_val/ prev_oi) - 1) * 100,
                    2
                )
            else:
                oi_change = 0
        # vol_ratio = round(float(df['Volume'].iloc[-1]) /float(df["Volume"].rolling(10).mean().iloc[-1]),2)
        avg_vol = float(
            df["Volume"].rolling(10).mean().iloc[-1]
        )

        if avg_vol > 0:
            vol_ratio = round(
                float(df['Volume'].iloc[-1]) / avg_vol,
                2
            )
        else:
            vol_ratio = 0
        analysis = {
            'symbol': item['symbol'],
            'high': float(df['High'].iloc[-1]),
            'low': float(df['Low'].iloc[-1]),
            'open': float(df['Open'].iloc[-1]),
            'volume_ratio': vol_ratio,
            'oi': oi_val,
            'oi_change': oi_change,
            'strike': item['strike'],
            'type': item['option_type'],
            'expiry': item['expiry'],
            'token': item['token'],
        }
        data, df = stock_data_analysis(df, timeframe, "options")
        analysis.update(data if isinstance(data, dict) else {})
        df_values = {
            'kijun': float(df['kijun'].iloc[-1]),
            'tenkan': float(df['tenkan'].iloc[-1]),
            'senkou_a': float(df['senkou_a'].iloc[-1]),
            'senkou_b': float(df['senkou_b'].iloc[-1]),
            'max_5': float(df['Close'].tail(5).max()),
            'max_10': float(df['Close'].tail(10).max()),
            'min_10': float(df['Close'].tail(10).min())
        }
        analysis.update(df_values if isinstance(df_values, dict) else {})
        return analysis
        # return "None"
    
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def get_historical_data(symbol, name, timeframe, kite_local):
    """
    Get historical OHLCV data from Kite
    """
    try:
        if name != "option":
            token = INSTRUMENT_MAP.get(name)
        else:
            token = (symbol)
        
        # Define date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        
        # Fetch data
        data = kite_local.historical_data(
            instrument_token=int(token),
            from_date=from_date.date(),
            to_date=to_date.date(),
            interval=timeframe,
            oi=True
        )
        
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.rename(columns={

            'open':'Open',
            'high':'High',
            'low':'Low',
            'close':'Close',
            'volume':'Volume',
            'oi':'OI'

        }, inplace=True)
        log2(f"sending historical data of index {df.columns}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None


def save_automation_settings(data):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO
    automation_settings(
        index_name,

        option_ce_symbol,
        option_pe_symbol,

        option_ce_token,
        option_pe_token,

        timeframe,
                
        lot_size,

        enabled,

        qty,

        risk_percent,

        target_type,

        target_percent,

        sl_base,

        sl_percent,

        sl_cap,

        refresh_seconds,

        updated_at

    )

    VALUES(
        ?,?,?,?,?,?,?,
        1,
        ?,?,?,?,?,?,?,?,
        datetime('now', '+5 hours', '+30 minutes')
    )
    """,(

        data["index_name"],

        data["option_ce_symbol"],
        data["option_pe_symbol"],

        data["option_ce_token"],
        data["option_pe_token"],
        
        data["timeframe"],

        data["lot_size"],

        data["qty"],

        data["risk_percent"],

        data["target_type"],

        data["target_percent"],

        data["sl_base"],

        data["sl_percent"],

        data["sl_cap"],

        data["refresh_seconds"]

    ))

    conn.commit()

    conn.close()

def has_open_position(index_name):
    log2(f"checking live positions -- {index_name}")
    with get_connection() as conn:

        row = conn.execute("""
        SELECT *
        FROM live_positions
        WHERE index_name=?
        AND status='OPEN'
        """,(index_name,)).fetchone()

    return row
def get_automation_settings(index_name):
    log2("fetching automation settings")
    with get_connection() as conn:

        row = conn.execute("""
        SELECT *
        FROM automation_settings
        WHERE index_name=?
        """,(index_name,)).fetchone()

    return dict(row) if row else None

# def restore_automations():
#     rows = conn.execute("""
#         SELECT index_name
#         FROM automation_settings
#         WHERE enabled = 1
#     """).fetchall()

#     for row in rows:
#         start_thread(row["index_name"])


def automation_loop(index_name):
    try:
        logger.info(
            f"{index_name} automation started"
        )
        access_token = read_access_token()
        kite_local = KiteConnect(api_key=API_KEY)
        kite_local.set_access_token(access_token)
        log2(f"automation loop is going to start -- {index_name}")
        log2(automation_flags.get(index_name, False))
        while automation_flags.get(index_name, False):
            try:
                settings = get_automation_settings(index_name)
                log2(settings)
                
                if not settings:
                    logger.info(f"{index_name} settings not found")
                    automation_flags[index_name] = False
                    return
                log2(f"settings --- {settings}")
                log2("in automation loop --- looping")
                process_index(kite_local, index_name, settings)
                # square_off_before_close()
            except Exception as e:
                logger.error(
                    f"{index_name} Error : {e}"
                )
                automation_flags[index_name] = False
                break
            # time.sleep(30)
            refresh_seconds = settings['refresh_seconds']
            wait_until_next_target_second(refresh_seconds)
        log2("automation stopped")
        logger.info(
            f"{index_name} automation stopped"
        )
    except Exception as e:
        logger.exception(f"{index_name} crashed: {e}")
    finally:
        logger.info(f"{index_name} automation stopped")
        automation_threads.pop(index_name, None)
        automation_flags.pop(index_name, None)


def process_index(kite_local, index_name,settings):

    try:
        ptime = datetime.now()
        # No entries after market cutoff
        # if not market_open_for_entries():
        #     return
        # Check existing position
        position = has_open_position(index_name)
        if position:
            monitor_position(kite_local, position, settings)
            return
        timeframe = settings['timeframe']
        nifty_symbol = "NSE:NIFTY 50"
        nifty_name = "NIFTY 50"
        banknifty_symbol = "NSE:NIFTY BANK"
        banknifty_name = "NIFTY BANK"
        # No position available
        with ThreadPoolExecutor(
            max_workers=4
        ) as executor:

            nifty_index = executor.submit(
                fetch_and_analyze,
                nifty_symbol,
                nifty_name,
                timeframe,
                kite_local
            )
            banknifty_index = executor.submit(
                fetch_and_analyze,
                banknifty_symbol,
                banknifty_name,
                timeframe,
                kite_local
            )
            ce_future = executor.submit(
                fetch_and_analyze_option,
                kite_local,
                {
                    "symbol": settings["option_ce_symbol"],
                    "token": settings["option_ce_token"],
                    "option_type":"CE",
                    "strike":0,
                    "expiry":""
                },
                index_name,
                timeframe
            )
            pe_future = executor.submit(
                fetch_and_analyze_option,
                kite_local,
                {
                    "symbol": settings["option_pe_symbol"],
                    "token": settings["option_pe_token"],
                    "option_type":"PE",
                    "strike":0,
                    "expiry":""
                },
                index_name,
                timeframe
            )
            nifty_data = nifty_index.result()
            banknifty_data = banknifty_index.result()
            ce_data = ce_future.result()
            pe_data = pe_future.result()
            log2(f"nifty data -- {nifty_data}  -- {banknifty_index}  -- {ce_future}  -- {pe_future}")
        log2((datetime.now() - ptime).total_seconds() )
        ce_target_price = settings['ce_target_price']
        pe_target_price = settings['pe_target_price']
        if (ce_target_price < ce_data['high']):
            log2("CE entry target hit")
            process_bullish_entry(index_name,ce_data,settings, ce_target_price)
            log2(f"CE entry target hit -- {ce_target_price} -- high price - {ce_data['high']}")
            reset_entry_targets(index_name)
        elif (pe_target_price < pe_data['high']):
            log2("PE entry target hit")
            process_bearish_entry(index_name,pe_data,settings, pe_target_price)
            log2(f"PE entry target hit -- {pe_target_price} -- high price - {pe_data['high']}")
            reset_entry_targets(index_name)
        else:
            run_entry_scan(index_name, settings, nifty_data, banknifty_data, ce_data, pe_data)
            set_entry_targets(index_name, settings, ce_data, pe_data)


    except Exception as e:
        logger.error(
            f"{index_name} Process Error : {e}"
        )
def set_entry_targets(index_name, settings, ce_data, pe_data):
    max_ce = max(ce_data['price'], ce_data['kijun'], ce_data['tenkan'], ce_data['senkou_a'], ce_data['senkou_b'], ce_data['max_5'])
    max_pe = max(pe_data['price'], pe_data['kijun'], pe_data['tenkan'], pe_data['senkou_a'], pe_data['senkou_b'], pe_data['max_5'])
    ce_entry_target = max_ce + max(max_ce*0.05, 10)
    pe_entry_target = max_pe + max(max_pe*0.05, 10)
    old_ce_target_price = settings['ce_target_price']
    old_pe_target_price = settings['pe_target_price']
    update_ce = (
        old_ce_target_price == 0 or
        abs(ce_entry_target - old_ce_target_price) / old_ce_target_price > 0.01
    )
    update_pe = (
        old_pe_target_price == 0 or
        abs(pe_entry_target - old_pe_target_price) / old_pe_target_price > 0.01
    )
    log2(f"setting new entry target price for -- CE - {ce_entry_target} and for PE - {pe_entry_target}")
    with get_connection() as conn:
        if update_ce:
            conn.execute("""
                UPDATE automation_settings
                SET ce_target_price = ?
                WHERE index_name = ?
            """, (
                round(ce_entry_target, 2),
                index_name
            ))
        if update_pe:
            conn.execute("""
                UPDATE automation_settings
                SET pe_target_price = ?
                WHERE index_name = ?
            """, (
                round(pe_entry_target, 2),
                index_name
            ))

def reset_entry_targets(index_name):
    with get_connection() as conn:
        conn.execute("""
            UPDATE automation_settings
            SET
                ce_target_price=?,
                pe_target_price=?
            WHERE index_name=?
        """, (
            0, 0,
            index_name
        ))

def run_entry_scan(index_name,
            settings,
            nifty_data,
            banknifty_data,
            ce_data,
            pe_data):

    logger.info(f"Scanning {index_name}")
    log2("Scanning start")
    signal_map = {
        "Buy": 1,
        "Strong Buy": 2,
        "Very Strong Buy": 3,
        "Sell": -1,
        "Strong Sell": -2,
        "Very Strong Sell": -3
    }

    # Use .title() instead of .upper()
    log2(f"ce data ---- {ce_data}")
    index_signal_score = signal_map.get(nifty_data['signal'].title(), 0) + signal_map.get(banknifty_data['signal'].title(), 0)
    ce_signal_score = signal_map.get(ce_data['signal'].title(), 0)
    pe_signal_score = signal_map.get(pe_data['signal'].title(), 0)
    net_score = index_signal_score + ce_signal_score - pe_signal_score

    log2(f"scanning complete execution --- {index_signal_score} -- {ce_signal_score} -- {pe_signal_score}")
    if ((net_score >= 4) and (ce_signal_score >= 1) and (pe_signal_score <= -1)):
        log2("Entry in CE")
        process_bullish_entry(index_name,ce_data,settings)
    elif (net_score <= -4) and (ce_signal_score <= -1) and (pe_signal_score >= 1):
        log2("Entry in PE")
        process_bearish_entry(index_name,pe_data,settings)
    
def process_bullish_entry(index_name, ce_data, settings, target_price = 0):
    try:
        logger.info(f"settings data -- {settings}")
        logger.info(f"ce data --- {ce_data}")
        logger.info(
            f"{index_name} Bullish Entry Found "
            f"{ce_data['symbol']}"
        )
        ce_symbol = settings['option_ce_symbol']
        ce_token = settings['option_ce_token']
        pos_type = "CE"
        qty = settings['qty']
        if target_price != 0:
            entry_price = target_price
        else:
            entry_price = ce_data['price']

        sl_price = stoploss_value(ce_data,settings, entry_price)
        tg_price = target_price(ce_data,settings)

        create_position(
            index_name,
            ce_symbol,
            ce_token,
            pos_type,
            qty,
            entry_price,
            sl_price,
            tg_price
        )

        logger.info(
            f"{index_name} CE Position Created | "
            f"Symbol={ce_symbol} "
            f"Entry={entry_price} "
            f"SL={sl_price} "
            f"Target={tg_price}"
        )

        return True

    except Exception as e:

        logger.exception(
            f"{index_name} Error while creating bullish entry: {e}"
        )

        return False


def process_bearish_entry(index_name, pe_data, settings, target_price = 0):
    logger.info(f"settings data -- {settings}")
    logger.info(f"pe data --- {pe_data}")
    logger.info(
        f"{index_name} Bearish Entry Found "
        f"{pe_data['symbol']}"
    )
    pe_symbol = settings['option_pe_symbol']
    pe_token = settings['option_pe_token']
    pos_type = "PE"
    qty = settings['qty']
    if target_price != 0:
            entry_price = target_price
    else:
        entry_price = pe_data['price']
    sl_price = stoploss_value(pe_data, settings, entry_price)
    tg_price = target_price(pe_data, settings)
    log2("going to update table")
    create_position(index_name, pe_symbol, pe_token, pos_type, qty, entry_price, sl_price, tg_price)
    return True

def create_position(index_name, symbol, token, pos_type, qty, entry_price, sl_price, tg_price):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO live_positions(
            index_name,
            symbol,
            token,
            position_type,

            qty,

            entry_price,
            current_price,

            stoploss,
            target,

            status,

            entry_time
        )
        VALUES(
            ?,?,?,?,?,?,?,?,?,
            'OPEN',
            datetime('now', '+5 hours', '+30 minutes')
        )
        """,(
            index_name,
            symbol,
            token,
            pos_type,
            qty,
            entry_price,
            entry_price,
            sl_price,
            tg_price

        ))

        cur.execute("""
        UPDATE automation_settings
        SET
            position_side=?,
            last_trade_time=datetime('now', '+5 hours', '+30 minutes')
        WHERE index_name=?
        """,(
            pos_type,
            index_name
        ))

        conn.commit()

        conn.close()

        logger.info(
            f"{index_name} Position Created "
            f"{symbol} @ {entry_price}"
        )
        log2("table updated")
        return True

    except Exception as e:

        logger.error(
            f"Create Position Error : {e}"
        )

        return False


def stoploss_value(option_data, settings, entry_price = 0):
    
    risk_percent = settings['risk_percent']
    sl_base = settings['sl_base']
    sl_per = settings['sl_percent']
    sl_cap = settings['sl_cap']
    ltp = option_data['price']
    high = option_data['max_10']
    logger.info(f"sl data -- {risk_percent} -- {sl_base} -- {sl_per} -- {sl_cap} -- {high}")
    log2(f"sl data -- {risk_percent} -- {sl_base} -- {sl_per} -- {sl_cap} -- {high}")
    if sl_base == "kijun":
        sl_base_value = option_data['kijun']
    
    sl1 = high - (high *risk_percent)/100
    sl2 = sl_base_value - (sl_base_value*sl_per)/100
    sl3 = sl_base_value - sl_cap
    if (high - entry_price > 15):
        sl4 = entry_price - 5
    else:
        sl4 = entry_price - 10
    log2(f"stoploss values ---  {sl1} -- {sl2} -- {sl3} -- {sl4}")
    sl = max(sl1,sl2,sl3, sl4)

    return round(sl,2)

    
def target_price(ce_data, settings):
    ltp = ce_data['price']
    tg_type = settings['target_type']
    if tg_type == "fixed":
        tg_per = settings['target_percent']
        return (ltp + (ltp*tg_per)/100)
    return 0


def monitor_open_positions():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM live_positions
    WHERE status='OPEN'
    """)

    positions = cur.fetchall()

    conn.close()

    for position in positions:

        monitor_position(position)

def monitor_position(kite_local,position, settings):
    pos_type = position['position_type']
    sl_price = position['stoploss']
    tg_price = position['target']
    cur_price_o = position['current_price']
    entry_price = position['entry_price']
    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        ce_future = executor.submit(
            fetch_and_analyze_option,
            kite_local,
            {
                "symbol": settings["option_ce_symbol"],
                "token": settings["option_ce_token"],
                "option_type":"CE",
                "strike":0,
                "expiry":""
            },
            position["index_name"],
            settings["timeframe"]
        )
        pe_future = executor.submit(
            fetch_and_analyze_option,
            kite_local,
            {
                "symbol": settings["option_pe_symbol"],
                "token": settings["option_pe_token"],
                "option_type":"PE",
                "strike":0,
                "expiry":""
            },
            position["index_name"],
            settings["timeframe"]
        )
        ce_data = ce_future.result()
        pe_data = pe_future.result()
    if not ce_data or not pe_data:
        logger.warning("Option data unavailable")
        return
    if pos_type == "CE":
        cur_price = ce_data['price']
        low_price = ce_data['low']
    elif pos_type == "PE":
        cur_price = pe_data['price']
    log2(f"monitor prices -- {cur_price} --  sl prices -- {sl_price}")
    ## check stoploss hit or not
    if sl_price > cur_price:
        # log2("stoploss hit")
        log2("Stoploss hit -- close positions")
        close_position(position['id'], sl_price, "Stoploss Hit")
        return
    if (tg_price != 0) and (tg_price < cur_price):
        log2("Target hit -- close positions")
        close_position(position['id'], tg_price, "Target Hit")
        return
    signal_map = {
        "Buy": 1,
        "Strong Buy": 2,
        "Sell": -1,
        "Strong Sell": -2
    }
    ce_signal_score = signal_map.get(ce_data['signal'].title(), 0)
    pe_signal_score = signal_map.get(pe_data['signal'].title(), 0)
    log2(f"signal score -- {ce_signal_score} -- {pe_signal_score}")
    if pos_type == "CE":
        if (ce_signal_score <= -1) or (pe_signal_score >= 1):
            close_position(position['id'], cur_price, "Sell Condition")
            return
        else:
            sl_value = stoploss_value(ce_data, settings, entry_price)
            if (round(sl_price, 2) != round(sl_value, 2)) or (round(cur_price_o, 2) != round(cur_price, 2)):
                update_position(position['id'], sl_value, cur_price, position)

    elif pos_type == "PE":
        if (pe_signal_score <= -1) or (ce_signal_score >= 1):
            close_position(position['id'], cur_price, "Sell Condition")
            return
        else:
            sl_value = stoploss_value(pe_data, settings, entry_price)
            if (round(sl_price, 2) != round(sl_value, 2)) or (round(cur_price_o, 2) != round(cur_price, 2)):
                update_position(position['id'], sl_value, cur_price, position)
    

    return None

def update_position(position_id, new_stoploss, cur_price, position):
    try:
        pnl = calculate_pnl(position["entry_price"],cur_price,position["qty"],side="BUY")
        with get_connection() as conn:
            conn.execute("""
                UPDATE live_positions
                SET
                    stoploss = ?,
                    current_price = ?,
                    pnl = ?
                WHERE id = ?
                AND status = 'OPEN'
            """, (
                new_stoploss,
                cur_price,
                pnl,
                position_id
            ))
    except Exception:
        logger.exception(f"SL update failed: {position_id}")
        

def market_open_for_entries():

    now = datetime.now()

    if now.hour < 15:
        return True

    if now.hour == 15 and now.minute < 25:
        return True

    return False

def square_off_before_close():
    return None

def calculate_pnl(entry_price,exit_price,qty,side="BUY"):
    log2(f"pnl calculation entry price -- {exit_price}")
    log2(f"pnl calculation exit price -- {entry_price}")
    if side == "BUY":
        pnl = (float(exit_price) - float(entry_price)) * qty
    else:
        pnl = (float(entry_price) - float(exit_price)) * qty

    return round(pnl, 2)

def close_position(position_id, exit_price, exit_reason):
    conn = get_connection()
    cur = conn.cursor()
    try:
        position = cur.execute("""
            SELECT *
            FROM live_positions
            WHERE id = ?
            AND status = 'OPEN'
        """,(position_id,)).fetchone()
        if not position:
            return False
        pnl = calculate_pnl(position["entry_price"], exit_price, position["qty"])
        log2(f"pnl --- {pnl}")
        
        cur.execute("""
            INSERT INTO trade_history(
                index_name,
                symbol,
                token,
                position_type,
                qty,
                entry_price,
                exit_price,
                pnl,
                exit_reason,
                entry_time,
                exit_time
            )
            VALUES(
                ?,?,?,?,?,?,?,?,?,?,
                datetime('now', '+5 hours', '+30 minutes')
            )
        """,(
            position["index_name"],
            position["symbol"],
            position["token"],
            position["position_type"],
            position["qty"],
            position["entry_price"],
            exit_price,
            pnl,
            exit_reason,
            position["entry_time"]
        ))

        cur.execute("""
            UPDATE live_positions
            SET
                current_price = ?,
                pnl = ?,
                status = 'CLOSED',
                exit_time = datetime('now', '+5 hours', '+30 minutes')
            WHERE id = ?
        """,(
            exit_price,
            pnl,
            position_id
        ))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def close_all_positions(index_name, exit_reason="MANUAL_STOP"):

    with get_connection() as conn:

        positions = conn.execute("""
            SELECT *
            FROM live_positions
            WHERE index_name = ?
            AND status = 'OPEN'
        """, (index_name,)).fetchall()

    access_token = read_access_token()

    kite_local = KiteConnect(api_key=API_KEY)
    kite_local.set_access_token(access_token)

    closed_count = 0

    for pos in positions:

        exit_price = get_live_price(
            kite_local,
            pos["symbol"]
        ) or pos["entry_price"]

        if close_position(
            pos["id"],
            exit_price,
            exit_reason
        ):
            closed_count += 1

    return closed_count
def get_live_price(kite_local,symbol):

    try:

        quote = kite_local.ltp(
            [f"NFO:{symbol}"]
        )
        log2(quote)
        return list(
            quote.values()
        )[0]["last_price"]

    except Exception as e:

        logger.exception(
            f"Failed to fetch LTP for token {token}: {e}"
        )

        return None
def get_current_option_price(position_id):
    conn = get_connection()
    pos = conn.execute("""
        SELECT current_price
        FROM live_positions
        WHERE id = ?
    """,(
        position_id,
    )).fetchone()
    conn.close()
    if pos:
        return pos["current_price"]
    return 0

