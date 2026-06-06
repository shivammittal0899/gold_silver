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
        
        log2("in option analysis")
        # log2(f"{symbol} -- {strike} -- {option_type} -- {index} -- {expiry} ")
        df = get_historical_data(item['token'], "option", timeframe, kite_local)
        log2(df.tail(5))
        
        if df is None or len(df) < 2:
            return None
        analysis, df = stock_data_analysis(df, timeframe)
        log2(f"data analysis -- {analysis}")
        analysis['symbol'] = symbol
        analysis['strike'] = item['strike']
        analysis['type'] = item['type']
        analysis['expiry'] = item['expiry']
        log2(analysis)
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
        
        log2(token)
        # Define date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        
        # Fetch data
        data = kite_local.historical_data(
            instrument_token=int(token),
            from_date=from_date.date(),
            to_date=to_date.date(),
            interval=timeframe
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
        CURRENT_TIMESTAMP
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

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM live_positions
    WHERE index_name=?
    AND status='OPEN'
    """,(index_name,))

    row = cur.fetchone()

    conn.close()

    return row
def get_automation_settings(index_name):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM automation_settings
        WHERE index_name=?
        AND enabled=1
    """, (index_name,))

    row = cur.fetchone()

    conn.close()

    if row:
        return dict(row)

    return None


def automation_loop(index_name):
    logger.info(
        f"{index_name} automation started"
    )
    access_token = read_access_token()
    kite_local = KiteConnect(api_key=API_KEY)
    kite_local.set_access_token(access_token)
    settings = get_automation_settings(index_name)
    if not settings:
        logger.info(
            f"{index_name} settings not found"
        )
        automation_flags[index_name] = False

    while automation_flags.get(index_name, False):
        try:
            process_index(kite_local, index_name, settings)
            # square_off_before_close()
        except Exception as e:
            logger.error(
                f"{index_name} Error : {e}"
            )
            automation_flags[index_name] = False
            break
        time.sleep(30)
    logger.info(
        f"{index_name} automation stopped"
    )


def process_index(kite_local, index_name,settings):

    try:

        # No entries after market cutoff
        if not market_open_for_entries():
            return
        # Check existing position
        position = has_open_position(index_name)
        if position:
            # monitor_position(position)
            return
        timeframe = settings['timeframe']
        nifty_symbol = "NSE:NIFTY 50"
        nifty_name = "NIFTY 50"
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
                nifty_symbol,
                nifty_name,
                timeframe,
                kite_local
            )
            ce_future = executor.submit(
                fetch_and_analyze_option,
                kite_local,
                {
                    "symbol": settings["option_ce_symbol"],
                    "token": settings["option_ce_token"],
                    "type":"CE",
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
                    "type":"PE",
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

        run_entry_scan(
            index_name,
            settings,
            nifty_data,
            banknifty_data,
            ce_data,
            pe_data
        )

    except Exception as e:
        logger.error(
            f"{index_name} Process Error : {e}"
        )

def run_entry_scan(index_name,
            settings,
            nifty_data,
            banknifty_data,
            ce_data,
            pe_data):

    logger.info(f"Scanning {index_name}")
    log2("Scanning start")

    # signal_data = get_index_analysis(
    #     index_name
    # )

    # if not signal_data:
    #     return

    # signal = signal_data["signal"]

    # if signal == "BUY":

    #     process_bullish_entry(
    #         index_name,
    #         signal_data
    #     )

    # elif signal == "SELL":

    #     process_bearish_entry(
    #         index_name,
    #         signal_data
    #     )



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

def monitor_position(position):
    """
    1 Update Current Price

    2 Update Stoploss

    3 Check Stoploss

    4 Check Sell Condition

    5 Check Target

    6 Update PnL

    SL Base
    EMA
    TENKAN
    KIJUN
    """
    return None

def market_open_for_entries():

    now = datetime.now()

    if now.hour < 15:
        return True

    if now.hour == 15 and now.minute < 25:
        return True

    return False

def square_off_before_close():
    return None