# db.py

import sqlite3

DB_NAME_OP = "options_automation.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME_OP)
    # conn.row_factory = sqlite3.Row
    return conn


def create_tables():

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE option_user (
        user_id INTEGER PRIMARY KEY,
        timeframe TEXT,
        nifty_strikes TEXT,
        banknifty_strikes TEXT,
        nifty_qty INTEGER,
        nifty_risk REAL,
        nifty_target REAL,
        nifty_sl_base TEXT,2
        nifty_sl_percent REAL,
        nifty_refresh INTEGER,
        banknifty_qty INTEGER,
        banknifty_risk REAL,
        banknifty_target REAL,
        banknifty_sl_base TEXT,
        banknifty_sl_percent REAL,
        banknifty_refresh INTEGER
    );
    """)
    # Option master
    cur.execute("""
    CREATE TABLE IF NOT EXISTS option_master(
        symbol TEXT PRIMARY KEY,
        index_name TEXT,
        strike INTEGER,
        option_type TEXT,
        expiry TEXT,
        token INTEGER UNIQUE,
        exchange TEXT,
        lot_size INTEGER,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    

    # User automation settings
    # cur.execute("""
    # CREATE TABLE IF NOT EXISTS automation_settings(
    #     index_name TEXT PRIMARY KEY,
    #     enabled INTEGER DEFAULT 0,
    #     qty INTEGER DEFAULT 1,
    #     risk_percent REAL DEFAULT 1,
    #     target_type TEXT DEFAULT 'fixed',
    #     target_percent REAL DEFAULT 30,
    #     sl_base TEXT DEFAULT 'kijun',
    #     sl_percent REAL DEFAULT 5,
    #     sl_cap REAL DEFAULT 0,
    #     refresh_seconds INTEGER DEFAULT 60,
    #     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    # )
    # """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS automation_settings(
        index_name TEXT PRIMARY KEY,
        option_ce_symbol TEXT,
        option_pe_symbol TEXT,
        option_ce_token INTEGER,
        option_pe_token INTEGER,
        timeframe TEXT,
        lot_size INTEGER,
        enabled INTEGER DEFAULT 0,
        qty INTEGER DEFAULT 1,
        risk_percent REAL DEFAULT 1,
        target_type TEXT DEFAULT 'fixed',
        target_percent REAL DEFAULT 30,
        sl_base TEXT DEFAULT 'kijun',
        sl_percent REAL DEFAULT 5,
        sl_cap REAL DEFAULT 0,
        refresh_seconds INTEGER DEFAULT 60,
        position_side TEXT DEFAULT 'NONE',
        last_trade_time TIMESTAMP,
        last_error TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Current live position
    cur.execute("""
    CREATE TABLE IF NOT EXISTS live_positions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        index_name TEXT,
        symbol TEXT,
        token INTEGER,
        position_type TEXT,
        qty INTEGER,
        entry_price REAL,
        current_price REAL,
        stoploss REAL,
        target REAL,
        pnl REAL DEFAULT 0,
        status TEXT DEFAULT 'OPEN',
        entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        exit_time TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        index_name TEXT,
        symbol TEXT,
        token INTEGER,
        position_type TEXT,
        qty INTEGER,
        entry_price REAL,
        exit_price REAL,
        pnl REAL,
        exit_reason TEXT,
        entry_time TIMESTAMP,
        exit_time TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    print("DB Created")