import sqlite3
def log2(msg):
    with open("static/logs.txt", "a") as f:
        f.write(f"{msg}\n")
def init_db():
    log2("creating db")
    conn = sqlite3.connect("instruments.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS instruments (
        tradingsymbol TEXT,
        name TEXT,
        instrument_token INTEGER,
        segment TEXT,
        expiry DATE,
        lot_size INTEGER,
        PRIMARY KEY (tradingsymbol, instrument_token)
    )
    """)

    conn.commit()
    conn.close()


import sqlite3
import os

def ensure_instruments_data(kite):
    db_path = "instruments.db"
    log2("checking database")
    # 🔹 Step 1: DB exists?
    db_exists = os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 🔹 Step 2: Table exists?
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='instruments'
    """)
    table_exists = cursor.fetchone()

    log2("checking table")
    if not table_exists:
        log2("⚠️ Table missing → creating + reloading")
        conn.close()
        init_db()
        reload_instruments(kite)
        return

    # 🔹 Step 3: Data exists?
    cursor.execute("SELECT COUNT(*) FROM instruments")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        print("⚠️ Empty DB → reloading instruments")
        reload_instruments(kite)

#  {'instrument_token': 864001,
#   'exchange_token': '3375',
#   'tradingsymbol': 'SURYAROSNI',
#   'name': 'SURYA ROSHNI',
#   'last_price': 0.0,
#   'expiry': '',
#   'strike': 0.0,
#   'tick_size': 0.01,
#   'lot_size': 1,
#   'instrument_type': 'EQ',
#   'segment': 'NSE',
#   'exchange': 'NSE'},


def reload_instruments(kite):
    conn = sqlite3.connect("instruments.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM instruments")

    # EQUITY
    for item in kite.instruments("NSE"):
        cursor.execute("""
            INSERT INTO instruments VALUES (?, ?, ?, ?, ?, ?)
        """, (
            item["tradingsymbol"],
            item["name"],
            item["instrument_token"],
            "EQ",
            None,
            1
        ))

    # FUTURES ONLY
    log2("fetching data from kite")
    for item in kite.instruments("NFO"):
        if item["instrument_type"] == "FUT":
            cursor.execute("""
                INSERT INTO instruments VALUES (?, ?, ?, ?, ?, ?)
            """, (
                item["tradingsymbol"],
                item["name"],
                item["instrument_token"],
                "FUT",
                item["expiry"],
                item["lot_size"]
            ))

    conn.commit()
    conn.close()

def get_stock_with_futures():
    conn = sqlite3.connect("instruments.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            eq.tradingsymbol AS eq_symbol,
            eq.instrument_token AS eq_token,
            fut.tradingsymbol AS fut_symbol,
            fut.instrument_token AS fut_token,
            fut.expiry,
            fut.lot_size
        FROM instruments eq
        LEFT JOIN instruments fut
            ON eq.tradingsymbol = fut.name   -- ✅ YOUR MATCHING LOGIC
            AND fut.segment = 'FUT'
        WHERE eq.segment = 'EQ'
        ORDER BY eq.tradingsymbol, fut.expiry ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    # 🔥 Grouping
    result_map = {}

    for row in rows:
        eq_symbol, eq_token, fut_symbol, fut_token, expiry, lot = row

        if eq_symbol not in result_map:
            result_map[eq_symbol] = {
                "symbol": eq_symbol,
                "eq_token": eq_token,
                "futures": []
            }

        # limit to 3 futures
        if fut_symbol and len(result_map[eq_symbol]["futures"]) < 3:
            result_map[eq_symbol]["futures"].append(
                (fut_symbol, fut_token, expiry, lot)
            )

    result = list(result_map.values())

    # 🔥 Futures first
    result.sort(key=lambda x: len(x["futures"]) == 0)

    return result