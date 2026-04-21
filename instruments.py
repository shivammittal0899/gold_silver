import sqlite3

def init_db():
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

    if not table_exists:
        print("⚠️ Table missing → creating + reloading")
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

    # Get all equities
    cursor.execute("""
        SELECT name, tradingsymbol, instrument_token
        FROM instruments
        WHERE segment='EQ'
    """)

    equities = cursor.fetchall()

    result = []

    for name, symbol, eq_token in equities:

        # Get futures sorted by expiry
        cursor.execute("""
            SELECT tradingsymbol, instrument_token, expiry, lot_size
            FROM instruments
            WHERE name=? AND segment='FUT'
            ORDER BY expiry ASC
            LIMIT 6
        """, (name,))

        futures = cursor.fetchall()

        result.append({
            "symbol": symbol,
            "eq_token": eq_token,
            "futures": futures
        })

    conn.close()

    # 🔥 FUTURES FIRST SORTING
    result.sort(key=lambda x: len(x["futures"]) == 0)

    return result