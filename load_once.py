
import sqlite3
import os

API_KEY = "0qw10pvn638g9jid"
API_SECRET = "6przxtyeeoi9jtvyx76qga4hrv7q86qr"
REDIRECT_URL = "http://localhost:8000/callback"

INSTRUMENT_MAP = {}
def load_instruments_once():

    global INSTRUMENT_MAP

    conn = sqlite3.connect("instruments.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tradingsymbol, instrument_token
        FROM instruments
    """)
    rows = cursor.fetchall()
    conn.close()
    # 🔥 BUILD MAP
    INSTRUMENT_MAP = {
        row[0]: row[1]
        for row in rows
    }
    # log1(f"✅ Loaded {len(INSTRUMENT_MAP)} instruments into memory")

load_instruments_once()

def save_access_token(token):
    with open("access_token.txt", "w") as f:
        f.write(token)

# ---------------------- READ TOKEN ----------------------
def read_access_token():
    if os.path.exists("access_token.txt"):
        with open("access_token.txt") as f:
            return f.read().strip()
    return None