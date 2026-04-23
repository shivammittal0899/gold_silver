import sqlite3


def init_watchlist_db():
    conn = sqlite3.connect("stocks_analysis.db")
    c = conn.cursor()

    # Watchlist names
    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    """)

    # Stocks inside watchlist
    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlist_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        watchlist_id INTEGER,
        symbol TEXT,
        FOREIGN KEY(watchlist_id) REFERENCES watchlists(id)
    )
    """)

    conn.commit()
    conn.close()