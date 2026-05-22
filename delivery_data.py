import sqlite3
import pandas as pd
# def init_delivery_db():
#     conn = sqlite3.connect(
#         "delivery_history.db"
#     )
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS delivery_history (
#             date TEXT,
#             symbol TEXT,
#             ttl_trd_qnty INTEGER,
#             turnover_lacs REAL,
#             no_of_trades INTEGER,
#             deliv_qty INTEGER,
#             deliv_per REAL,
#             PRIMARY KEY (
#                 date,
#                 symbol
#             )
#         )
#     """)
#     conn.commit()
#     conn.close()
# def update_delivery_data(df):
#     conn = sqlite3.connect(
#         "delivery_history.db"
#     )
#     cursor = conn.cursor()
#     # CLEAN
#     df.columns = [
#         c.strip()
#         for c in df.columns
#     ]
#     for _, row in df.iterrows():
#         try:
#             cursor.execute("""
#                 INSERT OR REPLACE INTO delivery_history (
#                     date,
#                     symbol,
#                     ttl_trd_qnty,
#                     turnover_lacs,
#                     no_of_trades,
#                     deliv_qty,
#                     deliv_per
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 row["DATE1"],
#                 row["SYMBOL"].strip(),
#                 int(row["TTL_TRD_QNTY"]),
#                 float(row["TURNOVER_LACS"]),
#                 int(row["NO_OF_TRADES"]),
#                 int(row["DELIV_QTY"]),
#                 float(row["DELIV_PER"])
#             ))
#         except Exception as e:
#             print(e)
#     conn.commit()
#     conn.close()
#     print("✅ Delivery data updated")

def init_delivery_db():
    conn = sqlite3.connect(
        "delivery_history.db"
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS delivery_history (
            date TEXT,
            symbol TEXT,
            close_price REAL,
            change_per REAL,
            ttl_trd_qnty INTEGER,
            turnover_lacs REAL,
            no_of_trades INTEGER,
            deliv_qty INTEGER,
            deliv_per REAL,
            volume_ratio REAL,
            delivery_score REAL,
            PRIMARY KEY (
                date,
                symbol
            )
        )
    """)
    conn.commit()
    conn.close()

init_delivery_db()
def update_delivery_data_file(df):
    conn = sqlite3.connect(
        "delivery_history.db"
    )
    cursor = conn.cursor()
    df = df.dropna(
        subset=[
            "SYMBOL"
        ]
    )
    # CLEAN COLUMNS
    df.columns = [
        c.strip()
        for c in df.columns
    ]
    # CLEAN NUMERIC COLUMNS

    numeric_cols = [

        "LAST_PRICE",
        "PREV_CLOSE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
        "NO_OF_TRADES",
        "DELIV_QTY",
        "DELIV_PER"

    ]

    for col in numeric_cols:

        df[col] = (

            df[col]

            .astype(str)

            .str.replace(',', '')

            .str.strip()

        )

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"

        )

    # ====================================
    # CHANGE %
    # ====================================

    df["change_per"] = ((df["LAST_PRICE"] - df["PREV_CLOSE"])/df["PREV_CLOSE"]) * 100
    # ====================================
    # VOLUME RATIO
    # ====================================

    df["volume_ratio"] = (

        df["TTL_TRD_QNTY"]

        /

        df["TTL_TRD_QNTY"].mean()

    )

    # ====================================
    # DELIVERY SCORE
    # ====================================

    df["delivery_score"] = (df["DELIV_PER"] * 0.6 + df["volume_ratio"] * 40)

    # ====================================
    # SAVE
    # ====================================

    for _, row in df.iterrows():

        try:

            cursor.execute("""

                INSERT OR REPLACE INTO delivery_history (

                    date,
                    symbol,
                    close_price,
                    change_per,
                    ttl_trd_qnty,
                    turnover_lacs,
                    no_of_trades,
                    deliv_qty,
                    deliv_per,
                    volume_ratio,
                    delivery_score

                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                row["DATE1"],

                row["SYMBOL"].strip(),

                float(row["LAST_PRICE"]),

                float(row["change_per"]),

                int(row["TTL_TRD_QNTY"]),

                float(row["TURNOVER_LACS"]),

                int(row["NO_OF_TRADES"]),

                int(row["DELIV_QTY"]),

                float(row["DELIV_PER"]),

                float(row["volume_ratio"]),

                float(row["delivery_score"])

            ))

        except Exception as e:

            print(e)

    conn.commit()

    conn.close()

    print("✅ Delivery data updated")