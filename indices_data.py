import pandas as pd
import requests
import sqlite3

from io import StringIO

# ============================================
# DATABASE
# ============================================

DB_NAME = "indices_data.db"

# ============================================
# INDEX MAP
# ============================================

INDEX_MAP = {

    # BROAD MARKET
    "NIFTY 50": "nifty50",
    "NIFTY NEXT 50": "niftynext50",
    "NIFTY 100": "nifty100",
    "NIFTY 200": "nifty200",
    "NIFTY 500": "nifty500",
    "NIFTY TOTAL MKT": "niftytotalmarket_",
    "NIFTY LARGEMID250": "niftylargemidcap250",
    "NIFTY MIDSML 400": "niftymidsmallcap400",

    # MIDCAP / SMALLCAP
    "NIFTY MIDCAP 50": "niftymidcap50",
    "NIFTY MIDCAP 100": "niftymidcap100",
    "NIFTY MIDCAP 150": "niftymidcap150",
    "NIFTY SMALLCAP 50": "niftysmallcap50",
    "NIFTY SMALLCAP 100": "niftysmallcap100",
    "NIFTY SMALLCAP 250": "niftysmallcap250",
    "NIFTY MICROCAP250": "niftymicrocap250_",

    # BANKING
    "NIFTY BANK": "nifty_privatebank",
    "NIFTY PSU BANK": "niftypsubank",
    "NIFTY PVT BANK": "niftyprivatebank",
    "NIFTY FIN SERVICE": "niftyfinance",

    # IT
    "NIFTY IT": "niftyit",
    "NIFTY INTERNET": "niftyinternet",
    "NIFTY IND DIGITAL": "niftyindiadigital_",

    # AUTO
    "NIFTY AUTO": "niftyauto",
    "NIFTY EV": "niftyevnewage",
    "NIFTY MOBILITY": "niftymobility_",

    # PHARMA
    "NIFTY PHARMA": "niftypharma",
    "NIFTY HEALTHCARE": "niftyhealthcare",

    # ENERGY
    "NIFTY ENERGY": "niftyenergy",
    "NIFTY OIL AND GAS": "niftyoilgas",
    "NIFTY METAL": "niftymetal",

    # INFRA
    "NIFTY INFRA": "niftyinfra",
    "NIFTY INDIA MFG": "niftyindiamanufacturing_",
    "NIFTY MULTI MFG": "",

    # REALTY
    "NIFTY REALTY": "niftyrealty",

    # CONSUMPTION
    "NIFTY FMCG": "niftyfmcg",
    "NIFTY CONSUMPTION": "niftyconsumption",

    "NIFTY CONSR DURBL": "niftyconsumerdurables",

    # DEFENCE
    "NIFTY IND DEFENCE": "niftyindiadefence_",

    # CAPITAL MKT
    "NIFTY CAPITAL MKT": "niftycapitalmarkets_",

    "NIFTY COMMODITIES": "niftycommodities",

    "NIFTY HOUSING": "niftycorehousing_",

    "NIFTY IND TOURISM": "niftyindiatourism_",

    "NIFTY MEDIA": "niftymedia",

    "NIFTY PSE": "niftypse",

    "NIFTY SERV SECTOR": "niftyservices",

    # MOMENTUM
    "NIFTY200MOMENTM30": "nifty200momentum30_",
    "NIFTY500MOMENTM50": "nifty500momentum50_",

    # QUALITY
    "NIFTY100 QUALTY30": "nifty100quality30",
    "NIFTY500 QLTY50": "nifty500quality50_",

    # VALUE
    "NIFTY50 VALUE 20": "nifty50value20",
    "NIFTY500 VALUE 50": "nifty500value50",

    # LOW VOL
    # "NIFTY LOW VOL 50": "niftylowvolatility50",

    # ALPHA
    "NIFTY ALPHA 50": "niftyalpha50",

    # ESG
    # "NIFTY100 ESG": "nifty100esg",

    # THEMATIC
    "NIFTY CHEMICALS": "niftychemicals_",
    "NIFTY RURAL": "niftyrural_"
    # "NIFTY IPO": "niftyipo"
}
def save_all_index_watchlist():

    conn = sqlite3.connect("index_analysis.db")

    c = conn.cursor()

    # ============================================
    # CREATE WATCHLIST IF NOT EXISTS
    # ============================================

    c.execute("""
        INSERT OR IGNORE INTO index_watchlists(name)
        VALUES (?)
    """, ("ALL INDEX",))

    # ============================================
    # GET WATCHLIST ID
    # ============================================

    c.execute("""
        SELECT id
        FROM index_watchlists
        WHERE name = ?
    """, ("ALL INDEX",))

    watchlist_id = c.fetchone()[0]

    # ============================================
    # CLEAR OLD SYMBOLS
    # ============================================

    c.execute("""
        DELETE FROM index_watchlist_items
        WHERE watchlist_id = ?
    """, (watchlist_id,))

    # ============================================
    # INSERT ALL INDEX SYMBOLS
    # ============================================

    for symbol in INDEX_MAP.keys():

        c.execute("""
            INSERT OR IGNORE INTO index_watchlist_items(
                watchlist_id,
                symbol
            )
            VALUES (?, ?)
        """, (
            watchlist_id,
            symbol
        ))

    conn.commit()

    conn.close()

# save_all_index_watchlist()
# ============================================
# DOWNLOAD ALL INDICES
# ============================================

def download_all_indices():

    all_rows = []
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for index_name, slug in INDEX_MAP.items():

        url = f"https://niftyindices.com/IndexConstituent/ind_{slug}list.csv"

        try:

            print("Downloading:", index_name)

            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code != 200:
                continue

            df = pd.read_csv(StringIO(response.text))

            if 'Symbol' not in df.columns:
                continue

            df['Index'] = index_name

            all_rows.append(df)

        except Exception as e:
            print(index_name, e)

    if len(all_rows) == 0:
        return pd.DataFrame()

    final_df = pd.concat(all_rows, ignore_index=True)

    return final_df

# ============================================
# CREATE MASTER TABLE
# ============================================

def create_master_table(df):

    grouped = (
        df.groupby('Symbol')
        .agg({
            'Company Name': 'first',
            'Industry': 'first',
            'Index': lambda x: ', '.join(sorted(set(x)))
        })
        .reset_index()
    )

    # ============================================
    # MARKET CAP
    # ============================================
    grouped['Large_Cap'] = grouped['Index'].apply(
        lambda x: any(
            i.strip() in ['NIFTY 50', 'NIFTY 100', 'NIFTY NEXT 50']
            for i in str(x).split(',')
        )
    )

    grouped['Mid_Cap'] = grouped['Index'].apply(
        lambda x: any(
            i.strip() in [
                'NIFTY MIDCAP 50',
                'NIFTY MIDCAP 100',
                'NIFTY MIDCAP 150'
            ]
            for i in str(x).split(',')
        )
    )

    grouped['Small_Cap'] = grouped['Index'].apply(
        lambda x: any(
            i.strip() in [
                'NIFTY SMALLCAP 50',
                'NIFTY SMALLCAP 100',
                'NIFTY SMALLCAP 250',
                'NIFTY MICROCAP250'
            ]
            for i in str(x).split(',')
        )
    )

    # ============================================
    # SECTORS
    # ============================================

    grouped['Banking'] = grouped['Index'].str.contains(
        'BANK',
        case=False
    )

    grouped['IT'] = grouped['Index'].str.contains(
        'IT|DIGITAL',
        case=False
    )

    grouped['Pharma'] = grouped['Index'].str.contains(
        'PHARMA|HEALTHCARE',
        case=False
    )

    grouped['Auto'] = grouped['Index'].str.contains(
        'AUTO|EV|MOBILITY',
        case=False
    )

    grouped['Energy'] = grouped['Index'].str.contains(
        'ENERGY|OIL|GAS',
        case=False
    )

    grouped['Infra'] = grouped['Index'].str.contains(
        'INFRA',
        case=False
    )

    grouped['Consumption'] = grouped['Index'].str.contains(
        'FMCG|CONSUM',
        case=False
    )

    grouped['Realty'] = grouped['Index'].str.contains(
        'REALTY',
        case=False
    )

    # ============================================
    # FACTORS
    # ============================================

    grouped['Momentum'] = grouped['Index'].str.contains(
        'MOMENT',
        case=False
    )

    grouped['Quality'] = grouped['Index'].str.contains(
        'QUAL',
        case=False
    )

    grouped['Value'] = grouped['Index'].str.contains(
        'VALUE',
        case=False
    )

    grouped['Low_Volatility'] = grouped['Index'].str.contains(
        'LOW VOL',
        case=False
    )

    grouped['Index_Count'] = grouped['Index'].apply(
        lambda x: len(x.split(','))
    )

    return grouped

# ============================================
# SAVE TO SQLITE
# ============================================

def save_to_database(df):

    conn = sqlite3.connect(DB_NAME)

    df.to_sql(
        "master_indices",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

# ============================================
# CREATE INDEX STOCK LIST TABLE
# ============================================

def create_index_stock_table(df):

    index_df = df[[
        'Index',
        'Symbol'
    ]].drop_duplicates()

    index_df = index_df.sort_values([
        'Index',
        'Symbol'
    ])

    return index_df

# def create_index_stock_table(df):

#     index_df = (
#         df.groupby('Index')
#         .agg({
#             'Symbol': lambda x: str(list(sorted(set(x))))
#         })
#         .reset_index()
#     )

#     index_df.rename(
#         columns={
#             'Symbol': 'Stocks'
#         },
#         inplace=True
#     )

#     return index_df

# ============================================
# SAVE INDEX STOCK TABLE
# ============================================

def save_index_stock_table(df):

    conn = sqlite3.connect(DB_NAME)

    df.to_sql(
        "index_stock_lists",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()
# ============================================
# REFRESH DATABASE
# ============================================

def refresh_indices_data():

    raw_df = download_all_indices()

    if raw_df.empty:
        return 0

    master_df = create_master_table(raw_df)

    index_stock_df = create_index_stock_table(raw_df)

    save_to_database(master_df)

    save_index_stock_table(index_stock_df)

    return len(master_df)

# ============================================
# GET DATA
# ============================================

def get_indices_data():

    conn = sqlite3.connect(DB_NAME)

    try:

        df = pd.read_sql(
            "SELECT * FROM master_indices",
            conn
        )

        conn.close()

        return df.fillna('').to_dict(
            orient='records'
        )

    except Exception as e:

        conn.close()

        print("DB ERROR:", e)

        return []