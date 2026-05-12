import pandas as pd
import requests
from flask import Flask, render_template, jsonify
from io import StringIO



INDEX_MAP = {
    "NIFTY 50": "nifty50",
    "NIFTY BANK": "niftybank",
    "NIFTY IT": "niftyit",
    "NIFTY AUTO": "niftyauto",
    "NIFTY PHARMA": "niftypharma",
    "NIFTY FMCG": "niftyfmcg",
    "NIFTY METAL": "niftymetal",
    "NIFTY REALTY": "niftyrealty",
    "NIFTY ENERGY": "niftyenergy",
    "NIFTY MIDCAP 100": "niftymidcap100",
    "NIFTY SMALLCAP 100": "niftysmallcap100",
    "NIFTY PSU BANK": "niftypsubank",
    "NIFTY PVT BANK": "niftypvtbank",
    "NIFTY FIN SERVICE": "niftyfinservice",
    "NIFTY INFRA": "niftyinfra",
    "NIFTY MEDIA": "niftymedia",
    "NIFTY CONSUMPTION": "niftyconsumption",
    "NIFTY OIL AND GAS": "niftyoilgas",
    "NIFTY HEALTHCARE": "niftyhealthcare",
    "NIFTY MIDCAP 50": "niftymidcap50",
    "NIFTY SMALLCAP 250": "niftysmallcap250"
}

# =========================
# DOWNLOAD ALL INDICES
# =========================

def download_all_indices():

    all_rows = []

    for index_name, slug in INDEX_MAP.items():

        url = f"https://niftyindices.com/IndexConstituent/ind_{slug}list.csv"

        try:
            response = requests.get(url, timeout=20)

            if response.status_code != 200:
                print("Failed:", index_name)
                continue

            df = pd.read_csv(StringIO(response.text))

            if 'Symbol' not in df.columns:
                continue

            df['Index'] = index_name

            all_rows.append(df)

            print(f"Downloaded {index_name}")

        except Exception as e:
            print(index_name, e)

    if len(all_rows) == 0:
        return pd.DataFrame()

    final_df = pd.concat(all_rows, ignore_index=True)

    return final_df

# =========================
# CREATE MASTER TABLE
# =========================

def create_master_table(df):

    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby('Symbol')
        .agg({
            'Company Name': 'first',
            'Industry': 'first',
            'Index': lambda x: ', '.join(sorted(set(x)))
        })
        .reset_index()
    )

    grouped['Index Count'] = grouped['Index'].apply(
        lambda x: len(x.split(','))
    )

    grouped['Large Cap'] = grouped['Index'].str.contains('NIFTY 50|NIFTY 100')

    grouped['Mid Cap'] = grouped['Index'].str.contains('MIDCAP')

    grouped['Small Cap'] = grouped['Index'].str.contains('SMALLCAP')

    grouped['Banking'] = grouped['Index'].str.contains('BANK')

    grouped['IT'] = grouped['Index'].str.contains('IT')

    grouped['Pharma'] = grouped['Index'].str.contains('PHARMA|HEALTHCARE')

    grouped['Auto'] = grouped['Index'].str.contains('AUTO|MOBILITY|EV')

    grouped['Momentum'] = grouped['Index'].str.contains('MOMENT')

    grouped['Quality'] = grouped['Index'].str.contains('QUAL')

    grouped['Value'] = grouped['Index'].str.contains('VALUE')

    grouped['Low Volatility'] = grouped['Index'].str.contains('LOWVOL')

    return grouped
