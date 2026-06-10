from ta.momentum import RSIIndicator
from ta.trend import IchimokuIndicator, EMAIndicator
from ta import volatility
from ta.volume import VolumeWeightedAveragePrice
import pandas as pd
import json
import sqlite3
from ta.trend import ADXIndicator

def indicator_values(df):
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    df["ema"] = EMAIndicator(df["Close"], window=9).ema_indicator()
    df["ema_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    adx_indicator = ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    )

    df['ADX'] = adx_indicator.adx()
    df['DI_PLUS'] = adx_indicator.adx_pos()
    df['DI_MINUS'] = adx_indicator.adx_neg()
    ichi = IchimokuIndicator(
        high=df['High'],
        low=df['Low'],
        window1=9,
        window2=26,
        window3=52
    )

    df['tenkan'] = ichi.ichimoku_conversion_line()
    df['kijun'] = ichi.ichimoku_base_line()
    df['senkou_af'] = ichi.ichimoku_a()
    df['senkou_bf'] = ichi.ichimoku_b()
    df['senkou_a'] = df['senkou_af'].shift(26)
    df['senkou_b'] = df['senkou_bf'].shift(26)

    vwap = VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume']
    )

    df['VWAP'] = vwap.volume_weighted_average_price()

    df['tenkan_kijun_max'] = df[['tenkan','kijun']].max(axis=1)
    df['tenkan_kijun_min'] = df[['tenkan','kijun']].min(axis=1)
    df['cloud_max'] = df[['senkou_a','senkou_b']].max(axis=1)
    df['cloud_min'] = df[['senkou_a','senkou_b']].min(axis=1)
    df['price_max'] = df[['Close','Open']].max(axis=1)
    df['price_min'] = df[['Close','Open']].min(axis=1)
    df['price_line_max'] = (df['tenkan_kijun_max'] >= df['price_max']).rolling(6).sum() >= 4
    df['price_line_min'] = (df['tenkan_kijun_min'] <= df['price_min']).rolling(6).sum() >= 4
    df['price_line'] = df['price_line_max'] & df['price_line_min']
    return df



def find_swings(df, window=5):
    df['rolling_max'] = df['High'].rolling(window).max()
    df['rolling_min'] = df['Low'].rolling(window).min()

    df['swing_high'] = df['High'].where(df['High'] == df['rolling_max'])
    df['swing_low'] = df['Low'].where(df['Low'] == df['rolling_min'])
    return df

def highlow_data(df):
    df = find_swings(df)

    swings = []

    # collect swing highs/lows
    for i in range(len(df)):
        if not pd.isna(df['swing_high'].iloc[i]):
            swings.append(("HIGH", df['High'].iloc[i]))

        elif not pd.isna(df['swing_low'].iloc[i]):
            swings.append(("LOW", df['Low'].iloc[i]))

    # need at least 4 swings to get 3 structures
    if len(swings) < 4:
        return []

    structure = []

    # only process last 4 swings → gives last 3 structure points
    recent_swings = swings[-10:]

    for i in range(1, len(recent_swings)):
        prev_type, prev_price = recent_swings[i-1]
        curr_type, curr_price = recent_swings[i]

        if curr_type == "HIGH" and prev_type == "HIGH":
            if curr_price > prev_price:
                structure.append("HH")
            else:
                structure.append("LH")

        elif curr_type == "LOW" and prev_type == "LOW":
            if curr_price > prev_price:
                structure.append("HL")
            else:
                structure.append("LL")

    return structure

def highlow_trend(df):
    df = find_swings(df)
    lookback=4
    # swing_highs = df[df['swing_high'].notna()]['High']
    # swing_lows  = df[df['swing_low'].notna()]['Low']
    swing_highs = df[df['swing_high'].notna()]['High'].tail(lookback).values
    swing_lows  = df[df['swing_low'].notna()]['Low'].tail(lookback).values

    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "UNKNOWN", [], None

    hh = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] > swing_highs[i-1])
    lh = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] < swing_highs[i-1])

    hl = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] > swing_lows[i-1])
    ll = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] < swing_lows[i-1])

    # scoring
    up_score = hh + hl
    down_score = lh + ll

    if up_score >= (lookback - 1) * 1.4:
        trend = "STRONG UPTREND"
    elif down_score >= (lookback - 1) * 1.4:
        trend = "STRONG DOWNTREND"
    elif up_score > down_score:
        trend = "UPTREND"
    elif down_score > up_score:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"
    high_list = [(float(swing_highs[-1])),(float(swing_highs[-2]))]
    # return trend, swing_highs[-1], swing_lows[-1]
    return trend, high_list, swing_lows[-1]

def sr_breakout(df, lookback=10):
    # Support & Resistance
    resistance = df['High'].rolling(lookback).max().iloc[-2]
    support    = df['Low'].rolling(lookback).min().iloc[-2]

    current_price = df['Close'].iloc[-1]

    breakout_type = "NONE"

    if current_price > resistance:
        breakout_type = "UP"

    elif current_price < support:
        breakout_type = "DOWN"

    return {
        "support": float(support),
        "resistance": float(resistance),
        "breakout": breakout_type
    }

def pullback(df):
    price = df['Close'].iloc[-1]
    ema20 = df['ema_20'].iloc[-1]

    return abs(price - ema20) / ema20 < 0.01



def volatility_analysis(df):
    df['atr'] = volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close'], window=14
    ).average_true_range()

    atr = df['atr'].iloc[-1]
    atr_avg = df['atr'].rolling(50).mean().iloc[-1]

    regime = "NORMAL"
    if atr > 1.5 * atr_avg:
        regime = "HIGH_VOL"
    elif atr < 0.7 * atr_avg:
        regime = "LOW_VOL"

    expansion = atr > float(df['atr'].rolling(20).mean().iloc[-1])

    return {
        "atr_val": float(round(atr,2)),
        "volatility_regime": regime,
        # "volatility_exp": expansion
        "volatility_exp": "Expansion" if expansion else "Contraction"
    }

def volatility_per_analysis(df, timeframe):
    import numpy as np

    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # MCX-specific factors
    if timeframe == "5m":
        factor = np.sqrt(252 * 174)
    elif timeframe == "15m":
        factor = np.sqrt(252 * 58)
    elif timeframe == "1h":
        factor = np.sqrt(252 * 14)
    else:
        factor = np.sqrt(252)

    df['vol_5'] = df['returns'].rolling(5).std() * factor * 100
    df['vol_20'] = df['returns'].rolling(20).std() * factor * 100

    current_vol = df['vol_20'].iloc[-1]

    # Dynamic thresholds (better than fixed)
    avg_vol = float(df['vol_20'].rolling(50).mean().iloc[-1])

    return {
        "volatility_per": float(current_vol),
        # "avg_volatility": avg_vol
        "avg_volatility": "None"
    }
def ichimoku_analysis(df):
    tenkan = df['tenkan'].iloc[-1]
    kijun = df['kijun'].iloc[-1]
    senkou_a = df['senkou_a'].iloc[-1]
    senkou_b = df['senkou_b'].iloc[-1]
    senkou_af = df['senkou_af'].iloc[-1]
    senkou_bf = df['senkou_bf'].iloc[-1]
    close = df['Close'].iloc[-1]
    open = df['Open'].iloc[-1]
    cloud = "green" if senkou_a >= senkou_b else "red"
    cloud_max = max(senkou_a, senkou_b)
    cloud_min = min(senkou_a, senkou_b)
    cloud_maxf = max(senkou_af, senkou_bf)
    cloud_minf = min(senkou_af, senkou_bf)
    sideways = df['price_line'].iloc[-1]

    
    if (tenkan > kijun) and (tenkan > cloud_max) and (tenkan > cloud_maxf) and (close > cloud_max):
        tenkan_kijun = "Strong Uptrend"
    elif (tenkan >= kijun) and (close > cloud_max):
        tenkan_kijun = "Uptrend"
    elif (tenkan < kijun) and (tenkan < cloud_min) and (tenkan < cloud_minf) and (close < cloud_min):
        tenkan_kijun = "Strong Downtrend"
    elif (tenkan <= kijun) and (close < cloud_min):
        tenkan_kijun = "Downtrend"
    else:
        tenkan_kijun = "Neutral"
    
    if (tenkan_kijun == "Strong Uptrend") and (close > tenkan):
        price_tenkan = "Strong Uptrend"
    elif (tenkan_kijun == "Uptrend") and (close > tenkan):
        price_tenkan = "Uptrend"
    elif (tenkan_kijun == "Strong Downtrend") and (close < tenkan):
        price_tenkan = "Strong Downtrend"
    elif (tenkan_kijun == "Downtrend") and (close < tenkan):
        price_tenkan = "Downtrend"
    else:
        price_tenkan = "Sideways"
    
    if (tenkan_kijun == "Uptrend") and (cloud == 'green'):
        cloud_trend = "Strong Uptrend"
    elif (close > cloud_max):
        cloud_trend = "Uptrend"
    elif (tenkan_kijun == "Downtrend") and (cloud == 'red'):
        cloud_trend = "Strong Downtrend"
    elif (close < cloud_min):
        cloud_trend = "Downtrend"
    else:
        cloud_trend = "Sideways"

    
    ichimoku_t = {
        "tenkan_kijun": tenkan_kijun,
        "price_tenkan": price_tenkan,
        "cloud_trend": cloud_trend
    }
    return ichimoku_t

def get_adx_strength_signal(df):
    row = df.iloc[-1]
    adx = row['ADX']
    di_plus = row['DI_PLUS']
    di_minus = row['DI_MINUS']

    if adx > 40:
        if di_plus > di_minus:
            return 2
        elif di_minus > di_plus:
            return -2

    elif adx > 25:
        if di_plus > di_minus:
            return 1
        elif di_minus > di_plus:
            return -1

    return 0
def signal_fun(data, df, ins_type):
    signal = "Neutral"
    if ins_type == "option":
        ret6 = data['ret6']
        if ret6 > 2:
            ret6_ = 1
        elif ret6 > 0:
            ret6_ = 0
        else:
            ret6_ = -1
        ret12 = data['ret12']
        if ret12 > 4:
            ret12_ = 1
        elif ret12 > -2:
            ret12_ = 0
        else:
            ret12_ = -1
    else:
        ret6 = data['ret6']
        if ret6 > 0.05:
            ret6_ = 1
        elif ret6 > 0:
            ret6_ = 0
        else:
            ret6_ = -1
        ret12 = data['ret12']
        if ret12 > 0.05:
            ret12_ = 1
        elif ret12 > -2:
            ret12_ = 0
        else:
            ret12_ = -1
    
    trend = data['trend']
    if trend == "STRONG UPTREND":
        trend_ = 2
    if trend == "UPTREND":
        trend_ = 1
    elif trend == "SIDEWAYS":
        trend_ = 0
    elif trend == "DOWNTREND":
        trend_ = -1
    elif trend == "STRONG DOWNTREND":
        trend_ = -2
    else:
        trend_ = 0
    
    tenkan_kijun = data['tenkan_kijun']
    if tenkan_kijun == "Strong Uptrend":
        tenkan_kijun_ = 2
    elif tenkan_kijun == "Uptrend":
        tenkan_kijun_ = 1
    elif tenkan_kijun == "Downtrend":
        tenkan_kijun_ = -1
    elif tenkan_kijun == "Strong Downtrend":
        tenkan_kijun_ = -2
    else:
        tenkan_kijun_ = 0
    
    price_tenkan = data['price_tenkan']
    if price_tenkan == "Strong Uptrend":
        price_tenkan_ = 2
    elif price_tenkan == "Uptrend":
        price_tenkan_ = 1
    elif price_tenkan == "Downtrend":
        price_tenkan_ = -1
    elif price_tenkan == "Strong Downtrend":
        price_tenkan_ = -2
    else:
        price_tenkan_ = 0
    rsi = data['rsi']
    if rsi > 70:
        rsi_ = 2
    elif rsi > 55:
        rsi_ = 1
    elif rsi < 30:
        rsi_ = -2
    elif rsi < 45:
        rsi_ = -1
    else:
        rsi_ = 0

    if data['vwap'] == "Above":
        vwap = 2
    else:
        vwap = 0
    signal_sum = ret6_ + ret12_ + trend_ + tenkan_kijun_ + price_tenkan_ + rsi_ + data['adx_signal'] + vwap
    
    if signal_sum >= 7:
        signal = "Strong Buy"
    elif signal_sum >= 3:
        signal = "Buy"
    elif signal_sum <= -7:
        signal = "Strong Sell"
    elif signal_sum <= -3:
        signal = "Sell"
    else:
        signal = "Neutral"

    
    return {"signal": signal}

def data_analysis(df, timeframe):

    df = indicator_values(df)
    df = df[-50:]
    # log()
    # print(df.tail())
    
    price = df['Close'].iat[-1]
    vwap_v = df['VWAP'].iat[-1]
    # ret6 = round((((price / df['Open'].iat[-6]) - 1)*100),2)
    ret6 = (price - df['Open'].iat[-6])
    # ret12 = round((((price / df['Open'].iat[-12]) - 1)*100),2)
    ret12 = (price - df['Open'].iat[-12])
    trend, last_high, last_low = highlow_trend(df)
    highlow = highlow_data(df)
    if isinstance(highlow, list):
        highlow = json.dumps(highlow)
    srb = sr_breakout(df)
    volatility = volatility_analysis(df)
    volatility_per = volatility_per_analysis(df, timeframe)
    ichimoku_d = ichimoku_analysis(df)
    if price > vwap_v:
        vwap_a = "Above"
    else:
        vwap_a = "Below"

    data = {
        "price": float(price),
        "ret6": float(ret6),
        "ret12": float(ret12),
        "trend": trend,
        "l_high": (last_high),
        "l_low": float(last_low),
        "highlow": highlow,
        "vwap": vwap_a,
        "rsi": float(round(df['RSI'].iloc[-1],2)),
        "adx": round(float(df['ADX'].iloc[-1]), 2),
        "volume": "High"
    }
    data.update(volatility if isinstance(volatility, dict) else {})
    data.update(volatility_per if isinstance(volatility_per, dict) else {})
    data.update(ichimoku_d if isinstance(ichimoku_d, dict) else {})
    signal = signal_fun(data, df)
    data.update(signal if isinstance(signal, dict) else {})
    return data


def stock_data_analysis(df, timeframe, ins_type = "equity"):

    df = indicator_values(df)
    df = df[-100:]
    # log()
    # print(df.tail())
    
    price = df['Close'].iat[-1]
    vwap_v = df['VWAP'].iat[-1]
    ret1 = round((((price / df['Close'].iat[-2]) - 1)*100),2)
    ret6 = round((((price / df['Open'].iat[-6]) - 1)*100),2)
    ret12 = round((((price / df['Open'].iat[-12]) - 1)*100),2)
    trend, last_high, last_low = highlow_trend(df)
    highlow = highlow_data(df)
    if isinstance(highlow, list):
        highlow = json.dumps(highlow)
    srb = sr_breakout(df)
    volatility = volatility_analysis(df)
    volatility_per = volatility_per_analysis(df, timeframe)
    ichimoku_d = ichimoku_analysis(df)
    adx_signal = get_adx_strength_signal(df)
    if price > vwap_v:
        vwap_a = "Above"
    else:
        vwap_a = "Below"

    data = {
        "price": float(price),
        "ret1": round(float(ret1), 2),
        "ret6": round(float(ret6), 2),
        "ret12": round(float(ret12), 2),
        "trend": trend,
        "l_high": (last_high),
        # "l_high": float(last_high),
        "l_low": float(last_low),
        "highlow": highlow,
        "vwap": vwap_a,
        "rsi": float(round(df['RSI'].iloc[-1],2)),
        "adx": 25,
        "volume": "High",
        "adx_signal": adx_signal,
    }
    data.update(volatility if isinstance(volatility, dict) else {})
    data.update(volatility_per if isinstance(volatility_per, dict) else {})
    data.update(ichimoku_d if isinstance(ichimoku_d, dict) else {})
    signal = signal_fun(data, df, ins_type)
    data.update(signal if isinstance(signal, dict) else {})
    return data, df

def stock_data_analysis_common(df):
    price = df['Close'].iat[-1]
    high = df['High'].iat[-1]
    low = df['Low'].iat[-1]
    weekHigh = max(df['High'].tail(5))
    weekLow = min(df['Low'].tail(5))
    monthHigh = max(df['High'].tail(20))
    monthLow = min(df['Low'].tail(20))
    yearHigh = max(df['High'].tail(5*48))
    yearLow = min(df['Low'].tail(5*48))
    data = {
        'ret1': float(((price/df['Close'].iat[-2])-1)*100),
        'ret5': float(((price/df['Open'].iat[-5])-1)*100),
        'ret15': float(((price/df['Open'].iat[-10]) - 1)*100),
        'ret30': float(((price/df['Open'].iat[-20]) - 1)*100),
        'ret90': float(((price/df['Open'].iat[-60]) - 1)*100),
        'retdayHigh': float(((price/high) - 1)*100),
        'retdayLow': float(((price/low) - 1)*100),
        'retweekHigh': float(((price/weekHigh) - 1)*100),
        'retweekLow': float(((price/weekLow) - 1)*100),
        'retmonthHigh': float(((price/monthHigh) - 1)*100),
        'retmonthLow': float(((price/monthLow) - 1)*100),
        'retyearHigh': float(((price/yearHigh) - 1)*100),
        'retyearLow': float(((price/yearLow) - 1)*100)
    }
    return data

def calculate_returns(price, df):
    periods = {
        'ret5': 5,
        'ret15': 10,
        'ret30': 20,
        'ret90': 60
    }
    result = {}
    for key, period in periods.items():
        if len(df) > period:
            result[key] = round(
                ((price / df['Open'].iat[-period]) - 1) * 100,
                2
            )
        else:
            result[key] = None
    return result

def safe_rs(stock_ret, index_ret):
    if stock_ret is None or index_ret is None:
        return None
    return float(round(stock_ret - index_ret, 2))

def rs_fun(result_ret, index_data):
    price = index_data['Close'].iat[-1]
    ret = calculate_returns(price, index_data)
    # ret1 = float(((price/index_data['Close'].iat[-2])-1)*100)
    # ret5 = float(((price/index_data['Open'].iat[-5])-1)*100)
    # ret15 = float(((price/index_data['Open'].iat[-10]) - 1)*100)
    # ret30 = float(((price/index_data['Open'].iat[-20]) - 1)*100)
    # ret90 = float(((price/index_data['Open'].iat[-60]) - 1)*100)

    data = {
        'rs5': safe_rs(result_ret.get('ret5'),ret['ret5']),
        'rs15': safe_rs(result_ret.get('ret15'),ret['ret15']),
        'rs30': safe_rs(result_ret.get('ret30'),ret['ret30']),
        'rs90': safe_rs(result_ret.get('ret90'),ret['ret90']),
    }

    return data
import sqlite3
import pandas as pd


def delivery_data_analysis(df, symbol):

    conn = sqlite3.connect(
        "delivery_history.db"
    )

    query = """
    SELECT
        date,
        symbol,
        close_price,
        change_per,
        ttl_trd_qnty,
        turnover_lacs,
        no_of_trades,
        deliv_qty,
        deliv_per

    FROM delivery_history
    WHERE symbol = ?
    ORDER BY date DESC
    LIMIT 5
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(symbol,)
    )
    conn.close()

    # RETURN EMPTY IF NO DATA
    if df.empty:
        return {
            "Symbol": symbol
        }

    # =========================================
    # CALCULATIONS
    # =========================================

    # Trade Ratio = Volume / Trades
    df["trade_ratio"] = (df["ttl_trd_qnty"]/df["no_of_trades"].replace(0,pd.NA))

    # Volume Ratio

    avg_volume = df["ttl_trd_qnty"].mean()

    df["volume_ratio"] = (df["ttl_trd_qnty"]/avg_volume)

    # Delivery Score

    df["delivery_score"] = (df["deliv_per"] * 0.5 + df["volume_ratio"] * 30 + (df["trade_ratio"]/df["trade_ratio"].mean()) * 20)

    # =========================================
    # OUTPUT ROW
    # =========================================

    row = {
        "Symbol": symbol
    }

    for _, r in df.iterrows():

        date_str = pd.to_datetime(
            r["date"]
        ).strftime("%d/%m")

        row[f"Delivery {date_str}"] = round(
            r["deliv_per"],
            2
        )

        row[f"TradeRatio {date_str}"] = round(
            r["trade_ratio"],
            2
        )

        row[f"VolumeRatio {date_str}"] = round(
            r["volume_ratio"],
            2
        )

        row[f"DeliveryScore {date_str}"] = round(
            r["delivery_score"],
            2
        )

    return row