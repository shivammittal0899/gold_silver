from ta.momentum import RSIIndicator
from ta.trend import IchimokuIndicator
from ta import volatility
from ta.volume import VolumeWeightedAveragePrice
import pandas as pd
import json
def indicator_values(df):
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

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
        return "UNKNOWN"

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
        "volatility_exp": "expansion"
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
    tenkan = df['tenkan'].iloc[-2]
    kijun = df['kijun'].iloc[-2]
    senkou_a = df['senkou_a'].iloc[-2]
    senkou_b = df['senkou_b'].iloc[-2]
    senkou_af = df['senkou_af'].iloc[-2]
    senkou_bf = df['senkou_bf'].iloc[-2]
    close = df['Close'].iloc[-2]
    open = df['Open'].iloc[-2]
    cloud = "green" if senkou_a >= senkou_b else "red"
    cloud_max = max(senkou_a, senkou_b)
    cloud_min = min(senkou_a, senkou_b)
    cloud_maxf = max(senkou_af, senkou_bf)
    cloud_minf = min(senkou_af, senkou_bf)
    sideways = df['price_line'].iloc[-2]

    
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


def signal_fun(data, df):
    signal = "Neutral"
    ret6 = data['ret6']
    if ret6 > 200:
        ret6_ = 1
    elif ret6 > -200:
        ret6_ = 0
    else:
        ret6_ = -1
    ret12 = data['ret12']
    if ret12 > 300:
        ret12_ = 1
    elif ret12 > -300:
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
    
    signal_sum = ret6_ + ret12_ + trend_ + tenkan_kijun_ + price_tenkan_
    
    if signal_sum >= 5:
        signal = "Strong Buy"
    elif signal_sum >= 2:
        signal = "Buy"
    elif signal_sum <= -5:
        signal = "Strong Sell"
    elif signal_sum <= -2:
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
        # "l_high": float(last_high),
        "l_low": float(last_low),
        "highlow": highlow,
        "vwap": vwap_a,
        "rsi": float(round(df['RSI'].iloc[-1],2)),
        "adx": 25,
        "volume": "High"
    }
    data.update(volatility if isinstance(volatility, dict) else {})
    data.update(volatility_per if isinstance(volatility_per, dict) else {})
    data.update(ichimoku_d if isinstance(ichimoku_d, dict) else {})
    signal = signal_fun(data, df)
    data.update(signal if isinstance(signal, dict) else {})
    return data


# data_analysis(df)