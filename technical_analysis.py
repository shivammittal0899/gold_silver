from ta.momentum import RSIIndicator
from ta.trend import IchimokuIndicator
from ta import volatility
from ta.volume import VolumeWeightedAveragePrice
import pandas as pd

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

    df['ichimoku_conversion_line'] = ichi.ichimoku_conversion_line()
    df['ichimoku_base_line'] = ichi.ichimoku_base_line()
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
    # print(df.tail())
    return df



def find_swings(df, window=5):
    # df['swing_high'] = df['High'][
    #     df['High'] == df['High'].rolling(window, center=True).max()
    # ]

    # df['swing_low'] = df['Low'][
    #     df['Low'] == df['Low'].rolling(window, center=True).min()
    # ]
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

    swing_highs = df[df['swing_high'].notna()]['High']
    swing_lows  = df[df['swing_low'].notna()]['Low']

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "UNKNOWN"

    last_high = swing_highs.iloc[-1]
    prev_high = swing_highs.iloc[-2]

    last_low = swing_lows.iloc[-1]
    prev_low = swing_lows.iloc[-2]

    if last_high > prev_high and last_low > prev_low:
        trend = "UPTREND"

    elif last_high < prev_high and last_low < prev_low:
        trend = "DOWNTREND"

    else:
        trend = "SIDEWAYS"
    return trend, last_high, last_low

def sr_breakout(df, lookback=20):
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
        "breakout": float(breakout_type)
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

    expansion = atr > df['atr'].rolling(20).mean().iloc[-1]

    return {
        "atr": atr,
        "regime": regime,
        "expansion": expansion
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
    avg_vol = df['vol_20'].rolling(50).mean().iloc[-1]

    return {
        "volatility": current_vol,
        "avg_vol": avg_vol
    }

def data_analysis(df):

    df = indicator_values(df)
    df = df[-50:]
    # log()
    # print(df.tail())
    timeframe = "5m"
    price = df['Close'].iat[-1]
    ret6 = round((((price / df['Open'].iat[-6]) - 1)*100),2)
    ret12 = round((((price / df['Open'].iat[-12]) - 1)*100),2)
    trend, last_high, last_low = highlow_trend(df)
    highlow = highlow_data(df)
    srb = sr_breakout(df)
    volatility = volatility_analysis(df)
    volatility_per = volatility_per_analysis(df, timeframe)
    data = {
        "price": float(price),
        "ret6": float(ret6),
        "ret12": float(ret12),
        "trend": trend,
        "l_high": float(last_high),
        "l_low": float(last_low),
        "highlow": highlow,
        # "support": float(srb["support"]),
        # "resistance": float(resistance),
        # "breakout": float(breakout),
        # "atr_val": float(atr_val),
        # "volatility_regime": volatility_regime,
        # "volatility_exp": volatility_exp,
        # "volatility_per": volatility_per,
        # "avg_volatility": avg_volatility,
        "vwap": "Above",
        "rsi": 60,
        "adx": 25,
        "volume": "High",
        "signal": "BUY"
    }
    data = data | srb
    data = data | volatility
    data = data | volatility_per

    return data


# data_analysis(df)