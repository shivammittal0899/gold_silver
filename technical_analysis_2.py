from ta.momentum import RSIIndicator
from ta.trend import IchimokuIndicator, EMAIndicator
from ta import volatility
from ta.volume import VolumeWeightedAveragePrice
import numpy as np
import pandas as pd
import json
import sqlite3
from ta.trend import ADXIndicator
import pandas as pd
from ta.volatility import AverageTrueRange


def calculate_choppiness_index(df, period=14):
    """
    Calculate Choppiness Index (CI)
    
    Parameters:
    df : DataFrame with High, Low, Close columns
    period : lookback period
    
    Returns:
    DataFrame with choppiness index column
    """

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=7
    ).average_true_range().round(2)
    df['atr'] = atr
    atr_sum = atr.rolling(period).sum()

    highest_high = df["High"].rolling(period).max()
    lowest_low = df["Low"].rolling(period).min()

    choppiness = (100 *
        np.log10(atr_sum / (highest_high - lowest_low))/np.log10(period)
    )
    df["choppiness"] = choppiness

    return df

def find_swings(df, window=3):
    df["rolling_max"] = df["High"].rolling(window).max()
    df["rolling_min"] = df["Low"].rolling(window).min()

    df["swing_high"] = df["High"].where(
        df["High"].round(2) == df["rolling_max"].round(2)
    )
    df["swing_low"] = df["Low"].where(
        df["Low"].round(2)  == df["rolling_min"].round(2) 
    )

    return df

def highlow_dataframe(df, window=5, lookback=5):
    df = find_swings(df).copy()

    swings = []
    structures = []
    for i in range(len(df)):
        if not pd.isna(df["swing_high"].iloc[i]):
            swings.append(("HIGH", df["High"].iloc[i]))
        elif not pd.isna(df["swing_low"].iloc[i]):
            swings.append(("LOW", df["Low"].iloc[i]))
        structure = []
        recent_swings = swings[-lookback:]
        for j in range(1, len(recent_swings)):
            prev_type, prev_price = recent_swings[j-1]
            curr_type, curr_price = recent_swings[j]
            if (
                curr_type == "HIGH"
                and prev_type == "HIGH"
            ):
                if curr_price > prev_price:
                    structure.append("HH")
                else:
                    structure.append("LH")

            elif (
                curr_type == "LOW"
                and prev_type == "LOW"
            ):
                if curr_price > prev_price:
                    structure.append("HL")
                else:
                    structure.append("LL")

        structures.append(
            structure.copy()
        )

    df["highlow_structure"] = structures

    return df

def calculate_structure_score(structure):
    if len(structure) < 3:
        return 0

    recent = structure[-4:]

    bullish = recent.count("HH") + recent.count("HL")
    bearish = recent.count("LH") + recent.count("LL")

    if bullish >= 3:
        return 3
    elif bullish >= 2:
        return 2
    elif bearish >= 3:
        return -3
    elif bearish >= 2:
        return -2
    else:
        return 0
def highlow_trend_dataframe(df, window=10, lookback=4):

    df = find_swings(df).copy()
    df["structure_score"] = (df["highlow_structure"].apply(calculate_structure_score))
    high_lists = []
    last_lows = []
    
    swing_highs = []
    swing_lows = []

    trends = []
    strengths = []
    breakout_strengths = []
    scores = []

    for i in range(len(df)):
        # Collect swings till current candle
        if not pd.isna(df["swing_high"].iloc[i]):
            swing_highs.append(df["High"].iloc[i])

        if not pd.isna(df["swing_low"].iloc[i]):
            swing_lows.append(df["Low"].iloc[i])

        recent_highs = swing_highs[-lookback:]
        recent_lows = swing_lows[-lookback:]

        # Not enough data
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            trends.append("UNKNOWN")
            high_lists.append([])
            last_lows.append(None)
            strengths.append(0)
            breakout_strengths.append(0)
            scores.append(0)
            continue

        hh = sum(
            recent_highs[j] > recent_highs[j-1]*1.1
            for j in range(1, len(recent_highs))
        )

        lh = sum(
            recent_highs[j] < recent_highs[j-1]
            for j in range(1, len(recent_highs))
        )

        hl = sum(
            recent_lows[j] > recent_lows[j-1]
            for j in range(1, len(recent_lows))
        )

        ll = sum(
            recent_lows[j] < recent_lows[j-1]*0.9
            for j in range(1, len(recent_lows))
        )

        up_score = hh + hl
        down_score = lh + ll
        

        high_move = abs(recent_highs[-1] - recent_highs[-2])
        low_move = abs(recent_lows[-1] - recent_lows[-2])
        atr = df["atr"].iloc[i]
        
        if (
            high_move < atr * 0.8
            and low_move < atr * 0.8
        ):
            trend = "SIDEWAYS"

        elif up_score >= (lookback - 1) * 1.5:
            trend = "STRONG UPTREND"

        elif down_score >= (lookback - 1) * 1.5:
            trend = "STRONG DOWNTREND"

        elif up_score > down_score:
            trend = "UPTREND"

        elif down_score > up_score:
            trend = "DOWNTREND"

        else:
            trend = "SIDEWAYS"


        # trend_strength = ((high_move + low_move) / (2 * df["ATR"].iloc[i]))
        if atr == 0:
            trend_strength = 0
        else:
            trend_strength = (
                high_move + low_move
            ) / (2 * atr)
        
        # =============================
        # Final Trend Score
        # =============================

        score = 0
        if trend == "STRONG UPTREND":
            if trend_strength > 2:
                score = 3
            else:
                score = 2

        elif trend == "UPTREND":
            score = 1
            if trend_strength > 1.5:
                score += 1

        elif trend == "SIDEWAYS":
            score = 0
        elif trend == "DOWNTREND":
            score = -1
            if trend_strength > 1.5:
                score -= 1
        elif trend == "STRONG DOWNTREND":
            if trend_strength > 2:
                score = -3
            else:
                score = -2
        # Add breakout bonus
        # score += breakout
        # Keep score bounded
        score = max(-3, min(3, score))
        trends.append(trend)
        strengths.append(round(trend_strength, 2))
        # breakout_strengths.append(breakout)
        scores.append(score)
        high_lists.append([
            float(recent_highs[-1]),
            float(recent_highs[-2])
        ])
        last_lows.append(
            float(recent_lows[-1])
        )
    
    df["trend"] = trends
    df["last_highs"] = high_lists
    df["last_low"] = last_lows
    df["trend_strength"] = strengths
    # df["breakout_strength"] = breakout_strengths
    df["trend_score"] = scores
    df["final_trend_score"] = (
        df["trend_score"] +
        (df["structure_score"])
    )
    df["final_trend_score"] = (
        df["final_trend_score"]
        .clip(-3, 3)
    )

    return df

def ichimoku_analysis_dataframe(df):

    df = df.copy()

    cloud_max = df[['senkou_a', 'senkou_b']].max(axis=1)
    cloud_min = df[['senkou_a', 'senkou_b']].min(axis=1)
    cloud_gap = abs(cloud_max - cloud_min)
    df["cloud_gap_pct"] = (
        abs(df["senkou_a"] - df["senkou_b"])
        / df["Close"]
    ) * 100

    cloud_maxf = df[['senkou_af', 'senkou_bf']].max(axis=1)
    cloud_minf = df[['senkou_af', 'senkou_bf']].min(axis=1)

    df["cloud_color"] = np.where(
        df["senkou_a"] >= df["senkou_b"],
        "green",
        "red"
    )
    df['price_tenkan_diff'] = (df['Close']/df['tenkan'] - 1)*100

    # Tenkan Kijun Trend
    conditions = [
        (
            (df["tenkan"] > df["kijun"]) &
            (df["tenkan"] > cloud_max) &
            (df["Close"] > cloud_max)
        ),

        (
            (df["tenkan"] >= df["kijun"]) &
            (df["Close"] > cloud_min)
        ),

        (
            (df["tenkan"] < df["kijun"]) &
            (df["tenkan"] < cloud_min) &
            (df["Close"] < cloud_min)
        ),

        (
            (df["tenkan"] <= df["kijun"]) &
            (df["Close"] < cloud_max)
        )
    ]

    choices = [
        "Strong Uptrend",
        "Uptrend",
        "Strong Downtrend",
        "Downtrend"
    ]

    df["tenkan_kijun"] = np.select(
        conditions,
        choices,
        default="Neutral"
    )


    # Price vs Tenkan
    conditions = [
        (
            (df["tenkan_kijun"] == "Strong Uptrend") &
            (df["Close"] > df["tenkan"])
        ),

        (
            (df["tenkan_kijun"] == "Uptrend") &
            (df["Close"] > df["tenkan"])
        ),

        (
            (df["Close"] > df["tenkan"]) &
            (df["Close"] > df["kijun"]) &
            (df['tenkan'] > df['kijun'])
        ),

        (
            (df["tenkan_kijun"] == "Strong Downtrend") &
            (df["Close"] < df["tenkan"])
        ),

        (
            (df["tenkan_kijun"] == "Downtrend") &
            (df["Close"] < df["tenkan"])
        ),

        (
            (df["Close"] < df["tenkan"]) &
            (df["Close"] < df["kijun"]) &
            (df['tenkan'] < df['kijun'])
        )
    ]

    choices = [
        "VStrong Uptrend",
        "Strong Uptrend",
        "Uptrend",
        "VStrong Downtrend",
        "Strong Downtrend",
        "Downtrend"
    ]

    df["price_tenkan"] = np.select(
        conditions,
        choices,
        default="Sideways"
    )


    # Cloud Trend
    conditions = [

        # Strong Uptrend
        (
            (df["tenkan_kijun"] == "Uptrend") &
            (df["cloud_color"] == "green")
        ),

        # Uptrend
        (
            (df["Close"] > cloud_max) &
            (
                (df["cloud_color"] == "green") |
                (
                    (df["cloud_color"] == "red") &
                    (df["cloud_gap_pct"] < 5)
                )
            )
        ),

        # Strong Downtrend
        (
            (df["tenkan_kijun"] == "Downtrend") &
            (df["cloud_color"] == "red")
        ),

        # Downtrend
        (
            (df["Close"] < cloud_min) &
            (
                (df["cloud_color"] == "red") |
                (
                    (df["cloud_color"] == "green") &
                    (df["cloud_gap_pct"] < 5)
                )
            )
        )

    ]

    choices = [
        "Strong Uptrend",
        "Uptrend",
        "Strong Downtrend",
        "Downtrend"
    ]

    df["cloud_trend"] = np.select(
        conditions,
        choices,
        default="Sideways"
    )

    df["tenkan_ret"] = (
        (df["tenkan"] / df["tenkan"].shift(3) - 1) * 100
    ).round(2)

    df["kijun_ret"] = (
        (df["kijun"] / df["kijun"].shift(3) - 1) * 100
    ).round(2)

    df["cloud_thickness"] = (
        abs(df["senkou_a"] - df["senkou_b"])
        / df["Close"] * 100
    ).round(2)

    bull_strong = (
        (df["tenkan_kijun"] == "Strong Uptrend")
        &
        (df["tenkan_ret"] > -1)
        &
        (df["kijun_ret"] > -1)
    )

    bull_normal = (
        (df["tenkan_kijun"] == "Uptrend")
        &
        (df["tenkan_ret"] > -1)
    )

    bear_strong = (
        (df["tenkan_kijun"] == "Strong Downtrend")
        &
        (df["tenkan_ret"] < 1)
        &
        (df["kijun_ret"] < 1)
    )

    bear_normal = (
        (df["tenkan_kijun"] == "Downtrend")
        &
        (df["tenkan_ret"] < 1)
    )

    df["tenkan_score"] = np.select(
        [
            bull_strong,
            bull_normal,
            bear_strong,
            bear_normal
        ],
        [
            3,
            2,
            -3,
            -2
        ],
        default=0
    )

    conditions = [

        (
            (df["price_tenkan"] == "VStrong Uptrend")
            &
            (df["price_tenkan_diff"] < 10)
        ),

        (
            (df["price_tenkan"] == "Strong Uptrend")
            &
            (df["price_tenkan_diff"] < 5)
        ),

        (
            (df["price_tenkan"] == "Uptrend")
        ),

        (
            (df["price_tenkan"] == "Downtrend")
        ),

        (
            (df["price_tenkan"] == "Strong Downtrend")
            &
            (df["price_tenkan_diff"] > -5)
        ),

        (
            (df["price_tenkan"] == "VStrong Downtrend")
            &
            (df["price_tenkan_diff"] > -10)
        )

    ]


    choices = [
        3,
        2,
        1,
        -1,
        -2,
        -3
    ]


    df["price_score"] = np.select(
        conditions,
        choices,
        default=0
    )

    cloud_map = {
        "Strong Uptrend": 2,
        "Uptrend": 1,
        "Sideways": 0,
        "Downtrend": -1,
        "Strong Downtrend": -2
    }

    df["cloud_score"] = (
        df["cloud_trend"]
        .map(cloud_map)
        .fillna(0)
    )
    future_cloud_max = df[["senkou_af", "senkou_bf"]].max(axis=1)
    future_cloud_min = df[["senkou_af", "senkou_bf"]].min(axis=1)

    df["future_cloud_gap_pct"] = (
        abs(df["senkou_af"] - df["senkou_bf"])
        / df["Close"]
    ) * 100

    future_cloud_green = df["senkou_af"] > df["senkou_bf"]
    df["future_cloud_score"] = np.select(

        [
            # Bullish
            (
                (df["Close"] > future_cloud_max) &
                (
                    future_cloud_green |
                    (
                        ~future_cloud_green &
                        (df["future_cloud_gap_pct"] < 5)
                    )
                )
            ),

            # Bearish
            (
                (df["Close"] < future_cloud_min) &
                (
                    (~future_cloud_green) |
                    (
                        future_cloud_green &
                        (df["future_cloud_gap_pct"] < 5)
                    )
                )
            )
        ],

        [
            1,
            -1
        ],

        default=0
    )

    df["ichimoku_score"] = (
        df["tenkan_score"] +
        df["price_score"] 
        # df["cloud_score"]
    ).clip(-5, 5)
    df["ichimoku_signal"] = np.select(
        [
            df["ichimoku_score"] >= 4,
            df["ichimoku_score"] >= 2,
            df["ichimoku_score"] > -2,
            df["ichimoku_score"] > -4,
            df["ichimoku_score"] <= -4,
        ],
        [
            "Strong Buy",
            "Buy",
            "Neutral",
            "Sell",
            "Strong Sell"
        ],
        default="Neutral"
    )

    return df

def get_adx_strength_signal_dataframe(df):

    df["adx_ret"] = (
        (df["ADX"] / df["ADX"].shift(3) - 1) * 100
    ).round(2)

    conditions = [
        # Very strong trend and strengthening
        (df["ADX"] > 40) &
        (df["DI_PLUS"] > df["DI_MINUS"]) &
        (df["adx_ret"] > 0),

        # Strong bullish trend but weakening
        (df["ADX"] > 40) &
        (df["DI_PLUS"] > df["DI_MINUS"]) &
        (df["adx_ret"] <= 0),

        # Very strong bearish trend
        (df["ADX"] > 40) &
        (df["DI_MINUS"] > df["DI_PLUS"]) &
        (df["adx_ret"] > 0),

        # Bearish but weakening
        (df["ADX"] > 40) &
        (df["DI_MINUS"] > df["DI_PLUS"]) &
        (df["adx_ret"] <= 0),

        # Good bullish trend
        (df["ADX"] > 17) &
        (df["DI_PLUS"] > df["DI_MINUS"]) &
        (df["adx_ret"] > 0),

        # Bullish but ADX falling
        (df["ADX"] > 17) &
        (df["DI_PLUS"] > df["DI_MINUS"]),

        # Good bearish trend
        (df["ADX"] > 17) &
        (df["DI_MINUS"] > df["DI_PLUS"]) &
        (df["adx_ret"] > 0),

        # Bearish but weakening
        (df["ADX"] > 17) &
        (df["DI_MINUS"] > df["DI_PLUS"]),

        (df["ADX"] < 17) &
        (df["adx_ret"] > 1),

        (df["ADX"] < 17) &
        (df["adx_ret"] < -1),
        # Sideways
        (df["ADX"] < 17),

        # # Trend developing
        # (df["ADX"] >= 15) &
        # (df["ADX"] <= 25)
    ]

    choices = [
        3,   # Explosive bullish trend
        2,   # Strong bullish but slowing

        -3,  # Explosive bearish trend
        -2,  # Strong bearish but slowing

        2,   # Bullish trend increasing
        1,   # Bullish but losing strength

        -2,  # Bearish trend increasing
        -1,  # Bearish but losing strength

        1,
        -1,
        
        0  # No trend
    ]

    df["adx_score"] = np.select(
        conditions,
        choices,
        default=0
    )

    return df

def signal_dataframe(df, ins_type="equity"):
    df = df.copy()
    # Return scores
    if ins_type == "options":
        df["ret6_score"] = np.select(
            [
                df["ret6"] > 10,
                df["ret6"] < 5
            ],
            [1, -1],
            default=2
        )
        df["ret12_score"] = np.select(
            [
                df["ret12"] > 2,
                df["ret12"] > -2
            ],
            [1, 0],
            default=-1
        )
    else:
        df["ret6_score"] = np.select(
            [
                df["ret6"] > 0.1,
                df["ret6"] > 0
            ],
            [1, 0],
            default=-1
        )
        df["ret12_score"] = np.select(
            [
                df["ret12"] > 0.1,
                df["ret12"] > -2
            ],
            [1, 0],
            default=-1
        )

    # Choppiness change %
    df["choppiness_ret"] = (
        (df["choppiness"] / df["choppiness"].shift(2) - 1) * 100
    ).round(2)

    condition1 = (df["choppiness_ret"] > 0) & (df["ret3"] > 0)
    condition2 = (df["choppiness_ret"] > 0) & (df["ret3"] <= 0)
    condition3 = (df["choppiness_ret"] <= 0) & (df["ret3"] > 1)
    condition4 = (df["choppiness_ret"] <= 0) & (df["ret3"] <= -1)

    df["chop_score"] = np.where(
        df["choppiness"] > 61,
        -2,
        np.where(
            df["choppiness"] > 50,
            np.select(
                [condition1, condition2, condition3, condition4],
                [0, 0, 2, -2]
            ),
            np.where(
                df["choppiness"] > 40,
                np.select(
                    [condition1, condition2, condition3, condition4],
                    [1, -1, 3, -3]
                ),
                np.where(
                    df["choppiness"] > 0,
                    np.select(
                        [condition1, condition2, condition3, condition4],
                        [1, -1, 2, -2]
                    ),
                    0
                )
            )
        )
    )

    df['rsi_ret'] = (
        (df["RSI"] / df["RSI"].shift(3) - 1) * 100
    ).round(2)


    df["rsi_score"] = np.select(
        [
            # Very strong momentum
            (df["RSI"] > 80) & (df["rsi_ret"] > 0),

            # Strong bullish
            (df["RSI"] > 65) & (df["rsi_ret"] > 0),

            # Bullish
            (df["RSI"] > 55) & (df["rsi_ret"] > 0),

            # RSI high but falling (exhaustion)
            (df["RSI"] > 70) & (df["rsi_ret"] < -0.5),

            # Neutral zone
            (df["RSI"] > 45) & (df["RSI"] <= 55),

            # Recovery from oversold
            (df["RSI"] > 35) & (df["rsi_ret"] > 0),

            # Bearish momentum
            (df["RSI"] > 35) & (df["rsi_ret"] < 0),

            # Oversold but improving
            (df["RSI"] <= 35) & (df["rsi_ret"] > 0),

            # Oversold and still falling
            (df["RSI"] <= 35) & (df["rsi_ret"] < 0),
        ],
        [
            3,   # Very strong uptrend
            2,   # Strong bullish
            1,   # Bullish
            -1,  # Losing momentum
            0,   # Neutral
            1,   # Recovery
            -1,  # Weakness
            0,   # Possible reversal
            -2   # Strong bearish
        ],
        default=0
    )
    
    

    df["vwap_diff"] = (
        (df["Close"] / df["VWAP"] - 1) * 100
    )
    df["vwap_ret"] = (
        (df["VWAP"] / df["VWAP"].shift(5) - 1) * 100
    )
    conditions = [
        (df["vwap_diff"] > 5),
        (df["vwap_diff"] > 2),
        (df["vwap_diff"] > 0.5) & (df["vwap_ret"] > 0),
        (df["vwap_diff"] > 0) & (df["vwap_ret"] > 0),
        (df["vwap_diff"] > 0.5),
        (df["vwap_diff"] > -0.5),
        (df["vwap_diff"] > -2),
    ]

    choices = [
        3,
        2,
        2,
        1,
        0,
        -1,
        -2
    ]

    df["vwap_score"] = np.select(
        conditions,
        choices,
        default=-3
    )
    df["signal_score"] = (
        df["final_trend_score"] +
        df["vwap_score"] +
        df["rsi_score"] +
        df["ichimoku_score"]
    )

    return df

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

    df = calculate_choppiness_index(df, period=14)

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

    # vwap = VolumeWeightedAveragePrice(
    #     high=df['High'],
    #     low=df['Low'],
    #     close=df['Close'],
    #     volume=df['Volume']
    # )

    # df['VWAP'] = vwap.volume_weighted_average_price()
    
    df["date_only"] = pd.to_datetime(df["date"]).dt.date

    df["tp"] = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    df["pv"] = df["tp"] * df["Volume"]

    df["VWAP"] = (
        df.groupby("date_only")["pv"].cumsum()
        /
        df.groupby("date_only")["Volume"].cumsum()
    )

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

def stock_data_analysis_2(df, ins_type = "equity"):

    df = indicator_values(df)
    df = df[-100:].copy()
    # log()
    # print(df.tail())
    df["ret1"] = ((df["Close"] / df["Close"].shift(1) - 1) * 100).round(2)
    df["ret3"] = ((df["Close"] / df["Close"].shift(2) - 1) * 100).round(2)
    df["ret6"] = ((df["Close"] / df["Close"].shift(6) - 1) * 100).round(2)
    df["ret12"] = ((df["Close"] / df["Close"].shift(12) - 1) * 100).round(2)
    df = highlow_dataframe(df).copy()
    df = highlow_trend_dataframe(df)
    df = highlow_dataframe(df)
    df = ichimoku_analysis_dataframe(df)
    df = get_adx_strength_signal_dataframe(df)
    df = signal_dataframe(df,"options")
    last = df.iloc[-1]
    latest_data = {
        "price": round(float(last['Close']),2),
        "ret1": round(float(last['ret1']),2),
        "ret3": round(float(last['ret3']),2),
        "ret6": round(float(last['ret6']),2),
        "ret12": round(float(last['ret12']),2),
        "trend": (last['trend']),
        "trend_score": round(float(last['trend_score']),2),
        "final_trend_score": round(float(last['final_trend_score']),2),
        "tenkan_score": round(float(last['tenkan_score']),2),
        "price_score": round(float(last['price_score']),2),
        "cloud_score": round(float(last['cloud_score']),2),
        "future_cloud_score": round(float(last['future_cloud_score']),2),
        "ichimoku_score": round(float(last['ichimoku_score']),2),
        "ichimoku_signal": last['ichimoku_signal'],
        "adx": round(float(last['ADX']),2),
        "adx_score": round(float(last['adx_score']),2),
        "chop_score": round(float(last['chop_score']),2),
        "rsi": round(float(last['RSI']),2),
        "rsi_score": round(float(last['rsi_score']),2),
        "vwap_score": round(float(last['vwap_score']),2),
        "signal_score": round(float(last['signal_score']),2),

    }
    positive_count = (
        (latest_data['cloud_score'] > 0) +
        (latest_data['future_cloud_score'] > 0) +
        (latest_data['adx_score'] > 0) +
        (latest_data['chop_score'] > 0) +
        (latest_data['final_trend_score'] > 0) +
        (latest_data['ichimoku_score'] > 0) +
        (latest_data['rsi_score'] > 0) +
        (latest_data['vwap_score'] > 0)
    )

    buy_condition = (
        (latest_data['signal_score'] >= 7) &
        (latest_data['cloud_score'] > 0) &
        (latest_data['future_cloud_score'] > 0) &
        (latest_data['final_trend_score'] > 0) &
        (latest_data['adx_score'] > 0) &
        (latest_data['chop_score'] > -1) &
        (latest_data['vwap_score'] > 0) &
        (positive_count >= 7)
    )
    negative_count = (
        (latest_data['cloud_score'] < 0) +
        (latest_data['future_cloud_score'] < 0) +
        (latest_data['adx_score'] < 0) +
        (latest_data['chop_score'] < 0) +
        (latest_data['final_trend_score'] < 0) +
        (latest_data['ichimoku_score'] < 0) +
        (latest_data['rsi_score'] < 0) +
        (latest_data['vwap_score'] < 0)
    )
    sell_condition = (
        (latest_data['signal_score'] >= -8) &
        (latest_data['cloud_score'] < 0) &
        (latest_data['future_cloud_score'] < 0) &
        (latest_data['final_trend_score'] < 0) &
        (latest_data['vwap_score'] < 0) &
        (positive_count >= 7)
    )

    buy_exit_condition = (
        (latest_data['signal_score'] <= 5) |
        (negative_count >= 3) |
        ((latest_data['signal_score'] == 6) & (negative_count >= 2))
    )

    sell_exit_condition = (
        (latest_data['signal_score'] >= -5) |
        (positive_count >= 3) |
        ((latest_data['signal_score'] == -6) & (positive_count >= 2))
    )

    # Final Signal
    if buy_condition:
        signal = "BUY"

    elif sell_condition:
        signal = "SELL"

    else:
        signal = "SIDEWAYS"
    

    if buy_exit_condition:
        signal_exit = "EXIT_BUY"
    elif sell_exit_condition:
        signal_exit = "EXIT_SELL"
    else:
        signal_exit = "HOLD"

    signal_dic = {'signal': signal,
                  'signal_exit': signal_exit}
    latest_data.update(signal_dic if isinstance(signal_dic, dict) else {})
    
    return latest_data, df