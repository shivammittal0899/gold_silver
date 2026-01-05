import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
import time
# from strategy import log
# ---------------------------
# Ichimoku

# ---------------------------
def compute_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high_t = df['High'].rolling(window=tenkan).max()
    low_t  = df['Low'].rolling(window=tenkan).min()
    df['tenkan'] = (high_t + low_t) / 2

    high_k = df['High'].rolling(window=kijun).max()
    low_k  = df['Low'].rolling(window=kijun).min()
    df['kijun'] = (high_k + low_k) / 2

    df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(kijun)

    high_s = df['High'].rolling(window=senkou_b).max()
    low_s  = df['Low'].rolling(window=senkou_b).min()
    df['senkou_b'] = ((high_s + low_s) / 2).shift(kijun)

    df['chikou'] = df['Close'].shift(-kijun)
    return df


def ATR(df, n=14):
    tr = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# === Step 2: Compute ADX manually ===
def compute_adx(df, n=14):
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    df['TR_smooth'] = df['TR'].rolling(n).sum()
    df['+DI'] = 100 * (df['+DM'].rolling(n).sum() / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM'].rolling(n).sum() / df['TR_smooth'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(n).mean()
    return df



# === Step 4: Classify market regimes ===

def classify_trend_ichimoku(row):
    """
    Pure Ichimoku-based market regime classifier.
    Works inside df.apply(...).
    """

    price = row['Close']
    tenkan = row['tenkan']
    kijun = row['kijun']
    senkou_a = row['senkou_a']
    senkou_b = row['senkou_b']
    adx = row['ADX']            # Added
    rsi = row['RSI']            # Added

    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    # Basic zones
    above_cloud = price > cloud_top
    below_cloud = price < cloud_bottom
    inside_cloud = cloud_bottom <= price <= cloud_top

    # Trend direction via TK cross
    tk_bull = tenkan > kijun
    tk_bear = tenkan < kijun

    # Future cloud direction
    cloud_bull = senkou_a > senkou_b
    cloud_bear = senkou_a < senkou_b

    # Cloud thinness (possible regime shift)
    cloud_thin = abs(senkou_a - senkou_b) < (0.0015 * price)

    # ================================
    # STRONG TREND (with ADX strength)
    # ================================
    price_low = (row['Low']+00) > kijun
    price_high = (row['High']-00) < kijun
    price_up =row['Price_above_tk']
    price_down =row['Price_below_tk']

    # ================================
    # STRONG TREND (with ADX strength)
    # ================================
    if above_cloud and tk_bull and cloud_bull and adx > 25 and rsi < 70 and price_low and price_up:
        return "Strong Trend"

    if below_cloud and tk_bear and cloud_bear and adx > 25 and rsi > 30 and price_high and price_down:
        return "Strong Trend"
    
    # ===============================================
    # WEAK TREND / EARLY BREAKOUT (ADX 15–25 zone)
    # ===============================================
    if above_cloud and tk_bull and 10 < adx <= 25:
        return "Weak Trend / Possible Breakout"

    if below_cloud and tk_bear and 10 < adx <= 25:
        return "Weak Trend / Possible Breakout"

    # ==========================================
    # RANGE (inside cloud or ADX < 15)
    # ==========================================
    if inside_cloud:
        return "Range / Mean Reversion"

    # ==========================================
    # VOLATILITY BREAKOUT / REGIME SHIFT
    # ==========================================
    if cloud_thin:
        return "Volatility Breakout / Regime Shift"

    if (above_cloud and tk_bear) or (below_cloud and tk_bull):
        return "Volatility Breakout / Regime Shift"

    # ==========================================
    # DEFAULT
    # ==========================================
    return "Choppy / Noisy"



# ---------------------------
# Generate signals
# ---------------------------
def generate_signals(df, p):
    df['vol_ma'] = df['Volume'].rolling(window=p['vol_ma_window']).mean()
    df['oi_ma']  = df['OI'].rolling(window=p['oi_ma_window']).mean()
    df['rsi_ma']  = df['RSI'].rolling(window=p['vol_ma_window']).mean()
    
    df['price_above_cloud'] = df['Close'] > df[['senkou_a','senkou_b']].max(axis=1)
    df['price_below_cloud'] = df['Close'] < df[['senkou_a','senkou_b']].min(axis=1)
    df['price_in_cloud'] = ((df['Close'] < df[['senkou_a','senkou_b']].max(axis=1)) & (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1))) 

    df['tenkan_above_kijun'] = df['tenkan'] > df['kijun']
    df['tenkan_below_kijun'] = df['tenkan'] < df['kijun']
    
    df['price_below_tenkan'] = df['Close'] < df['tenkan']
    df['price_above_tenkan'] = df['Close'] > df['tenkan']
    
    df['price_below_kijun'] = df['Close'] < df['kijun']
    df['price_above_kijun'] = df['Close'] > df['kijun']

    df['price_above_tk'] = (df['Close'] > df['tenkan']) & (df['Close'] > df['kijun'])
    df['price_below_tk'] = (df['Close'] < df['tenkan']) & (df['Close'] < df['kijun'])

    df['green_cloud'] = ((df['senkou_a'] > df['senkou_b'].shift(1)).rolling(3).sum() >= 3)
    df['red_cloud'] = ((df['senkou_a'] <= df['senkou_b'].shift(1)).rolling(3).sum() >= 3)
    
    df['line_gap'] = abs(df['tenkan'] - df['kijun'])
    df['cloud_gap'] = abs(df['senkou_a'] - df['senkou_b'])
    df['candel_size'] = abs(df['Close'] - df['Open'])
    df['price_tenkan_gap'] = abs(df['Close'] - df['tenkan'])
    df['tenkan_kijun_gap'] = abs(df['tenkan'] - df['kijun'])
    df['price_cloud_gap'] = abs(df['Close'] - df[['senkou_a','senkou_b']].max(axis=1))
    df['DI_gap'] = abs(df['+DI'] - df['-DI'])
    df['+DI_up'] = df['+DI'] >= df['-DI']
    df['-DI_up'] = df['+DI'] < df['-DI']
    
    # DI Cross detection
    df['DI_cross'] = np.where(
        (df['+DI_up']) & (df['+DI_up'].shift(1) == False),
        "Cross +",
        np.where(
            (df['-DI_up']) & (df['-DI_up'].shift(1) == False),
            "Cross -",
            None
        )
    )
    
    df['tk_cross_up'] = df['tenkan_above_kijun'] & (~df['tenkan_above_kijun']).shift(1).fillna(False)
    df['tk_cross_down'] = df['tenkan_below_kijun'] & (~df['tenkan_below_kijun']).shift(1).fillna(False)

    df['+DI_move_up'] = (df['+DI'] > df['+DI'].shift(1)).rolling(3).sum() >= 2
    df['+DI_move_down'] = (df['+DI'] < df['+DI'].shift(1)).rolling(3).sum() >= 2
    df['-DI_move_up'] = (df['-DI'] > df['-DI'].shift(1)).rolling(3).sum() >= 2
    df['-DI_move_down'] = (df['-DI'] < df['-DI'].shift(1)).rolling(3).sum() >= 2
    df['price_move_up'] = (df['Close'] > df['Close'].shift(1)).rolling(3).sum() >= 2
    df['price_move_down'] = (df['Close'] < df['Close'].shift(1)).rolling(3).sum() >= 2
    df['price_green_candle'] = (df['Close'] > df['Open'].shift(1)).rolling(2).sum() >= 2
    df['price_red_candle'] = (df['Close'] < df['Open'].shift(1)).rolling(2).sum() >= 2

    # --- Sideways detection ---
    lookback = 10
    range_high = df['High'].rolling(lookback).max()
    range_low = df['Low'].rolling(lookback).min()

    # Range width as % of mid-range
    range_width = (range_high - range_low) / ((range_high + range_low) / 2)

    # Cloud midline and proximity
    cloud_mid = (df['senkou_a'] + df['senkou_b']) / 2
    cloud_near = (abs(df['Close'] - cloud_mid) / cloud_mid) < 0.001   # within 1%
    cloud_top = df[['senkou_a', 'senkou_b']].max(axis=1)
    cloud_bottom = df[['senkou_a', 'senkou_b']].min(axis=1)
    # Sideways flag: narrow range + near cloud
    df['sideways_zone'] = ((range_width < 0.002) & (cloud_near)).astype(int)

    # Conditions for breakout confirmation
    price_above_tk = (df['Close'] > df['tenkan']) & (df['Close'] > df['kijun'])
    price_below_tk = (df['Close'] < df['tenkan']) & (df['Close'] < df['kijun'])

    # Breakout confirmation: price breaks last range high and confirms next bar
    break_above_range = (df['Close'] > range_high.shift(0))
    
    break_below_range = (df['Close'] < range_low.shift(0))
    
    exit_condition = (df['Close'] < df['Open']).rolling(3).sum() >= 3
    entry_condition = (df['Close'] > df['Open']).rolling(3).sum() >= 3

    df['rolling_min_10'] = df['Close'].rolling(window=10).min()
    df['rolling_max_10'] = df['Close'].rolling(window=10).max()
    df['rolling_min_5'] = df['Close'].rolling(window=5).min()
    df['rolling_max_5'] = df['Close'].rolling(window=5).max()
    
    df['open_min_5'] = df['Open'].rolling(window=5).min()
    df['open_max_5'] = df['Open'].rolling(window=5).max()
    
    df['rolling_min_120'] = df['Close'].rolling(window=120).min()
    df['rolling_max_120'] = df['Close'].rolling(window=120).max()
    df['rolling_min_60'] = df['Close'].rolling(window=60).min()
    df['rolling_max_60'] = df['Close'].rolling(window=60).max()
    df['range1'] = df['rolling_max_10'] - df['rolling_min_10']
    df['range2'] = df['rolling_max_5'] - df['rolling_min_5']
    
    df['Entry_long_above_all'] = (
        df['price_above_cloud'] &
        (df['tenkan_above_kijun']) &
        (df['range1'] > df['ATR']) & 
        ((df['Close'] > df['Open']).rolling(2).sum() >= 2) &
        ((df['price_above_tk'].rolling(2).sum()>=2) &
         ((df['price_tenkan_gap'] > df['ATR']).rolling(2).sum() >= 2) &
         df['price_move_up'] &
         (((df['Close'] > df['rolling_max_10'].shift(1)).rolling(2).sum() >= 2)) &
         ((df['senkou_a'] - df['senkou_b']) > df['ATR']*2) &
         (df['candel_size'] > 150) &
         ((df['candel_size'] > df['candel_size'].shift(1)).rolling(3).sum()>=3) ) &
        (df['OI'] > df['oi_ma']) 
        
        
        # (df['price_move_up'] |
        #  (((df['Close'] > df['rolling_max_10'].shift(1)).rolling(2).sum() >= 2)) ) &
        #  df['+DI_move_up'] &
        # # ((df['DI_gap'] > df['DI_gap'].shift(1)).rolling(3).sum() >=2) &
        # ((df['candel_size'] > df['candel_size'].shift(1)).rolling(2).sum()>1) &
        
    )

    df['entry_long_tenkan_cross'] = (
        # df['price_above_tenkan'] & 
        df['price_below_tenkan'].shift(1) &
        ((df['Close'] + 50) > df['tenkan']) &
        ((df['Close'] > df['Open']).rolling(3).sum() >=3) &
        df['price_move_up'] &
        (df['Close'] >= df['rolling_max_5']-df['ATR']*0.2) &
        # df['+DI_move_up'] &
        (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)).rolling(2).sum() >=2)) &
        (df['RSI'] > df['rsi_ma']) 
    )
    df['entry_long_tk_cross'] = (
        (df['price_below_cloud'] &
        ((df['price_above_tk'] & df['price_below_tk'].shift(2)))&
        (((df['Close'] + 30) > df['Open']).rolling(2).sum() >=2) &
        (df['candel_size'] > 100) &
        (df['line_gap'] < df['ATR'] * 1.5) &
        # df['price_move_up'] &
        (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)).rolling(3).sum() >=2)) ) 
    )
    df['entry_long_bt_tk'] = (
        df['price_below_cloud'] &
        # ((df['Close']) < df['kijun']) &
        # (df['Close'] > df['tenkan']) &
        (df['Close'].shift(1) < df[['tenkan', 'kijun']].max(axis=1)) &
        (df['Close'].shift(1) > df[['tenkan', 'kijun']].min(axis=1)) &
        # df['price_above_tk'] &
        (df['tenkan_kijun_gap'] > df['ATR']*1.5) &
        df['price_move_up'] &
        (((df['Close']-20) > df['Open']).rolling(2).sum() >=2) &
        (df['Close'] > df['rolling_min_10'] + df['ATR']*1) &
        (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)).rolling(2).sum() >=2)) &
        (df['line_gap'] < df['line_gap'].shift(3))
    )
    df['entry_long_below_tk'] = (
        df['price_above_cloud'] &
        df['price_below_tk'].shift(2) &
        ((df['Close'] > df['Open']).rolling(2).sum() >= 2) &
        
        (df['green_cloud'] &
        #  (df['tenkan_below_kijun']) &
         ((df['Close'] - 50) > df[['tenkan','kijun']].min(axis=1)) &
         (df['line_gap'] > df['ATR']*1) &
         (df['Close'] > df['rolling_max_5'].shift(1) + df['ATR']*0) &
         (df['price_move_up']) &
        #  (df['+DI_move_up']) &
         (df['ADX'] > df['ADX'].shift(1)) 
        #  (df['OI'] > df['oi_ma'])
        ) | 
        ((df['green_cloud']) &
         (df['price_tenkan_gap'] < df['price_tenkan_gap'].shift(2)) &
         ((df['tenkan'] - df['senkou_a']) > df['ATR']*1) &
         (df['candel_size'] > 100) &
         (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_up'])) &
         (df['RSI'] > df['rsi_ma']*0.9) 
        ) |
        (((df['Close'] - 0) > df[['tenkan','kijun']].min(axis=1)) &
        #  ((df['Low'] - 50) < df[['tenkan','kijun']].min(axis=1)) &
         (df['tenkan_above_kijun']) &
        #  (df['green_cloud']) &
         (df['candel_size'] > 100) &
         (df['price_move_up']) &
         (df['Close'] >= df['rolling_max_5'].shift(1)) &
         (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_up'])) 
        #  (df['RSI'] > df['rsi_ma']*0.9) 
        ) |
        (
         ((df['Close']) > df[['tenkan','kijun']].min(axis=1)) &
          (df['Close'] > df['rolling_max_5'].shift(1)) &
          (((df['Open'] - df['Low']) > (df['High'] - df['Open'])*4).rolling(2).sum() >= 1) &
          (df['RSI'] > df['rsi_ma']*0.95) &
          (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_up'])) 
        )
    )
    df['entry_long_below_tk_cloud'] = (
        df['price_below_cloud'] &
        df['price_below_tk'].shift(2) & 
        (((df['Close']+50) > df['Open']).rolling(2).sum() >= 2) &
        (df['Close'] > df['rolling_min_10'] + df['ATR']) &
        
        (df['price_move_up'] &
         (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_up'])) &
         (df['price_tenkan_gap'] > df['ATR']*2)) |
        
        (((df['Close']) > df[['tenkan','kijun']].max(axis=1)) &
         (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_up'])) &
         (((df['Close']-50) > df['Open']).rolling(2).sum() >= 2) &
         (df['range1'] > df['ATR']*2) &
         (df['Close'] > df['rolling_min_10'] + df['ATR']) &
         (df['rolling_max_10'].shift(1) < df[['tenkan','kijun']].min(axis=1) + df['ATR']*1) 

        )
    )
    df['entry_long_cloud_enter'] = (
        df['price_below_cloud'].shift(2) &
        # (df['RSI'] > df['rsi_ma']) &
        # (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['+DI_move_up'])) &
        df['price_move_up'] &
        df['price_above_tk'] &
        
        (df['green_cloud'] &
         (df['Close'] > df['senkou_b']) &
         ((df['Close'] - 100) > df['Open']) &
         (df['Close'] > df['rolling_max_10'] - df['ATR']*1) &
         (df['line_gap'] > df['line_gap'].shift(2)) &
         (df['tenkan_above_kijun'])
        ) |
        
        (df['red_cloud'] &
         ((df['Close'] - 50) > df['senkou_a']) &
         (df['cloud_gap'] < df['cloud_gap'].shift(1)) &
         (((df['Close'] - 10) > df['Open']).rolling(3).sum() >= 3) &
         (df['Close'] >= df['rolling_max_10'] - df['ATR']*0.5) &
         (df['range1'] > df['ATR']*2) &
         (df['line_gap'] > df['line_gap'].shift(1)) &
         (df['ADX'] > df['ADX'].shift(2))
        ) |
        ((df['price_above_tk'].rolling(2).sum() == 1) &
         (((df['Close']+20) > df['Open']).rolling(3).sum()>=3) &
         (df['Close'] > df['rolling_max_10'] - df['ATR']*1) &
         (df['cloud_gap'] > df['ATR']*1) &
         (df['candel_size'] > 50) &
        #  df['green_cloud'] &
         (df['range1'] > df['ATR']*2) )
    )
    df['entry_long_in_cloud'] = (
        (df['range1'] > df['ATR']*1.5) &
        df['price_in_cloud'] &
        (df['ADX'] > df['ADX'].shift(2)) &
        (((df['DI_gap'] > df['DI_gap'].shift(1)) & df['+DI_up']) | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['-DI_up'])) &
        (df['Close'] > df['rolling_min_10'] + df['ATR']*1) &
        (df['price_above_tenkan']) &
        
        (((((df['cloud_gap'] > df['cloud_gap'].shift(1)) & df['green_cloud']) | ((df['cloud_gap'] < df['cloud_gap'].shift(1)) & df['red_cloud']))&
        (df['price_move_up'] & 
        (((df['Close']+20) > df['Open']).rolling(2).sum() == 2)) &
        
        (((df['Close'] + 0) > df['Open']).rolling(3).sum() >= 2)) )
    )
    df['entry_long_head_candle'] = (
                                    df['price_below_tk'].shift(2) &
                                    (((df['Close'] - 20) > df['Open']).rolling(2).sum() >= 2) &
                                    ((df['Close']) > df[['tenkan','kijun']].min(axis=1)) &
                                    (df['Close'] >= df['rolling_min_5'].shift(1) + df['ATR']*0.5) &
                                    (df['price_move_up'] | (df['Close'] > df['Open'])) &
                                    (((df['Open'] - df['Low']) > (df['High'] - df['Open'])*3).rolling(2).sum() >= 1) &
                                    (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_up'])) 
                                    
                                    )
   
    
    df['exit_long_price_tk_cross1'] = (
        df['price_below_tk'] & 
         df['price_above_tk'].shift(2) &
        # (((df['Close']-20) < df['tenkan']).rolling(2).sum() == 2) &
        (df['price_above_cloud'] &
         (df['Close'] <= df['rolling_min_5']) &
        (df['line_gap'] < df['ATR'] * 1.5) &
        ((df['line_gap']-20) < df['line_gap'].shift(1)) &
        df['+DI_move_down'] &
        (df['candel_size'] > 100) )
    )
    df['exit_long_price_tk_cross2'] = (

        df['price_below_tk'] & df['price_above_tk'].shift(1) &
        df['+DI_move_down'] &
        # df['price_move_down'] &
        (((df['Close']-20) < df['Close'].shift(1)).rolling(3).sum() >= 2) &
        (((df['Close']-20) < df['Open']).rolling(3).sum() >= 2) &
        df['price_red_candle'] &
        (df['candel_size'] > 100) &
        # (df['RSI'] < df['rsi_ma']) &
        
        (df['price_above_cloud'] &
         (df['range1'] > df['ATR']*1) &
        (df['line_gap'] < df['ATR'] * 1) &
        (df['candel_size'] > 100) ) |
        
        ((df['green_cloud']) &
         (((df['Close']-50) < df[['senkou_a','senkou_b']].max(axis=1))) &
         ((df['candel_size'] > 50).rolling(2).sum() >= 2) &
         (df['RSI'] < df['rsi_ma']) &
         (df['RSI'] < 40) &
         (df['tenkan_below_kijun'].rolling(3).sum() >= 3) &
         (df['line_gap'] > (df['line_gap'].shift(1))*1.02) &
         (df['OI'] < df['oi_ma']*0.9)) 
    )
    

    df['exit_long_price_tk_cross3'] = (
        (df['ADX'] < 30) &
        df['price_below_tk'] &
        (df['+DI_move_down']) &
        (df['Close'] < (df['rolling_min_10']+df['ATR']*0.5)) &
        (df['range1'] > df['ATR']*1.5) &
        (((df['Close'].shift(1)-20) > df[['kijun','tenkan']].min(axis=1)).rolling(2).sum() >= 1) &
        df['tenkan_below_kijun'] &
        ((df['Close'] + 50) < df['tenkan']) &
        ((df['RSI'] < df['rsi_ma']).rolling(4).sum() >= 2) &
        df['price_move_down'] & 
        df['price_red_candle'] &
        (((df['Close'] - 20) < df['Open']).rolling(2).sum()==2)
    )
    ## used in entry short
    df['exit_long_price_tenkan'] = (
        df['price_move_down'] &
        (((df['Close'] + 20) < df['tenkan']).rolling(2).sum() >= 1) &
        ((df['RSI'] < df['rsi_ma']).rolling(2).sum() >= 2) &
        (((df['Close'] + 20) < df['Open']).rolling(2).sum() >= 2) &
        (df['Close'] < df['rolling_max_5'] - df['ATR']*1) &
        (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['+DI_move_down'])) &
        
        ((df['tenkan_kijun_gap'] > df['ATR']*2) &
         (df['price_above_cloud']) &
         (df['Close'] < df['rolling_max_5'] - df['ATR']*2) &
         (df['price_above_tk'].shift(3)) &
         (df['price_cloud_gap'] > df['ATR']*3)) |
        
        ((((df['Close'] - 50) < df['tenkan']).rolling(2).sum() == 2) &
         (df['price_above_cloud']) &
         (df['price_above_tk'].shift(3)) &
         (df['line_gap'] < df['line_gap'].shift(1)) &
         (df['ADX'] < 30) &
         (df['range1'] > df['ATR']*1.5) &
         (df['price_cloud_gap'] > df['ATR']*2) &
         ((df['ADX'] < df['ADX'].shift(1)).rolling(2).sum() == 2))
        # ((df['price_below_tenkan'].rolling(2).sum()==2) &
        #  (df['Close'] < df['rolling_max_5'] - df['ATR']*1) &)
        
        # ((df['price_above_cloud']) &
        #  (df['Close'] < df['rolling_max_10'] - df['ATR']*1.5) &
        # #  (df['Close'] <= df['rolling_min_5'] + df['ATR']*0.5) &
        #  (df['range1'] > df['ATR']*2.5) &
        #  (df['tenkan_above_kijun']) &
        #  (df['price_cloud_gap'] > df['ATR']*4) &
        #  (((df['Close']+50) < df['Close'].shift(1)).rolling(2).sum() >= 2)&
        #  (df['ADX'] < df['ADX'].shift(1)) &
        #  (df['Volume'] < (df['vol_ma']*1.1)) &
        #  (df['ADX'] < 30)

        # ) |
        # (((df['Close'] < df[['senkou_a','senkou_b']].max(axis=1)) & (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1))) &
        #   (((df['Close'] + 20) < df['Open']).rolling(3).sum() >= 3) &
        #  (df['range1'].shift(1) > df['ATR']*1.5) &
        #  (df['Close'] <= df['rolling_min_10'] + df['ATR']*0) &
        #  (df['line_gap'] < df['line_gap'].shift(1))&
        #  (df['ADX'] < 30))
    )
    df['exit_long_price_tenkan1'] = (
        df['price_move_down'] &
        (((df['Close'] + 50) < df['tenkan']).rolling(2).sum() >= 1) &
        # ((df['RSI'] < df['rsi_ma']).rolling(2).sum() >= 2) &
        (((df['Close'] + 50) < df['Open']).rolling(2).sum() >= 2) &
        (df['Close'] < df['rolling_max_5'] - df['ATR']*1) &
        (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['+DI_move_down'])) &
        
        
        ((((df['Close'] - 50) < df['tenkan']).rolling(2).sum() == 2) &
         (df['price_above_cloud']) &
         (df['price_above_tk'].shift(3)) &
         (df['line_gap'] < df['line_gap'].shift(1)) &
         (df['ADX'] < 30) &
         (df['range1'] > df['ATR']*1.5) &
         (df['price_cloud_gap'] > df['ATR']*2) &
         ((df['ADX'] < df['ADX'].shift(1)).rolling(2).sum() == 2)) |
        
        (df['price_below_cloud'] &
         df['price_above_tenkan'].shift(2) &
         ((df['Close'] -50) < df['Open']) &
         (df['-DI_up'] | df['-DI_move_up']) &
         df['tenkan_below_kijun'] &
         (df['Close'] <= df['rolling_min_5'].shift(1) + df['ATR']*0.2) &
         (df['ADX'] < df['ADX'].shift(1)) &
         (df['RSI'] < 50) &
         (df['candel_size'] > 100) &
         (df['red_cloud'] | (df['green_cloud'] & (df['cloud_gap'] < df['cloud_gap'].shift(1)))))
    )
    df['exit_long_price_kijun'] = (
        
        ((df['Close'] < df['kijun']).rolling(2).sum() >= 2) &
        (df['price_move_down']) &
        (((df['Close']+50) < df['Open']).rolling(2).sum() >= 2) &
        # (df['RSI'] < df['rsi_ma']) &
        (df['+DI_move_down']) &
        (df['Close'] < df['rolling_max_10'] - df['ATR']*1) &
        
        (df['tenkan_above_kijun'] &
         (df['line_gap'] < df['line_gap'].shift(1) ) &
         (df['range1'] > df['ATR']*1.5) ) |
        
        (df['tenkan_below_kijun'] &
         df['price_above_cloud'] &
         (df['price_above_tk'].shift(3)) &
        #  (df['price_below_tk'].shift(0)) &
         ((df['line_gap'] < df['line_gap'].shift(1)) | (df['line_gap'] < df['ATR'])) &
         (df['Close'] <= df['rolling_min_10'] + df['ATR']*0.5) &
         (df['ADX'] < df['ADX'].shift(1)) &
         (df['range1'].shift(1) < df['ATR']) &
         (df['Close'] < df['kijun'])
        ) 
    )
    df['exit_long_price_kijjun1'] = (
        
        df['price_above_cloud'] &
        (df['price_below_tk'].rolling(2).sum() == 2) & 
        (((df['Close'] - 150) < df['Open']).rolling(2).sum() >= 1) &
        ((df['Close']) <= df['rolling_min_5'].shift(1)) &
        (df['RSI'] < df['rsi_ma']) &
        (df['ADX'] < df['ADX'].shift(1)) &
        (df['price_cloud_gap'] < df['price_cloud_gap'].shift(1)) &
        # (df['+DI_move_down']) &
        # (df['DI_gap'] < 10) &
        (df['ADX'] < 25) &
        (df['+DI'] < 25) &
        (df['cloud_gap'] < df['cloud_gap'].shift(1))
    )
    df['exit_long_cloud_entry'] = (
        (df['Close'] < df[['senkou_a','senkou_b']].max(axis=1) - df['ATR']*0.5) &
        (df['RSI'] < df['rsi_ma']) &
        (df['price_move_down']) &
        (df['-DI_up'] | ((df['DI_gap'] > df['DI_gap'].shift(1)) & df['+DI_move_down'])) &
        
        (df['price_above_cloud'].shift(2) &
         (df['price_below_tk'] | (df['Low'] < df[['tenkan','kijun']].max(axis=1))) &
        #  df['red_cloud'] &
         df['range1'] > df['ATR']) |

        ((df['Close'] > df[['senkou_a','senkou_b']].min(axis=1)) &
         (df['price_below_tk']) &
         (df['range1'] > df['ATR']*2) &
         (df['tenkan_below_kijun']) &
         (df['line_gap'] > df['line_gap'].shift(1)) &
         (((df['Close']+50) < df['Open']).rolling(2).sum() >= 2) &
         (df['Close'] <= df['rolling_min_10'].shift(1)) &
         (df['cloud_gap'] > df['ATR']*2)
        )
    )
    df['exit_long_cloud_exit'] = (
        (df['Close'] < df[['senkou_a','senkou_b']].max(axis=1)) &
        (df['Open'] > df[['senkou_a','senkou_b']].min(axis=1)) &
        ((df['Close'] - 10) < df[['senkou_a','senkou_b']].min(axis=1)) &
        (
         df['-DI_up'] &
        #  (df['ADX'] > df['ADX'].shift(1)) &
         (((df['Close'] + 50) < df['Open']).rolling(2).sum() == 2) &
         (df['candel_size'] > 100) &
         (df['line_gap'] < df['line_gap'].shift(1)) &
         (df['RSI'] < df['rsi_ma']*1) &
         df['tenkan_below_kijun']
        ) |
        (df['+DI_up'] &
         (df['DI_gap'] < df['DI_gap'].shift(1)) &
         ((df['Close'] + 50) < df[['senkou_a','senkou_b']].min(axis=1)) &
         (((df['Close'] - 10) <= df['Open']).rolling(3).sum() >=3) &
         ((df['Close'] + 200) < df['Open']) &
         (df['price_below_tk']) &
         df['red_cloud'] &
         (df['cloud_gap'] > df['ATR']*1) &
         ((df['High'] - 50) > df['tenkan']) &
         (df['RSI'] < df['rsi_ma']*1.1) &
         
         df['price_move_down'] 
         )

    )
    df['exit_long_below_tk'] = (
        df['tenkan_below_kijun'] &
        (df['price_below_cloud']) &
        df['price_below_tk'] &
        df['-DI_up'] &
        (df['DI_gap'] > df['DI_gap'].shift(1)) &
        (df['RSI'] < df['rsi_ma']) &
        (df['line_gap'] > df['line_gap'].shift(1)) &
        (((df['Close']) < df['Open']).rolling(3).sum() >= 2) &
        (df['price_move_down'])
    )
    df = df.copy()
    df['exit_short_cloud_enter'] = (
        (((df['Close'].shift(1)-10) < df[['senkou_a','senkou_b']].min(axis=1)).rolling(2).sum() >=1) &
        ((df['Close']+10) > df[['senkou_a','senkou_b']].min(axis=1)) &
        
        (df['green_cloud'] &
        ((df['Close'] + 10) > df['senkou_b']) &
        (df['price_move_up']) &
        (df['candel_size'] > 150)) |
        
        (df['green_cloud'] &
         (df['price_green_candle']) &
         (((df['Close'] - 20)> df['Open']).rolling(3).sum() >= 2) &
         (df['price_move_up']) &
         (df['candel_size'] > 100) &
         (df['RSI'] > df['rsi_ma']*0.95) &
         (df['Volume'] > df['vol_ma']) &
         (df['cloud_gap'] > df['ATR']) &
         ((df['Close'] + 50) > df['senkou_b']) &
         ((df['Close'] + 50) > df['tenkan'])) |
         
         (df['red_cloud'] &
         ((df['Close'] - 0) > df['senkou_a']) &
         (df['cloud_gap'] > df['ATR']) &
         (df['range1'] > df['ATR']*1) &
         (df['candel_size'] > 100) &
        #  (df['RSI'] > df['rsi_ma']) &
         (df['Close'] > df['rolling_max_10'].shift(1) - df['ATR']) &
        #  (df['price_move_up']) &
         entry_condition
        ) |
        ((df['cloud_gap'].shift(1) < df['ATR']*2) &
         (df['candel_size'] > 50) &
         (df['price_move_up']) &
         ((df['Close']) > df['tenkan']) &
         (df['RSI'] > df['rsi_ma']) &
         (df['+DI_move_up']) &
         (df['Volume'] > df['vol_ma']*0.8) &
         (df['Close'] >= (df['rolling_max_5'].shift(1) + df['ATR']*0.)) 
        ) 
    )
    df['exit_short_price_tk_cross1'] = (
        df['price_above_tk'] & df['price_below_tk'].shift(1) &
        df['-DI_move_down'] &
        (df['price_below_cloud'] &
        (df['line_gap'] < df['ATR'] * 1.5) &
        (df['candel_size'] > 100) ) 
    )
    df['exit_short_price_tk_cross2'] = (

        df['price_above_tk'] & df['price_below_tk'].shift(2) &
        # df['-DI_move_down'] &
        df['price_move_up'] &
        # df['price_green_candle'] &
        (df['candel_size'] > 100) &
        # (df['RSI'] > df['rsi_ma']) &
        
        ((df['price_below_cloud'] &
         (df['range1'] > df['ATR']*1) &
        (df['line_gap'] < df['ATR'] * 2) &
        (df['candel_size'] > 150) ) |
        
        (
         (((df['Close']+50) > df[['senkou_a','senkou_b']].min(axis=1))) &
         ((df['candel_size'] > 50).rolling(2).sum() >= 2) &
        #  (df['RSI'] > df['rsi_ma']) &
         (df['RSI'] > 40) &
         (df['tenkan_above_kijun'].rolling(3).sum() >= 2) &
         (df['line_gap'] > (df['line_gap'].shift(1))*1.0) &
         (df['OI'] < df['oi_ma']*1.1)) )
    )
    df['exit_short_tenkan_cross'] = (
        df['price_move_up'] &
        df['price_above_tenkan'] &
        (((df['Close'] + 30) > df['Open']).rolling(2).sum() >= 2) &
        (df['Close'] > df['rolling_min_5'] + df['ATR']*0.5) &
        df['-DI_move_down'] &
        
        ((((df['Close'] - 20) > df['tenkan']).rolling(2).sum() >= 2) &
         (df['price_below_cloud']) &
         df['price_below_tk'].shift(2)) |
        
        (df['tenkan_below_kijun'] &
         (df['tenkan_kijun_gap'] > df['ATR']*2) &
         ((df['Close'] + 100) > df['tenkan']) &
         (df['candel_size'] > 100) &
        #  df['price_below_cloud'] &
         (df['Close'] > df['rolling_min_5'] + df['ATR']*1) &
         df['price_below_tk'].shift(1)) |
        
        (df['tenkan_above_kijun'] &
         (df['+DI_up']) &
         (df['price_below_tenkan'].shift(2)) &
         ((df['Close'] + 100) > df['tenkan']) &
         ((df['candel_size'] > 100).rolling(2).sum() >= 2)&
        #  (((df['Close'] - 50) > df['Open']).rolling(2).sum() >= 2) &
        #  (df['ADX'] < 25) &
         ((df['RSI'] > df['RSI'].shift(1)).rolling(2).sum()>= 1)) 
        
    )
    df['exit_short_price_kijun'] = (
        
        (((df['Close'] + 20)> df['kijun']).rolling(3).sum() >= 2) &
        (df['price_move_up']) &
        (((df['Close']-50) > df['Open']).rolling(2).sum() >= 2) &
        (df['Close'] > df['rolling_min_10'] + df['ATR']*0.5) &
        
        (df['tenkan_below_kijun'] &
         (df['price_below_kijun'].shift(3)) &
         (df['range1'] > df['ATR']*1) ) 
    )
    df['exit_short_avoid_range'] = (
        (df['range1']< (df['ATR']*1.5)) &
        (df['Close'] < df['rolling_max_10'] + df['ATR']*0.5) &
        (df['Close'] > df['rolling_min_10'] - df['ATR']*0.5) &
        (
            # df['price_below_cloud'] &
         ((df['Close']+50) > df[['senkou_a','senkou_b']].min(axis=1)) &
         ((df['Close']-100) < df[['senkou_a','senkou_b']].max(axis=1)) &
         (df['price_above_tenkan'].rolling(2).sum() >=1) &
         (df['cloud_gap'] < df['ATR']*3) &
         (df['ADX'] < 30) &
         (df['ADX'] < df['ADX'].shift(1)) &
         (df['RSI'] > 30) &
         (df['line_gap'] < df['line_gap'].shift(2)) &
         (df['-DI_move_down']) &
         (df['Volume'] < df['vol_ma'])
         ) 
        # (df['Close'] < df[['senkou_a','senkou_b']].max(axis=1)) &
        # (df['+DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['-DI_move_down'])) &
        # (df['line_gap'] < df['ATR']*1)
        )
    df['exit_short_at_top'] = (
        df['price_above_cloud'] &
        (df['tenkan_below_kijun']) &
        df['price_above_tenkan'] &
        df['price_below_tenkan'].shift(2) &
        (df['range1'].shift(1) < df['ATR']*2) &
        (df['Close'] >= df['rolling_max_10'] - df['ATR']*0.5) &
        # ((df['red_cloud'] & (df['cloud_gap'] < df['cloud_gap'].shift(1))) | (df['green_cloud'] & (df['cloud_gap'] > df['cloud_gap'].shift(1)))) &
        (((df['Close'] + 50) > df['tenkan']) & (df['candel_size'] > 100) & 
         df['price_move_up'] & 
         df['price_green_candle']) 

    )
    df['exit_short_price_move_up'] = (
        (((df['Close'] - 0) > df['Open']).rolling(3).sum() == 3) &
        (((df['High'] - 100) > df['Low']).rolling(3).sum() == 2) &
        # ((df['Close']-20) > df['Close'].shift(3)) &
        (df['green_cloud']) &
        # (df['cloud_gap'] > df['cloud_gap'].shift(1)) &
        (((df['tenkan'] + 50)> df['kijun']).rolling(2).sum() == 2) &
        # df['tenkan_above_kijun'] &
        (df['price_cloud_gap'] > df['ATR']*3)
    )

    df['entry_short_price_top'] = (
        df['price_above_tk'] &
        df['tenkan_above_kijun'] &
        (df['price_cloud_gap'] > df['ATR']*3) &
        ((df['ADX'] < df['ADX'].shift(1)).rolling(2).sum() >= 2) &
        ((df['Close'] - 50) < df['tenkan']) &
        ((df['Close'] + 200) < df['Open']) 
    )
    df['entry_short_above_tk'] = (
        df['price_above_tk'].shift(2) &
        (((df['Close'] + 10) < df['Open']).rolling(2).sum() >= 2) &
        
        (df['red_cloud'] &
         df['price_below_cloud'] &
         (df['cloud_gap'] > df['cloud_gap'].shift(1)) &
         ((df['Close'] - 50) < df[['tenkan','kijun']].min(axis=1)) &
         (df['Close'] < df['rolling_min_5'].shift(1) - df['ATR']*0) &
         (df['price_move_down']) &
         (df['-DI_move_up']) 
        ) |
        (df['red_cloud'] &
         (df['cloud_gap'] > df['cloud_gap'].shift(1)) &
         df['price_below_cloud'] &
         (df['tenkan_below_kijun']) &
         ((df['Close'] + 50) < df[['tenkan','kijun']].max(axis=1)) &
         (df['Close'] < df['rolling_min_5'].shift(1) - df['ATR']*0.2) &
         (df['candel_size'] > 100) &
         (df['price_move_down']) ) |
        
        ((df['green_cloud']) &
         df['price_below_cloud'] &
         (df['candel_size'] > 100) &
         (df['price_tenkan_gap'] < df['price_tenkan_gap'].shift(1)) &
         ((((df['Close'] + 50) < df[['tenkan','kijun']].max(axis=1)) & (df['RSI'] > df['rsi_ma']) ) ) 
         
        ) |
        (
         ((df['Close'] + 20) < df[['tenkan','kijun']].min(axis=1)) &
         ((df['High'] - 0) > df[['tenkan','kijun']].min(axis=1)) &
         ((df['Open'] - 20) > df[['tenkan','kijun']].min(axis=1)) &
         (df['tenkan_below_kijun']) &
         (df['red_cloud']) &
         (df['candel_size'] > 100) &
         (df['price_move_down']) &
         (df['RSI'] < df['rsi_ma']) &
         (df['Close'] <= df['rolling_min_5'].shift(1) - df['ATR']*0.2) &
         (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['-DI_move_up'])) 
        ) |
        (
            # df['red_cloud'] &
         ((df['Close']) < df[['tenkan','kijun']].min(axis=1)) &
          (df['Close'] < df['rolling_min_5'].shift(1) -df['ATR']*0.2) &
          (df['price_move_down'] | (df['Close']<df['Open'])) &
          (((df['High'] - df['Open']) >(df['Open'] - df['Low'])*4).rolling(2).sum() >= 1) &
          (df['RSI'] < df['rsi_ma']) &
          (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['-DI_move_up'])) 
        )
    )
    df['entry_short_cloud_enter'] = (
        df['price_above_cloud'].shift(2) &
        df['price_move_down'] &
        df['price_below_tk'] &
        
        (df['green_cloud'] &
         (df['Close'] < df['senkou_a']) &
         ((df['Close'] + 50) < df['Open']) &
         (df['Close'] <= df['rolling_min_10'] + df['ATR']*0.5) &
         (df['RSI'] < 50) &
         (df['ADX'] > df['ADX'].shift(1)) &
         (df['-DI_up']) &
         (df['tenkan_below_kijun'])
        ) 
    )
    df['entry_short_cloud_exit'] = (
        (((df['Close']) > df[['senkou_a','senkou_b']].min(axis=1)).shift(2)) &
        df['price_move_down'] &
        df['price_below_tk'] &
        
        (
         df['price_below_cloud'] &
         (((df['Close'] + 50) < df['Open']).rolling(2).sum() == 2) &
         (df['Close'] <= df['rolling_min_10'] + df['ATR']*0.5) &
         (df['ADX'] > df['ADX'].shift(1)) &
         (df['-DI_up']) &
         (df['tenkan_below_kijun'])
        ) 
    )
    df['entry_short_price_bottom'] = (
        ((df['Close'] < df['kijun']).rolling(3).sum()==3) &
        ((df['Close']-10) < df['tenkan']) &
        df['tenkan_below_kijun'] &
        df['-DI_up'] &
        (df['Close'] <= df['rolling_min_5']) &
        (df['RSI'] < 50) &
        (df['red_cloud'] | 
         (df['green_cloud'] & (df['cloud_gap'] < df['ATR']) & 
          (df['cloud_gap'] < df['cloud_gap'].shift(1)))) &
        (df['OI'] > df['oi_ma']*1.02) &
        (((df['Close']-50) < df['Open']).rolling(3).sum() >= 2) &
        (df['candel_size'] > 80)
    )
    ####################################################################################
    ####################################################################################
    ####################################################################################

    
    df['Price_above_tk'] = (((df['Close'] > df['Close'].shift(1)).rolling(5).sum() >= 3) | 
                            ((df['Close'] > df['tenkan']) & 
                             (df['Close'] > df['kijun']) & 
                             (df['price_above_cloud']) &
                             ((df['Close'] > df['Open']).rolling(5).sum() >= 3)))
    df['Price_below_tk'] = ((df['Close'] < df['Close'].shift(1)).rolling(5).sum() >= 2)
    df['green_cloud'] = ((df['senkou_a'] > df['senkou_b'].shift(1)).rolling(5).sum() >= 4)
    df['red_cloud'] = ((df['senkou_a'] < df['senkou_b'].shift(1)).rolling(5).sum() >= 4)
    df['price_in_red_cloud'] = ((~df['price_above_cloud']) & (~df['price_below_cloud']) & df['red_cloud'])
    df['price_in_green_cloud'] = ((~df['price_above_cloud']) & (~df['price_below_cloud']) & df['green_cloud'])
    # df['price_in_cloud'] = ((~df['price_above_cloud']) & (~df['price_below_cloud']))
    df['price_in_bt_lines'] = ((~df['Price_above_tk']) & (~df['Price_below_tk']))
    df['Market_Regime'] = df.apply(classify_trend_ichimoku, axis=1)
    
    
    # Combine all into long entry
    

    
    df = df.copy()
    df['entry_long_below_cloud'] = (df['price_below_cloud'] &
                                    (df['Close'] > (df['rolling_min_10'].shift(1)+df['ATR']*0)) &
                                    ((df['Close']+150) > df['tenkan']) & 
                                    (df['Close'] > df['Close'].shift(1)) & 
                                    # (df['Close'].shift(1) > df['Open'].shift(1)) & 
                                    ((df['Close']-250) > df['Open']))
    df['entry_long_below_cloud'] = (df['price_below_cloud'] &
                                    (df['Close'] > (df['rolling_min_10'].shift(1)+df['ATR']*0.5)) &
                                    ((df['Close']+250) > df['tenkan']) & 
                                    (df['Close'] > df['Close'].shift(1)) & 
                                    ((df['Close']-250) > df['Open']))
    
    df['entry_long_below_cloud1'] = (df['price_below_cloud'] &
                                    (df['Close'] > (df['rolling_min_10'].shift(1)+df['ATR']*0.5)) &
                                    ((df['Close']+250) > df['tenkan']) & 
                                    (df['Close'] > df['Close'].shift(1)) & 
                                    # (df['Close'].shift(1) > df['Open'].shift(1)) & 
                                    (((df['Close']-50) > df['Open'])) &
                                    (((df['Close']+df['ATR']*0.4) > df['Open']).rolling(3).sum() > 1))
    df['entry_long_price']=(
                                (((df['Close'] > df['Open']).rolling(3).sum() >= 3) | (df['Close'] > df['rolling_max_10'].shift(1))+df['ATR']*0) &
                                (df['Close'] > cloud_top) &
                                ((df['Close'] < df['tenkan']).rolling(3).sum() >= 1)  &
                                price_above_tk
                            )
    df['entry_long_price_cloud'] = (
                                        entry_condition & 
                                        df['green_cloud'] &
                                        (df['Close'] > cloud_bottom) &
                                        ((df['Close']-50) > df['Close'].shift(1)) &
                                        (df['Close'] >= (df['rolling_min_10'] + df['ATR']*0.5))
                                    )
    df['entry_long_oi_vol'] = (
                                    df['tenkan_above_kijun'] &
                                    ((df['Close'] > df['Open']).rolling(3).sum() >= 2) &
                                    # (df['Close'] >= (df['rolling_min_10'] + df['ATR']*0.)) &
                                    ((df['Close'] >= (df['rolling_max_10'] - df['ATR']*1.5)) | 
                                     ((df['Close'] + 100) > df['Open']) |
                                     (df['price_above_tk'])
                                      )&
                                    (df['price_move_up']) )
    df['entry_long_oi_vol1'] = (
                                    df['tenkan_above_kijun'] &
                                    ((df['Close'] > df['Open']).rolling(3).sum() >= 2)& 
                                    (df['OI'] > (df['oi_ma']*0.95)) &
                                    (df['OI'] < (df['oi_ma']*1.1)))
    
    df['entry_long_price_enter_cloud'] = ((((df['Close']) > df['Open']).rolling(3).sum()>=3) &
                                          (((df['Close']-20) > df['tenkan']).rolling(2).sum()>=2) &
                                          (((df['Close']+10) > df['kijun']).rolling(2).sum()>=2) )
    df['entry_long_high'] = ( df['tenkan_above_kijun'] &
                              ((df['Close'] > df['Open']).rolling(3).sum() >= 2) &
                              ((df['Close']) >= df['rolling_max_10']-df['ATR']) &
                              (df['ADX'] > df['ADX'].shift(2)))    
    df['exit_long_price_drop1'] = (((df['line_gap'] > df['line_gap'].shift(1)).rolling(3).sum() >= 2) &
                                #    df['tenkan_below_kijun'] &
                                  ((df['Close'] > df['Close'].shift(1)).rolling(3).sum() >= 3) &
                                #   ((df['Close'] < df['kijun'].shift(1)).rolling(3).sum() >= 3) &
                                  ((df['Close'] > df['Open'].shift(1)).rolling(3).sum() >= 3))
    
    df['entry_gap_long'] = (
                                # (df['Close'] > df['Open']) & 
                                (((df['Close']+0) > df['Close'].shift(1)).rolling(3).sum() >= 2)&
                                (((df['Close']+100) > df['Open'].shift(1)).rolling(2).sum() >= 2)&
                                (((df['Close']+150) > df['tenkan'].shift(0)).rolling(2).sum() >= 1)&
                                df['price_above_cloud'] &
                                ((df['tenkan_above_kijun']) | 
                                 ((df['line_gap'] < df['line_gap'].shift(1)) &
                                  df['tenkan_below_kijun']) | 
                                 ((df['ADX'] > df['ADX'].shift(1)) &
                                  ((df['Close']-0) > df['Open']) &
                                  (((df['High'] - df['Low']) > 100).rolling(2).sum()==2)
                                  ))
                            )
    df['entry_pullback_long'] = (
                                    break_above_range &
                                    (df['Close'] > df[['senkou_a','senkou_b']].max(axis=1)) &  # trend up
                                    # (df['Close'] > df['kijun']) &                              # rebound from Kijun
                                    ((df['Low'].shift(1) < df['kijun'].shift(1)) | 
                                    (df['Low'].shift(2) < df['tenkan'].shift(2))) &              # yesterday dipped below kijun
                                    (df['Volume'] > df['vol_ma']*0.8)
                                )
    
    
    df['trailing_entry_long0'] = (
                                (df['Close'] > (df['rolling_min_10'].shift(1)+df['ATR']*0.5)) &
                                (df['Close'] > df['rolling_max_10'].shift(1)+df['ATR']*1) &
                                # (df['Close'] > df['trailing_entry_long']* 0.96) &
                                ((df['Open']+300) > df['Open'].shift(1)) &
                                # (((df['Open']+100) > df['Open'].shift(1)).rolling(2).sum() >= 2) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']-50) > df['Open'])& (df['OI'] > (df['oi_ma']*1.1))
                                )
    df['trailing_entry_long1'] = (
                                ((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*2)) &
                                (df['Close'] > (df['rolling_min_5'].shift(1)+df['ATR']*1.5)) &
                                # ((df['rolling_max_10'].shift(1) - df['rolling_min_10'].shift(1)) > df['ATR'].shift(1)*2) &
                                # (df['Close'] > df['trailing_entry_long']* 0.96) &
                                ((df['Open']+200) > df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']+100) > df['Open'].shift(0))& (df['OI'] > (df['oi_ma']*1.0))
                                )
    
    df['trailing_entry_long3'] = (
                                    ((df['Close'] > (df['rolling_max_10'] - df['ATR']*0)) |
                                    (df['Close'] > (df['rolling_min_10'] + df['ATR']*0.5))) &
                                    ((df['Close']+100) > df['Open'].shift(0))
                                )
    
    df = df.copy()

    # df['entry_short_kumo_break'] = (((df['Close']+100) < df['rolling_max_10'].shift(1)) &
    #                               (df['Close'].shift(1) > df[['senkou_a','senkou_b']].min(axis=1)) &
    #                               (df['Close'] < df[['senkou_a','senkou_b']].min(axis=1)) &
    #                               (df['Close'] < df['tenkan']) &
    #                               ((df['Close'] < df['Close'].shift(1)).rolling(2).sum() >= 2) &
    #                               (df['Volume'] > (df['vol_ma']*1)))
    df['entry_gap_short'] = (
                                ((df['Close'] < df['Open']).rolling(3).sum() >= 2) & 
                                (((df['Close']+50) < df['Close'].shift(1)).rolling(3).sum() >= 2)&
                                # (df['Close'].shift(1) < df['Open'].shift(1)) & 
                                ((df['Open']+50) < df['Open'].shift(1)) &
                                df['price_below_cloud']
                            )
    df['entry_short_above_cloud'] = np.where(df['price_above_cloud'], (
                                    ((df['Close']+150) < df['tenkan']) & 
                                    ((df['tenkan'] < df['tenkan'].shift(1)).rolling(2).sum() >= 2) &
                                    ((df['Close']) < df['Close'].shift(1)) &
                                    (df['price_red_candle']) &
                                    (df['Close'] < (df['rolling_max_10'].shift(0) - df['ATR']*1.5)) &
                                    (df['RSI'] < df['rsi_ma']) &
                                    
                                    ((df['Close'].shift(1) + 100) < df['Open'].shift(1)) 
                                    # (((df['Close'] + 100) < df['Open']).rolling(2).sum()>=2)
                                    ),False)
    df['entry_short_above_cloud2'] = np.where(df['price_above_cloud'], (
                                    ((df['Close']+250) < df['tenkan']) & 
                                    ((df['tenkan'] < df['tenkan'].shift(1)).rolling(2).sum() >= 2) &
                                    ((df['Close']) < df['Close'].shift(1)) &
                                    (df['Close'] < (df['rolling_max_10'].shift(0) - df['ATR']*1.5)) &
                                    ((df['Close'].shift(1)) < df['Open'].shift(1)) 
                                    ),False)
    df['entry_short_above_cloud1'] = ((df['Close'] <= df[['senkou_a','senkou_b']].min(axis=1)) &
                                   ((df['Close']) < df['Close'].shift(1)) &
                                   (df['Close'] < (df['rolling_min_10']+df['ATR'])) &
                                   ((df['Close']-50) < df['tenkan']))

    df['entry_short_oi_vol'] = (
                                (df['OI'] > df['oi_ma']) &
                                (df['Volume'] > df['vol_ma']) &
                                df['tenkan_below_kijun'] &
                                (df['Close'] < df['tenkan'] )&
                                df['price_move_down'] &
                                (((df['Close']+50) < df['Open']).rolling(3).sum() >= 2))
    
    df['entry_pullback_short'] = (
                                    ((df['Close']+100) < df[['senkou_a','senkou_b']].min(axis=1)) &  # trend down
                                    ((df['Close']-df['ATR']*0.8) < df['kijun']) &
                                    ((df['High'].shift(1) > df['kijun'].shift(1)) | 
                                    (df['Open'].shift(2) > df['tenkan'].shift(2))) &
                                    # (((df['Close']-100) < df['Close'].shift(1)).rolling(2).sum()>=1) &
                                    (((df['Close']) < df['rolling_max_10']-df['ATR']*0.1).rolling(1).sum()>=1) &
                                    (df['Volume'] > df['vol_ma']*0.8) 
                                )

    df['entry_short_price_drop'] = (((df['line_gap'] <= df['line_gap'].shift(1)).rolling(3).sum() >= 2) &
                                  (((df['Close']+20) < df['Close'].shift(1)).rolling(3).sum() >= 2) &
                                  (((df['Close']-0) < df['tenkan'].shift(0)).rolling(3).sum() >= 2) &
                                  (((df['Close']+0) < df['Open'].shift(0)).rolling(3).sum() >= 3) &
                                  (df['Close'] < (df['rolling_max_10'].shift(1) - df['ATR']*1.5)) &
                                  (df['price_above_cloud']))
    

    
    df['trailing_entry_short01'] = np.where(df['price_above_cloud'],
                                (
                                    # (df['Close'] < df['rolling_max_60']* 0.99) &
                                    (df['Close'] > df['rolling_min_60']* 1.0+df['ATR']) &
                                    (df['Close'] < (df['rolling_max_10']-100)) &
                                    ((df['Open']-100) < df['Open'].shift(1)) &
                                    ((df['Close']-50) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.1)) ),

                                    # ((df['Close'] < df['trailing_entry_short']* 0.96) &
                                    ((df['Close'] > df['rolling_min_60']* 1.0 + df['ATR']) &
                                    ((df['Open']+200) > df['Open'].shift(1)) &
                                    (((df['Close']-50) > df['Open'].shift(0))) &
                                    (((df['Close']-250) < df['Open'].shift(0))) &
                                    (df['OI'] > (df['oi_ma']*1.)) )
                                )
    df['trailing_entry_short0'] = np.where(df['price_above_cloud'],
                                (
                                    (df['Close'] > df['rolling_min_60']* 1.0+df['ATR']) &
                                    (df['Close'] < (df['rolling_max_10']-100)) &
                                    ((df['Open']-50) < df['Open'].shift(1)) &
                                    ((df['Close']-10) > df['Open']) &
                                    ((df['Close']-150) < df['Open']) &
                                    (df['ADX'] < 50)),
                        
                                    # ((df['Close'] < df['trailing_entry_short']* 0.96) &
                                    (
                                    (df['Close'] > df['rolling_min_60']* 1.0 + df['ATR']*1.3) &
                                    (df['Close'] <= df['rolling_max_10']* 1.0 + df['ATR']*0.1) &
                                    ((df['Open']+250) > df['Open'].shift(1)) &
                                    ((df['Open']-150) < df['Open'].shift(1)) &
                                    (((df['Close']-50) > df['Open'].shift(0))) &
                                    (((df['Close']-250) < df['Open'].shift(0))) &
                                    (df['RSI'] < 50) &
                                    (df['ADX'] > 25) &
                                    (df['OI'] > (df['oi_ma'])*0.9) )
                                )
    df['trailing_entry_short02'] = np.where(df['price_above_cloud'],
                                (
                                    (df['Close'] > df['rolling_min_60']* 1.0+df['ATR']) &
                                    (df['Close'] < (df['rolling_max_10']-100)) &
                                    ((df['Open']-50) < df['Open'].shift(1)) &
                                    ((df['Close']-50) > df['Open']) &
                                    (df['ADX'] < 50)),

                                    # ((df['Close'] < df['trailing_entry_short']* 0.96) &
                                    ((df['Close'] > df['rolling_min_60']* 1.0 + df['ATR']) &
                                    (df['Close'] < (df['rolling_max_10']-df['ATR'])) &
                                    #  (df['ADX'] > df['ADX'].shift(1)) &
                                    ((df['Open']+100) > df['Open'].shift(1)) &
                                    (((df['Close']-50) > df['Open'].shift(0))) &
                                    (((df['Close']-200) < df['Open'].shift(0))) &
                                    (df['OI'] > (df['oi_ma'])*1.0)&
                                    (df['ADX'] > 20) )
                                )
    
    df['trailing_entry_short1'] = np.where(df['price_above_cloud'],
                                (
                                    ((df['Close']-0) < (df['rolling_max_10']-df['ATR']*1)) &
                                    (((df['Close']) < df['kijun']).rolling(2).sum() >= 2 )&
                                    ((df['Open']-10) < df['Open'].shift(1)) &
                                    (df['ADX'] > df['ADX'].shift(1)*0.92) &
                                    ((df['Close']+100) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.07))
                                ),
                                (
                                    (((df['Close'])< df['kijun']).rolling(3).sum() >= 2 )&
                                    ((df['Open']-10) < df['Open'].shift(1)) &
                                    (df['ADX'] > df['ADX'].shift(1)*0.92) &
                                    ((df['Close']+0) < df['Open'])& (df['OI'] > (df['oi_ma']*0.9)) &
                                    (df['OI'] < (df['oi_ma']*1.1))
                                ))
    
    
    
    df['entry_long_avoid'] = (
                              df['price_below_cloud'] &
                              ((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*2)))
    df['entry_long_avoid1'] = (
                              (df['price_below_cloud']) &
                              ((df['rolling_max_5'] - df['rolling_min_5']) < (df['ATR']*2)))
    df['entry_short_avoid'] = (df['price_above_cloud'] &
                              ((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*2)))
    df['entry_short_avoid1'] = (df['price_above_cloud'] &
                              ((df['rolling_max_5'] - df['rolling_min_5']) < (df['ATR']*2)))
    
    # Exit Long
    df = df.copy()
    # df['exit_long_price_turn'] = ((((df['High'] - df['Open']) > (df['Open'] - df['Low'])*1.5).rolling(2).sum()>=1) &
    #                                (df['price_above_cloud']) &
    #                                (((df['Close']+50) < df['Open']).rolling(4).sum()>=3) &
    #                                (((df['Close']+10) < df['Close'].shift(1)).rolling(2).sum()>=2) &
    #                                (df['Close'] < (df['rolling_max_10'] - df['ATR']*0)) &
    #                                ((df['Close']+100) < df['tenkan'])) 
    # price_difference = ((abs(df['kijun'] - df['senkou_a']) > abs(df['senkou_a'] - df['senkou_b'])) &
    #                     (abs(df['kijun'] - df['senkou_a']) > df['ATR']*2))
    
    # price_difference = ((abs(df['senkou_a'] - df['senkou_b']) > df['ATR']*2) &
    #                     # (abs(df['kijun'] - df['senkou_a']) > abs(df['senkou_a'] - df['senkou_b'])) &
    #                     (abs(df['kijun'] - df['senkou_a']) > df['ATR']*3))

    # df['exit_long_price_turn2'] = np.where(price_difference,
    #                                 (((df['High'] - df['Open']) > (df['Open'] - df['Low'])*1.5).rolling(2).sum()>=1) &
    #                                (df['price_above_cloud']) &
    #                                (((df['Close']+50) < df['Open']).rolling(4).sum()>=3) &
    #                                (((df['Close']+10) < df['Close'].shift(1)).rolling(2).sum()>=2) &
    #                                (df['Close'] < (df['rolling_max_10'] - df['ATR']*0)) &
    #                                ((df['Close']+100) < df['tenkan']) &
    #                                ((df['Close']+30) < df['kijun']),
    #                                (((df['High'] - df['Open']) > (df['Open'] - df['Low'])*1.5).rolling(2).sum()>=1) &
    #                                (df['price_above_cloud']) &
    #                                (((df['Close']+50) < df['Open']).rolling(4).sum()>=3) &
    #                                (((df['Close']+10) < df['Close'].shift(1)).rolling(2).sum()>=2) &
    #                                (df['Close'] < (df['rolling_max_10'] - df['ATR']*0)) &
    #                                ((df['Close']+100) < df['tenkan'])) 
    
    # df['exit_long_price_turn1'] =  ((((df['High'] - df['Open']) > (df['Open'] - df['Low'])*1).rolling(3).sum()>=2) &
    #                                 (df['price_above_cloud']) &
    #                                (((df['Close']+20) < df['Open']).rolling(3).sum()>=2) &
    #                                (df['Close'] < (df['rolling_max_10'] - df['ATR']*1)) &
    #                                (((df['Close']) < df['kijun']).rolling(2).sum()>=2) &
    #                                ((df['RSI'] < df['rsi_ma']).rolling(3).sum() >= 3)
    #                                )
    
    df['exit_long_avoid'] = (((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*0.8)))
    
    # df['exit_long_price_drop'] = (((df['line_gap'] <= df['line_gap'].shift(1)).rolling(3).sum() >= 2) &
    #                               ((df['Close'] < df['Close'].shift(1)).rolling(3).sum() >= 2) &
    #                               ((df['Close'] < df['tenkan'].shift(0)).rolling(3).sum() >= 3) &
    #                               ((df['Close'] < df['Open'].shift(0)).rolling(3).sum() >= 3) &
    #                               (df['Close'] < (df['rolling_max_10'].shift(1) - df['ATR']*1)) &
    #                               (df['price_above_cloud']))
    
    df['exit_long_price_down'] = (df['price_above_cloud'] &
                                  (((df['Close']+20) < df['Close'].shift(1)).rolling(2).sum() >= 2)&
                                  (((df['Close']-50) < df['tenkan'].shift(1)).rolling(2).sum() >= 1)&
                                  (df['Close'] < (df['rolling_max_10']-(df['ATR']*2))) &
                                  (df['range1'] > df['ATR']*2) &
                                  (df['ADX'] < df['ADX'].shift(1)) &
                                  ((df['-DI_up'] & (df['RSI'] < 70)) | ((df['DI_gap'] < df['DI_gap'].shift(2)) & df['+DI_move_down'] & (df['RSI'] < 60)) | 
                                   ((df['OI'] < df['oi_ma'])) ) &
                                  (df['Volume'] < (df['vol_ma']*1.1))
                                  )
    df['exit_long_kijun'] = (((df['Close'] < df['kijun']).rolling(3).sum() >= 3) &
                             (((df['Close']-100) < df['Open'].shift(0)).rolling(2).sum()>=1) &
                             (((df['tenkan']+50) < df['kijun']).rolling(2).sum()>=2) &
                             (df['RSI'] < 50) &
                             ((df['OI'] < df['oi_ma']*1.07).rolling(2).sum()>=2)
                             )
    df['exit_long_kumo_break'] = (((df['Close']) < df['rolling_max_10'].shift(1) - df['ATR']*0.5) &
                                  (df['Close'].shift(1) > df[['senkou_a','senkou_b']].min(axis=1) ) &
                                  ((df['Close']) < df[['senkou_a','senkou_b']].min(axis=1)) &
                                  ((df['Close']+50) < df['tenkan']) &
                                  (((df['Close']) < df['Close'].shift(1)).rolling(3).sum() >= 2) &
                                  (df['Volume'] > (df['vol_ma']*1)))
    

    df['exit_long_tkcross'] = ((df['tk_cross_down'].rolling(3).sum() >= 2) | 
                                
                                (((df['Close'].shift(1) > df['tenkan']) | (df['Open'].shift(1) > df['kijun'])) &
                                (df['Open'] > (df['Close']+200)) &
                                ((df['Close']+50) < df['tenkan']) &
                                (df['Close'] < df['kijun']) &
                                (df['Close'] <= (df['rolling_min_10']+df['ATR']*0.5)) &
                                (df['Volume'] < df['vol_ma']*1.1) ) | (
                                
                                df['price_above_cloud'] &
                                (df['Close'] <= (df['rolling_min_10']+df['ATR']*0.1)) &
                                ((df['line_gap'] <= df['line_gap'].shift(1)).rolling(2).sum() == 2) &
                                ((df['Close']+50) < df['tenkan']) &
                                (df['Close'] < df['kijun']) &
                                (df['Open'] > (df['Close']+200)) &
                                (((df['Close'].shift(1)+50) > df['tenkan']) | ((df['Open'].shift(1)+50) > df['kijun'].shift(1))) 
                                ))
    
    # df['exit_long_tkcross1'] = (((df['Close'] <= df['kijun']).rolling(3).sum() >= 3) &
    #                             (df['OI'] < df['oi_ma']*1.05) &
    #                             (((df['tenkan']-df['ATR']) <= df['kijun']).rolling(3).sum() >= 2) &
    #                             ((df['Close'] <= df['Close'].shift(1)).rolling(3).sum() >= 2) &
    #                             (df['Close'] < (df['rolling_max_10']-df['ATR'])) )
    # df['exit_long_price_reversion'] = ((((df['Close']+0) < df['Open'].shift(0)).rolling(3).sum() >= 3) &
    #                                    (((df['Close']+10) < df['Close'].shift(1)).rolling(2).sum() >= 2) &
    #                                    (((df['Close']+50) < df['tenkan']).rolling(2).sum() >= 2) &
    #                                    ((df['RSI'] < df['RSI'].shift(1)).rolling(3).sum()>=3))
    extra_condition = ((df['Open'] - df['Close']) >= 50).rolling(2).sum() >= 2

    df['exit_long_price_cloud'] = np.where(
        extra_condition,
        ((df['Close'] >= cloud_top) & (df['Close'] < df['kijun']) | (
            (df['Close'] < df['senkou_b']) &
            (df['Open'] > df['senkou_b']) &
            (df['Close'] < df['kijun']) 
        )),   # WHEN ABOVE CLOUD + TENKAN
        exit_condition & (df['Low'] >= cloud_top)                      # OTHERWISE
    )


    
    
    df['trailing_stop_long0'] = (df['Close'].expanding().max() - 2 * df['ATR'])
    # df['trailing_stop_long0'] = (df['Close'].rolling(window=100).max() - 2 * df['ATR'])
    
    
    
    # df['trailing_stop_long21'] = ((
    #                             df['price_above_cloud'] &
    #                             (df['Close'] < df['trailing_stop_long0']* 0.955) &
    #                             ((df['Open']-20) < df['Open'].shift(1)) &
    #                             ((df['Close']+300) > df['Open'].shift(0)) & 
    #                             ((df['Close']-200) < df['Open'].shift(0)) & 
    #                             (df['OI'] < (df['oi_ma']*1.2))
    #                             ) | (
    #                             # df['price_below_cloud'] &
    #                             (df['Close'] <= (df['rolling_max_10']-(df['ATR']*1.5))) &
    #                             (((df['Open']-30) < df['Open'].shift(1)).rolling(2).sum() >= 2) &
    #                             ((df['Close']-150) > df['Open'].shift(0)) & 
    #                             (((df['Close']-100) < df['Open']).rolling(3).sum() > 1) &
    #                             (df['OI'] < (df['oi_ma']*1.3))
    #                             ))
    df['trailing_stop_long21'] = ((
                                df['price_above_cloud'] &
                                df['green_cloud'] &
                                (((df['Close'] - 100) < df['tenkan'].shift(1)).rolling(2).sum() == 2) &
                                (df['RSI'] < df['rsi_ma']) &
                                (df['Close'] <= (df['rolling_min_10']+(df['ATR']*0.5))) &
                                ((df['Close']+50) < df['Open'].shift(0)) 
                                ) |
                                
                                (df['price_above_cloud'] &
                                 ((df['Close'] < df['Open']).rolling(2).sum() == 2) &
                                 (df['candel_size'] > 100) &
                                 (df['Close'] < (df['rolling_min_5'].shift(1))) &
                                 (df['RSI'] < df['rsi_ma']) &
                                 df['RSI']
                                
                                ) | (
                                # df['price_below_cloud'] &
                                (df['Close'] <= (df['rolling_max_10']-(df['ATR']*1.5))) &
                                (((df['Open']-30) < df['Open'].shift(1)).rolling(2).sum() >= 2) &
                                ((df['Close']-150) > df['Open'].shift(0)) & 
                                (((df['Close']-100) < df['Open']).rolling(3).sum() > 1) &
                                (df['OI'] < (df['oi_ma']*1.3))
                                )
                                )
    df['trailing_stop_long'] = (
                                (df['Close'] < (df['rolling_max_10']-df['ATR'])) &
                                ((df['Open']-10) < df['Open'].shift(1)) &
                                ((df['Close']-50) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.1))
                                )
    df['trailing_stop_long11'] = (
                                (df['Close'] < (df['rolling_min_10']+df['ATR'])) &
                                ((df['Open']+50) < df['Open'].shift(1)) &
                                (((df['Close']+100) < df['Open']).rolling(2).sum() >= 2)& 
                                (df['OI'] < (df['oi_ma']*1.1))
                                )
    df['trailing_stop_long3'] = np.where(df['price_above_cloud'],
                                (((df['Close'] - 150)< df['Open']) & df['trailing_stop_long'] &  (df['OI'] < (df['oi_ma']*0.9))),
                                df['trailing_stop_long11'])
    
    df['trailing_stop_long4'] = np.where(df['price_above_cloud'],
                                         (((df['Close'] - 150) > df['Open']) & 
                                          (df['Close'] < df['rolling_min_5'] + df['ATR']*1) &
                                          (((df['Close'] - 50) < df['Close'].shift(1)).rolling(3).sum() == 2) &
                                          (df['price_below_tenkan']) & 
                                          (df['ADX'] < df['ADX'].shift(1)) &
                                          (df['range1'].shift(1) < df['ATR']*2.5)
                                          ),
                                         False)
    


    # Exit Short
    # Exit Short
    # Exit Short
    # Exit Short
    df = df.copy()
    extra_condition1 = ((df['Close'] - df['Open']) >= 20).rolling(2).sum() >= 2
    df['exit_short_price_cloud1'] = np.where(
                                            extra_condition1,
                                            (df['Close'] <= cloud_bottom) & (df['High'] <= cloud_top),   # WHEN ABOVE CLOUD + TENKAN
                                            entry_condition & (df['High'] <= cloud_top)                   # OTHERWISE
                                        ) 
    # df['exit_short_price_cloud1'] = np.where(
    #                                         extra_condition1,
    #                                         ((df['Close'] <= cloud_bottom) & 
    #                                          (df['High'] <= cloud_top) &
    #                                          ((df['red_cloud'] &
    #                                          (df['Close'] > df['rolling_max_5']- df['ATR']*0.8) &
    #                                          (df['RSI'] > 30)
    #                                          ) | 
    #                                          (df['green_cloud']))),   # WHEN ABOVE CLOUD + TENKAN
    #                                         entry_condition & (df['High'] <= cloud_top)                   # OTHERWISE
    #                                     ) 
    # df['exit_short_high'] = (((df['Close']-100) > df['rolling_min_10'].shift(0)) &
    #                         ((df['Close'] > df['Open']).rolling(3).sum() >= 2) &
    #                         ((df['Close'] > df['Close'].shift(1)).rolling(3).sum() >= 2) &
    #                         ((df['High'] > df['tenkan'].shift(1)).rolling(3).sum() >= 1) &
    #                         (df['Volume'] > (df['vol_ma']*1))
    #                         )
    # df['exit_short_cloud_tenkan'] = (df['price_below_cloud'] &
    #                                  df['price_above_tenkan'] &
    #                                  df['tenkan_above_kijun'] &
    #                                  ((df['Close'] > df['Open']).rolling(3).sum() >= 2) )
    

    
    df['exit_short_avoid'] = (
                              (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1)) &
                              ((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*1)))
    
    df['exit_short_price_cloud'] = (((df['Close'] < df['Open']).rolling(3).sum() == 3) & 
                                    (df['RSI'] < df['rsi_ma']) &
                                    (df['RSI'] < 50) &
                                    (df['High'] <= cloud_top))
    df['exit_short_price_cloud2'] = (
                                    (((df['Close']-50) < df['Open']).rolling(3).sum() == 3) & 
                                    (df['RSI'] < df['rsi_ma']) &
                                    (df['RSI'] > 20) &
                                    (df['High'] <= cloud_top))
    
    df['exit_short_tenkan'] = (((df['Close'] > df['Open']).rolling(3).sum() >= 3) &
                               (((df['Close']-50) > (df['tenkan'])).rolling(3).sum() >= 2))
    
    df['exit_short_above_cloud'] = ((df['Close'] >= df[['senkou_a','senkou_b']].max(axis=1)) &
                                    ((df['Close']-100) > df['Close'].shift(1)) &
                                    (((df['Close']).shift(1) + 200) > df['Open'].shift(1)) &
                                    (((df['Close']) > df['tenkan']).rolling(1).sum() == 1) &
                                    (df['RSI'] < 60) &
                                    (df['RSI'] > 40) &
                                    (df['ADX'] < 70)
                                    ) 
    
    df['exit_short_kijun'] = (((df['Close'] > df['kijun']).rolling(3).sum() >= 3) &
                              (df['Close'] > df['rolling_min_10']+df['ATR']))
    


    # df['trailing_stop_short0'] = df['Close'].expanding().max() - 2 * df['ATR']
    df['trailing_stop_short'] = (
                                # (df['Close'] > (df['trailing_stop_short0']*0.89)) &
                                (df['Close'] > (df['rolling_min_10']+(df['ATR']*1))) &
                                ((df['Open']-50) > df['Open'].shift(1)) &
                                ((df['Open']-0) > df['Close']) 
                                )
    df['trailing_stop_short01'] = (
                                # (df['Close'] > (df['trailing_stop_short0']*0.89)) &
                                (df['Close'] > (df['rolling_min_10']+(df['ATR']*1))) &
                                ((df['Open']-100) > df['Open'].shift(1)) &
                                ((df['Open'] + 50) > df['Close']) &
                                ((df['Open'] - 150) < df['Close']) &
                                (df['ADX'] < df['ADX'].shift(1))
                                )
    df['trailing_stop_short1'] = (
                                (df['Close'] > (df['rolling_min_10']+(df['ATR']*0.8))) &
                                ((df['Close']-0) < df['Open'].shift(1)) &
                                ((df['Open']+0) < df['Close'].shift(0)) &
                                (df['Volume'] < df['vol_ma']*1) 
                                )
    df['trailing_stop_short3'] = (
                                (df['Close'] > (df['rolling_min_10']+(df['ATR']*0.9))) &
                                ((df['Close']+df['ATR']) > df['Open'].shift(1)) &
                                ((df['Open']+df['ATR']) < df['Close']) 
                                )
    df['exit_long_tenkan_cross'] = (
        df['price_above_tenkan'].shift(1) &
        df['price_below_tenkan'] &
        df['price_above_cloud'] &
        ((df['Close'] + 200) < df['Open']) &
        ((df['RSI'] < df['RSI'].shift(1)).rolling(2).sum() == 2) &
        # (df['+DI_move_down']) &
        df['red_cloud'] 
        # (df['DI_gap'] < 20) &

    )
    df['exit_long_below_cloud_kijun'] = (
        df['price_below_cloud'] &
        df['tenkan_below_kijun'] &
        # ((df['tenkan_kijun_gap'] > df['tenkan_kijun_gap'].shift(1)).rolling(3).sum() == ) &
        (((df['Close'] + 150) < df['kijun']).rolling(3).sum() == 3) &
        (((df['Close'] + 0) > df['tenkan']).rolling(3).sum() == 3) &
        (((df['Close']+50) < df['Open']).rolling(3).sum() == 3) &
        (df['Close'] < df['rolling_max_5']-df['ATR']*1) &
        (df['line_gap'] > df['ATR']*1) &
        
        # (df['-DI_move_'])
        (df['-DI_up'])
    )

    df['exit_short_move_up'] = (
        df['-DI_move_down'] &
        (((df['Close']) > df['Open']).rolling(2).sum() >=1) &
        (df['Close'] > df['rolling_min_5']+df['ATR']*1) &
        df['red_cloud'] &
        ((df['Open'].rolling(2).max() > (df['Close']))) 
    )
    df['exit_long_incloud'] = (
        (df['Close'] < df[['senkou_a','senkou_b']].max(axis=1)) &
        (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1)) &
        # ((df['Close']+df['ATR']*2) > df[['tenkan','kijun']].min(axis=1)) &
        df['tenkan_below_kijun'] &
        df['price_below_tk'] &
        (((df['Close'] + 30) < df['Open']).rolling(3).sum() >= 3) &
        (df['Volume'] > df['vol_ma']*1.1) &
        df['price_move_down'] &
        df['-DI_up'] &
        (df['+DI_move_down'].rolling(4).sum() >= 3) &
        ((df['Close']-10) <= df['rolling_min_5']) &
        (df['RSI'] < df['rsi_ma']*0.8)
    )
    df['exit_long_below_cloud'] = (
        (df['Close'] < df[['senkou_a','senkou_b']].max(axis=1)) &
        # (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1)) &
        # ((df['Close']+df['ATR']*2) > df[['tenkan','kijun']].min(axis=1)) &
        df['tenkan_below_kijun'] &
        ((df['tenkan'] < df['kijun']).rolling(3).sum() >= 3) &
        # df['red_cloud'] &
        
        df['price_above_tenkan'].shift(2) &
        df['price_below_tk'] &
        (((df['Close'] + 100) < df['Open']).rolling(2).sum() >= 2) &
        (df['candel_size'] > 200) &
        (df['Volume'] > df['vol_ma']*1.2) &
        df['price_move_down'] &
        df['-DI_up'] &
        (df['+DI_move_down'].rolling(3).sum() >= 3) &
        (df['-DI_move_up'].rolling(3).sum() >= 3) &
        ((df['Close']+0) <= df['rolling_min_5']) &
        (df['RSI'] < df['rsi_ma']*0.8) &
        (((df['RSI']) < df['RSI'].shift(1)).rolling(3).sum() >= 3)

    )
    df['exit_short_pullback'] = (
        ((df['Open'] - 20) < df[['senkou_a','senkou_b']].max(axis=1)) &
        (df['Close'] > df[['senkou_a','senkou_b']].max(axis=1)) &
        # df['green_cloud'] &
        ((df['Close'] - 200) > df['Open']) &
        ((df['High'] + 50) > df[['tenkan','kijun']].min(axis=1)) &
        ((df['Close'] + 50) > df['Open'].shift(1))
    )
    df = df.copy()
    #### df['exit_short_cloud_tenkan'] | df['entry_kumo_break_long'] | df['entry_long'] | df['entry_pullback_long'] | df['trailing_entry_long2'] | 
    final_unique_long_condition = (df['entry_long_price'] | df['entry_long_price_cloud'] | df['entry_long_oi_vol'] | df['entry_long_below_cloud'] | 
                                   df['entry_long_oi_vol1'] | df['entry_long_price_enter_cloud'] | df['entry_long_high'] | 
                                   df['exit_long_price_drop1'] | df['entry_gap_long'] | 
                                   df['trailing_entry_long0'] | df['trailing_entry_long1'] | df['trailing_entry_long3']
                                )

    df['final_entry_long'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [
            df['entry_long_below_tk'] | df['trailing_entry_long1'] | df['exit_long_price_drop1'] |  df['entry_long_below_cloud']  | ((df['exit_short_price_kijun']| df['entry_long_high'] | df['trailing_stop_short3']| df['trailing_entry_long0'])& (~df['entry_long_avoid'])),  
            df['exit_short_price_tk_cross2'] | df['entry_long_bt_tk'] |((df['exit_short_tenkan_cross'] | df['entry_long_below_tk'] |   df['trailing_entry_long3']) & (~df['entry_long_avoid1']) & (~df['entry_short_avoid'])),  
            df['exit_short_price_tk_cross2'] | df['exit_short_tenkan_cross'] | df['entry_long_bt_tk'] | df['entry_long_below_tk'] | df['exit_long_price_drop1'] | df['entry_long_price_enter_cloud'] | df['entry_long_price_cloud'] | df['entry_long_oi_vol1'] ,
            df['exit_short_price_kijun']| df['entry_long_below_tk'] | df['exit_long_price_drop1'] | df['entry_long_below_cloud'] | df['entry_long_high'] |  ((df['entry_long_oi_vol'] | df['entry_long_price']  | df['entry_gap_long']) & (~df['entry_long_avoid']))  
        ],
        default= df['exit_short_price_kijun']| df['exit_short_price_tk_cross2']|  df['exit_short_price_move_up'] | df['exit_short_tenkan_cross'] | df['entry_long_bt_tk'] |   df['entry_long_below_cloud'] | df['exit_short_price_cloud'] 
    ) | ( df['entry_long_tenkan_cross'] |df['entry_long_cloud_enter'] |  df['entry_long_tk_cross'] |  df['entry_long_head_candle'] | 
         df['entry_long_below_tk_cloud'] | df['entry_long_in_cloud'] | 
          df['exit_short_at_top']  ##df['exit_short_price_tk_cross1'] | 
           ) 
        # df['Entry_long_above_all'])
    # df['final_entry_long'] = False
    
    
    final_unique_exit_long = (df['entry_short_above_cloud'] | 
                              df['exit_long_tkcross'] | df['exit_long_kijun'] |  df['exit_long_price_cloud'] | 
                              df['exit_long_price_down'] | df['exit_long_kumo_break'] | 
                              df['trailing_stop_long21'] | df['trailing_stop_long3'] | df['trailing_stop_long4'] 
                            )

    
    df['exit_long_final'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [   
            ((df['exit_long_price_kijjun1']|  df['exit_long_cloud_entry'] |df['exit_long_price_kijun'] | df['trailing_stop_long21']| df['exit_long_price_tk_cross2'] )& (~df['entry_long_avoid'])& (~df['entry_short_avoid'])) , 
            df['exit_long_price_kijjun1']| df['exit_long_cloud_entry'] |df['exit_long_price_kijun'] |df['exit_long_price_tenkan1'] | df['exit_long_price_tk_cross2'] ,
            df['exit_long_price_kijjun1'] | df['exit_long_incloud']|  df['exit_long_price_kijun'] | df['exit_long_tkcross'] | df['exit_long_price_tenkan1']| df['exit_long_price_tk_cross2'] ,  
            ((df['exit_long_price_kijjun1']| df['exit_long_incloud'] | df['exit_long_cloud_entry']  | df['exit_long_price_kijun']   | df['exit_long_kijun'] | df['trailing_stop_long3'] | df['exit_long_kumo_break'])& (~df['entry_short_avoid1']))###| df['exit_long_price_cloud']    ###df['exit_long_price_drop'] |df['trailing_stop_long4'] |  ##| df['exit_long_price_down'] | df['exit_long_price_tk_cross2']| df['exit_long_price_tenkan1']
        ],
        default= df['exit_long_price_kijjun1'] | df['exit_long_cloud_entry'] | df['exit_long_price_tenkan1'] | df['exit_long_price_tk_cross2'] |  df['entry_short_above_cloud'] |  df['exit_long_avoid'] |  ((df['exit_long_price_down'] | df['exit_long_tkcross'] | df['trailing_stop_long4'])& (~df['entry_long_avoid']))     ### ###df['exit_long_price_turn2'] | | df['trailing_stop_long3'] 
    ) | ( df['exit_long_price_tk_cross3'] | df['entry_short_price_top']| df['exit_long_cloud_exit']   ) ##|  df['exit_long_below_tk'] 
    ### df['exit_long_price_tk_cross2'] | df['exit_long_price_tenkan1'] | df['exit_long_price_kijun'] | df['exit_long_price_kijjun1']| df['exit_long_cloud_entry'] 
    
    
    ##df['entry_short_kumo_break'] | 
    final_unique_entry_short = (df['entry_gap_short'] | df['entry_pullback_short'] | 
                                df['entry_short_oi_vol']  | df['entry_short_above_cloud1'] | df['entry_short_above_cloud'] | 
                                df['trailing_entry_short0'] | df['trailing_entry_short1'])


    df['final_entry_short'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [   
            ((df['exit_long_price_tk_cross2'] | df['entry_short_price_bottom'] | df['exit_long_price_tk_cross3'] | df['trailing_entry_short0'] )& (~df['entry_short_avoid'])) , 
            df['exit_long_price_tk_cross1'] | df['exit_long_price_tenkan1']| df['exit_long_price_kijun'] | (( df['entry_gap_short']  | df['entry_short_price_bottom'] | df['trailing_entry_short01'] | df['trailing_entry_short1'] ) & (~df['entry_short_avoid'])) ,  ##| df['entry_short_above_cloud1']
            ((df['exit_long_price_kijun'] | df['exit_long_price_tk_cross2'] | df['entry_short_price_bottom'] | df['exit_long_price_tenkan1'] | df['trailing_entry_short02'] | df['exit_long_tkcross'] )& (~df['entry_short_avoid'])& (~df['entry_long_avoid'])),  
            df['exit_long_price_tk_cross1'] |((df['exit_long_price_tk_cross2'] | df['exit_long_price_tenkan1']| df['entry_pullback_short'] | df['entry_short_oi_vol']  )) ## | df['entry_short_above_cloud']   
        ],
        default=  ((df['exit_long_price_tk_cross2']  | df['entry_short_price_bottom'] | df['exit_long_price_tk_cross3'] | df['entry_pullback_short'] | df['entry_short_price_drop'] | 
                    df['exit_long_price_down'] | df['trailing_entry_short1'] | df['entry_short_above_cloud2'])) ## df['exit_long_price_drop'] |
    )| ( df['exit_long_price_kijjun1']| df['exit_long_price_tenkan'] |  df['exit_long_cloud_entry'] |
        df['entry_short_price_top'] | df['entry_short_above_tk'] | df['entry_short_cloud_exit']
          ) ##df['entry_short_cloud_enter'] | 
    # df['final_entry_short'] = False
    ## df['exit_long_price_tk_cross1'] |df['exit_long_price_tk_cross3'] | 
##| df['trailing_entry_long2'] df['exit_short_high'] | df['exit_short_cloud_tenkan'] | df['exit_short_price_turn'] | df['exit_short_tkcross'] | 
    final_unique_exit_short = (df['exit_short_price_cloud'] | df['exit_short_price_cloud1'] | 
                               df['exit_short_avoid'] | 
                               df['exit_short_kijun'] | df['exit_short_tenkan'] | df['exit_short_above_cloud'] |
                               df['trailing_stop_short'] | df['trailing_stop_short1']  | df['trailing_stop_short3']
    ) 
    
    
    df['final_exit_short'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [

            ((df['entry_long_cloud_enter'] | df['trailing_entry_long0'] | df['entry_long_below_cloud1'] | df['trailing_stop_short01']  )& (~df['entry_short_avoid'])) ,  
            False , ##df['exit_short_price_cloud2'] | df['exit_short_cloud_enter'],
            df['exit_short_cloud_enter'] |  df['exit_short_avoid'] |  ((  df['exit_short_kijun'] | df['exit_short_tenkan'])& (~df['entry_short_avoid'])& (~df['entry_long_avoid'])) ,
            ((df['entry_long_cloud_enter'] | df['exit_short_price_cloud2'] | df['exit_short_above_cloud'] )& (~df['entry_long_avoid'])) ##df['exit_short_cloud_enter'] | 
        ],
        default = df['entry_long_cloud_enter'] | df['exit_short_avoid']| df['exit_short_cloud_enter'] | df['exit_short_tenkan'] | df['exit_short_price_cloud1'] | df['trailing_stop_short3']
    ) | (    ### df['entry_long_bt_tk'] | 
        df['exit_short_tenkan_cross'] | df['exit_short_price_tk_cross2'] |  df['exit_short_price_tk_cross1'] | df['exit_short_at_top'] | df['exit_short_avoid_range'] |
         df['exit_short_price_kijun'] |  df['exit_short_price_move_up'] |
          df['entry_long_below_tk_cloud'] | df['exit_short_move_up'] | df['exit_short_pullback'])
    
    
    # df['exit_long'] | df['exit_long_below_cloud'] | df['exit_long_tkcross'] | df['exit_long_price_cloud'] | df['exit_long_tenkan'] | df['exit_long_kijun'] | df['trailing_stop_long']
    # df['exit_short'] | df['exit_short_above_cloud'] | df['exit_short_tkcross'] | df['exit_short_price_cloud'] | df['exit_short_tenkan'] | df['exit_short_kijun'] | df['trailing_stop_short']
    
    # df.to_csv('strategy_file.csv', index=False)
    return df
def stopless_point1(row, position, entry_price, prow):
    price = row['Close']
    open = row['Open']
    high = row['High']
    pprice = prow['Close']
    tenkan = row['tenkan']
    kijun = row['kijun']
    senkou_a = row['senkou_a']
    senkou_b = row['senkou_b']
    atr = row['ATR']
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    price_tenkan = True if (price > tenkan) else False
    price_kijun = True if (price > kijun) else False
    price_cloud = True if (price > max(senkou_a, senkou_b)) else False
    price_cloud1 = True if (price < min(senkou_a, senkou_b)) else False
    price_incloud = True if (cloud_top >= price >= cloud_bottom) else False
    tenkan_kijun = True if (tenkan >= kijun) else False
    tenkan_senkou_a = True if (tenkan >=senkou_a) else False
    tenkan_senkou_b = True if (tenkan >=senkou_b) else False
    kijun_senkou_a = True if (kijun >=senkou_a) else False
    kijun_senkou_b = True if (kijun >=senkou_b) else False
    senkou_a_b = True if (senkou_a > senkou_b) else False
    price_tenkan_diff = price - tenkan
    price_kijun_diff = price - kijun
    tenkan_kijun_diff = tenkan - kijun
    tenkan_senkou_a_diff = tenkan - senkou_a
    tenkan_senkou_b_diff = tenkan - senkou_b
    kijun_senkou_a_diff = kijun - senkou_a
    kijun_senkou_b_diff = kijun - senkou_b
    senkou_a_b_diff = abs(senkou_a - senkou_b)
    price_senkou_a_diff = price - senkou_a
    multiplier = 3
    stoploss_value = 0
    if position == 1:
        if price_tenkan & price_kijun & price_cloud:
            # stoploss_value = 0
            if tenkan_kijun & senkou_a_b:
                if (tenkan_senkou_a_diff > atr*3) and (tenkan_kijun_diff > atr*1):
                    stoploss_value = kijun + atr*0
                elif (tenkan_senkou_a_diff > atr*1) and (tenkan_kijun_diff > atr):
                    stoploss_value = senkou_a + atr*0
                elif (tenkan_kijun_diff < atr):
                    stoploss_value = kijun + atr*2
            elif tenkan_kijun and not senkou_a_b:
                if (tenkan_senkou_a_diff > atr*3) and (tenkan_kijun_diff > atr):
                    stoploss_value = kijun + (tenkan - kijun)/2 + atr*0.2
                elif (tenkan_senkou_a_diff > atr*1) and (tenkan_kijun_diff < atr):
                    stoploss_value = senkou_a - atr*1.5
                elif (tenkan_kijun_diff < atr):
                    stoploss_value = kijun
                else:
                    stoploss_value = senkou_b
            elif not tenkan_kijun and senkou_a_b:
                if (tenkan_senkou_a_diff > atr*3):
                    stoploss_value = tenkan - atr*2
                elif (tenkan_senkou_a_diff > atr*1):
                    stoploss_value = senkou_a + atr*0.5
                elif (tenkan_kijun_diff < atr):
                    stoploss_value = senkou_a - atr*1
            elif not tenkan_kijun and not senkou_a_b:
                if tenkan_senkou_b_diff > atr:
                    stoploss_value = senkou_b
                elif tenkan_senkou_b_diff <= atr:
                    stoploss_value = senkou_a - atr*2
        if not price_tenkan & price_kijun & price_cloud:
            if (kijun_senkou_a_diff > atr*2) and tenkan_kijun_diff > atr:
                stoploss_value = senkou_a + atr*1
            elif (kijun_senkou_a_diff > atr*2) and tenkan_kijun_diff <= atr:
                stoploss_value = max(senkou_a, senkou_b) - atr*2
            elif (kijun_senkou_a_diff < atr*2):
                if senkou_a_b:
                    stoploss_value = max(pprice,price) - atr*5
                else:
                    stoploss_value = max(pprice,price) - atr*4
        
        if price_tenkan and not price_kijun and price_cloud:
            if (tenkan_senkou_a_diff > atr*2) and tenkan_kijun_diff > atr:
                stoploss_value = senkou_a + atr*1
                stoploss_value = max(price, entry_price) - atr*5
            else:
                stoploss_value = price - atr*1
                stoploss_value = max(price, entry_price, pprice) - atr*5
        if not price_kijun and not price_tenkan and price_cloud:
            stoploss_value = max(price, entry_price, pprice) - atr*3.5
        
        if price_incloud:
            if senkou_a_b and (senkou_a_b_diff > atr*1):
                if price_tenkan:
                    stoploss_value = tenkan - atr*1.5
                else:
                    stoploss_value = price - atr*3
            elif senkou_a_b:
                stoploss_value = price - atr*2
            elif not senkou_a_b and (senkou_a_b_diff > atr):
                # if price_tenkan and ((tenkan - senkou_a) > 200):
                #     stoploss_value = senkou_a - atr*0
                if price_tenkan:
                    stoploss_value = senkou_a - atr*2
                else:
                    stoploss_value = senkou_a - atr*1.5

        if price_kijun and price_tenkan and price_cloud1:
            if tenkan_kijun and senkou_a_b:
                stoploss_value = price - atr*5
            if tenkan_kijun and (not senkou_a_b):
                stoploss_value = max(price, entry_price, pprice) - atr*2
            if not tenkan_kijun and senkou_a_b:
                stoploss_value = max(price, pprice) - atr*1.5
            elif not tenkan_kijun and (not senkou_a_b):
                stoploss_value = tenkan - atr*2
        if not price_tenkan and price_cloud1:
            if price_kijun:
                # stoploss_value = price - atr*2
                stoploss_value = kijun - atr*1.5
            else:
                stoploss_value = price - atr*2
        min5 = row['rolling_min_5']
        open_min5 = row['open_min_5']
        if (price_senkou_a_diff > 500) and (not senkou_a_b) and ((price - min5) > 500) and (min5 < senkou_b):
            stoploss_value = price - 400
        if (senkou_a < senkou_b) and (high > senkou_b) and (senkou_a_b_diff > atr*1.5) and ((price-open) > 150):
            stoploss_value = open - 5
        
        if ((price - min5) > 200)and price_cloud1:
            stoploss_value = min5+10
        elif ((price - open_min5) > atr*2) and price_cloud1 and (open_min5 < kijun):
            stoploss_value = open_min5 + 10
        if stoploss_value == 0:
            stoploss_value = price - atr*2
        if stoploss_value < price:
            # print(stoploss_value)
            if stoploss_value > (price - 100):
                return int(stoploss_value) - 80
            return int(stoploss_value)
    return 0

def stopless_point(row, position, entry_price, prow):
    price = row['Close']
    open = row['Open']
    high = row['High']
    pprice = prow['Close']
    tenkan = row['tenkan']
    kijun = row['kijun']
    senkou_a = row['senkou_a']
    senkou_b = row['senkou_b']
    atr = row['ATR']
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    price_tenkan = True if (price > tenkan) else False
    price_kijun = True if (price > kijun) else False
    price_cloud = True if (price > max(senkou_a, senkou_b)) else False
    price_cloud1 = True if (price < min(senkou_a, senkou_b)) else False
    price_incloud = True if (cloud_top >= price >= cloud_bottom) else False
    tenkan_kijun = True if (tenkan >= kijun) else False
    tenkan_senkou_a = True if (tenkan >=senkou_a) else False
    tenkan_senkou_b = True if (tenkan >=senkou_b) else False
    kijun_senkou_a = True if (kijun >=senkou_a) else False
    kijun_senkou_b = True if (kijun >=senkou_b) else False
    senkou_a_b = True if (senkou_a > senkou_b) else False
    price_tenkan_diff = price - tenkan
    price_kijun_diff = price - kijun
    tenkan_kijun_diff = tenkan - kijun
    tenkan_senkou_a_diff = tenkan - senkou_a
    tenkan_senkou_b_diff = tenkan - senkou_b
    kijun_senkou_a_diff = kijun - senkou_a
    kijun_senkou_b_diff = kijun - senkou_b
    senkou_a_b_diff = abs(senkou_a - senkou_b)
    price_senkou_a_diff = price - senkou_a
    multiplier = 3
    stoploss_value = 0
    reverse_sl = 0
    if position == 1:
        if price_tenkan & price_kijun & price_cloud:
            # stoploss_value = 0
            if tenkan_kijun & senkou_a_b:
                if (tenkan_senkou_a_diff > atr*3) and (tenkan_kijun_diff > atr*1):
                    stoploss_value = kijun + atr*0
                elif (tenkan_senkou_a_diff > atr*1) and (tenkan_kijun_diff > atr):
                    stoploss_value = senkou_a + atr*0
                elif (tenkan_kijun_diff < atr):
                    stoploss_value = kijun + atr*2
            elif tenkan_kijun and not senkou_a_b:
                if (tenkan_senkou_a_diff > atr*3) and (tenkan_kijun_diff > atr):
                    stoploss_value = kijun + (tenkan - kijun)/2 + atr*0.2
                elif (tenkan_senkou_a_diff > atr*1) and (tenkan_kijun_diff < atr):
                    stoploss_value = senkou_a - atr*1.5
                elif (tenkan_kijun_diff < atr):
                    stoploss_value = kijun
                else:
                    stoploss_value = senkou_b
            elif not tenkan_kijun and senkou_a_b:
                if (tenkan_senkou_a_diff > atr*3):
                    stoploss_value = tenkan - atr*2
                elif (tenkan_senkou_a_diff > atr*1):
                    stoploss_value = senkou_a + atr*0.5
                elif (tenkan_kijun_diff < atr):
                    stoploss_value = senkou_a - atr*1
            elif not tenkan_kijun and not senkou_a_b:
                if tenkan_senkou_b_diff > atr:
                    stoploss_value = senkou_b
                elif tenkan_senkou_b_diff <= atr:
                    stoploss_value = senkou_a - atr*2
        if not price_tenkan & price_kijun & price_cloud:
            if (kijun_senkou_a_diff > atr*2) and tenkan_kijun_diff > atr:
                stoploss_value = senkou_a + atr*1
            elif (kijun_senkou_a_diff > atr*2) and tenkan_kijun_diff <= atr:
                stoploss_value = max(senkou_a, senkou_b) - atr*2
            elif (kijun_senkou_a_diff < atr*2):
                if senkou_a_b:
                    stoploss_value = max(pprice,price) - atr*5
                else:
                    stoploss_value = max(pprice,price) - atr*4
        
        if price_tenkan and not price_kijun and price_cloud:
            if (tenkan_senkou_a_diff > atr*2) and tenkan_kijun_diff > atr:
                stoploss_value = senkou_a + atr*1
                stoploss_value = max(price, entry_price) - atr*5
            else:
                stoploss_value = price - atr*1
                stoploss_value = max(price, entry_price, pprice) - atr*5
        if not price_kijun and not price_tenkan and price_cloud:
            stoploss_value = max(price, entry_price, pprice) - atr*3.5
        
        if price_incloud:
            if senkou_a_b and (senkou_a_b_diff > atr*1):
                if price_tenkan:
                    stoploss_value = tenkan - atr*1.5
                else:
                    stoploss_value = price - atr*3
            elif senkou_a_b:
                stoploss_value = price - atr*2
            elif not senkou_a_b and (senkou_a_b_diff > atr):
                # if price_tenkan and ((tenkan - senkou_a) > 200):
                #     stoploss_value = senkou_a - atr*0
                if price_tenkan:
                    stoploss_value = senkou_a - atr*2
                else:
                    stoploss_value = senkou_a - atr*1.5

        if price_kijun and price_tenkan and price_cloud1:
            if tenkan_kijun and senkou_a_b:
                stoploss_value = price - atr*5
            if tenkan_kijun and (not senkou_a_b):
                stoploss_value = max(price, entry_price, pprice) - atr*2
            if not tenkan_kijun and senkou_a_b:
                stoploss_value = max(price, pprice) - atr*1.5
            elif not tenkan_kijun and (not senkou_a_b):
                stoploss_value = tenkan - atr*2
        if not price_tenkan and price_cloud1:
            if price_kijun:
                # stoploss_value = price - atr*2
                stoploss_value = kijun - atr*1.5
            else:
                stoploss_value = price - atr*2
        min5 = row['rolling_min_5']
        max5 = row['rolling_max_5']
        pmin5 = prow['rolling_min_5']
        pmax5 = prow['rolling_max_5']
        open_min5 = row['open_min_5']
        if (price_senkou_a_diff > 500) and (not senkou_a_b) and ((price - min5) > 500) and (min5 < senkou_b):
            stoploss_value = price - 400
        if (senkou_a < senkou_b) and (high > senkou_b) and (senkou_a_b_diff > atr*1.5) and ((price-open) > 150):
            stoploss_value = open - 5
        
        if ((price - min5) > 200)and price_cloud1:
            stoploss_value = min5+10
        elif ((price - open_min5) > atr*2) and price_cloud1 and (open_min5 < kijun):
            stoploss_value = open_min5 + 10
            
        if ((min5 - stoploss_value) > 100):
            reverse_sl = stoploss_value - 50
        elif ((min5 - stoploss_value) > 50):
            reverse_sl = stoploss_value - 70
        if stoploss_value == 0:
            stoploss_value = price - atr*2
        if stoploss_value < price:
            # print(stoploss_value)
            if stoploss_value > (price - 100):
                return int(stoploss_value) - 80, (int(reverse_sl) - 120)
            return int(stoploss_value), int(reverse_sl)
    return 0,reverse_sl

def stopless_point_short(row, position):
    price = row['Close']
    open = row['Open']
    high = row['High']
    low = row['Low']
    tenkan = row['tenkan']
    kijun = row['kijun']
    senkou_a = row['senkou_a']
    senkou_b = row['senkou_b']
    atr = row['ATR']
    price_tenkan = True if (price < tenkan) else False
    price_kijun = True if (price < kijun) else False
    tenkan_kijun = True if (tenkan <= kijun) else False
    tenkan_senkou_a = True if (tenkan <=senkou_a) else False
    tenkan_senkou_b = True if (tenkan <=senkou_b) else False
    kijun_senkou_a = True if (kijun <=senkou_a) else False
    kijun_senkou_b = True if (kijun <=senkou_b) else False
    senkou_a_b = True if (senkou_a < senkou_b) else False
    price_tenkan_diff = tenkan - price
    price_kijun_diff = kijun - price
    tenkan_kijun_diff = kijun - tenkan
    tenkan_senkou_a_diff = senkou_a - tenkan
    tenkan_senkou_b_diff = senkou_b - tenkan
    kijun_senkou_a_diff = senkou_a - kijun
    kijun_senkou_b_diff = senkou_b - kijun
    senkou_a_b_diff = abs(senkou_a - senkou_b)
    price_senkou_a_diff = senkou_a - price
    stoploss_value = 0
    reverse_sl = 0
    if position == -1:
        if price_tenkan & price_kijun & tenkan_kijun:
            if kijun_senkou_a & senkou_a_b:
                if (price_kijun_diff > (kijun_senkou_a_diff)):
                    if (price_tenkan_diff > atr*2):
                        stoploss_value = tenkan
                    elif (price_kijun_diff > atr*1):
                        stoploss_value = kijun + atr*0.5
                    else:
                        stoploss_value = senkou_a - atr
                elif (tenkan_kijun_diff > price_tenkan_diff*2):
                    stoploss_value = tenkan + atr*1
                elif (kijun_senkou_a_diff > price_kijun_diff*2):
                    stoploss_value = kijun
            elif kijun_senkou_a and not senkou_a_b:
                if (price_kijun_diff > (kijun_senkou_b_diff)):
                    if (price_kijun_diff > atr*2):
                        stoploss_value = kijun - atr*0.2
                    elif price_tenkan_diff > atr*1.5:
                        stoploss_value = tenkan - atr*0.7
                elif (tenkan_kijun_diff > price_tenkan_diff*2):
                    stoploss_value = tenkan + atr
                elif (kijun_senkou_a_diff > price_kijun_diff*2):
                    if tenkan_kijun_diff > atr*0.5:
                        stoploss_value = kijun - atr*0.8
                    elif price_kijun_diff > atr*1.5 and (tenkan < senkou_b):
                        stoploss_value = kijun - atr*1
                    elif price_kijun_diff > atr*1.5:
                        stoploss_value = kijun - atr*0.8
                    else:
                        stoploss_value = kijun
            elif not kijun_senkou_a:
                if price_tenkan_diff > tenkan_kijun_diff*1.5:
                    stoploss_value = tenkan + atr
        elif price_tenkan and price_kijun and not tenkan_kijun:
            if kijun_senkou_b and (tenkan_senkou_a_diff > price_tenkan_diff*2):
                stoploss_value = max(tenkan, kijun) - atr*0.5
            elif kijun_senkou_b:
                stoploss_value = max(senkou_a, senkou_b)
            elif not kijun_senkou_b:
                stoploss_value = max(tenkan, kijun) + atr*0.9
        
        elif price_tenkan and not price_kijun:
            if tenkan_senkou_a:
                if tenkan_senkou_a_diff > price_tenkan_diff:
                    stoploss_value = tenkan + atr*0.2
                else:
                    stoploss_value = tenkan 
            else:
                stoploss_value = tenkan + atr*0.8
        elif not price_tenkan and price_kijun:
            if price_kijun_diff > atr:
                stoploss_value = kijun + atr*0.5
            elif kijun_senkou_a and senkou_a_b:
                stoploss_value = max(senkou_a, senkou_b) - atr*0.5
            else:
                stoploss_value = max(senkou_a, kijun) + atr
        elif not price_kijun and not price_tenkan:
            if price_senkou_a_diff > 0 and senkou_a_b:
                stoploss_value = min(senkou_a, senkou_b) + atr*0.5
            else:
                stoploss_value = price + atr*2
        # if stoploss_value  (tenkan + atr*3):
            # stoploss_value = tenkan - atr*3
        # if (senkou_a > senkou_b) and (low < senkou_a) and (senkou_a_b_diff > atr*1.5) and ((open - price) > 150):
        #     stoploss_value = open + 5
        max5 = row['rolling_max_5']
        price_cloud1 = True if (price >=max(senkou_a, senkou_b)) else False
        if ((open - price) > 500) and price_tenkan and price_kijun and (open > tenkan) and ((tenkan - price) > 250)and ((kijun - price) > 200):
            stoploss_value = kijun - 40
        if stoploss_value == 0:
            stoploss_value = price + atr*2
        if ((max5 - price) > 350) and price_cloud1:
            stoploss_value = max5 - 100
        # if ((stoploss_value - max5) > 50):
        #     reverse_sl = stoploss_value + 100
        # elif ((min5 - stoploss_value) > 50):
        #     reverse_sl = stoploss_value - 70
        if stoploss_value > price:
            if stoploss_value < (price + 100):
                return int(price) + 80, int(reverse_sl)
            return int(stoploss_value), int(reverse_sl)
    return 0, reverse_sl



def stoploss_exit_price(row, stoploss_val):
    price = row['Close']
    low = row['Low']
    high = row['High']
    if low < stoploss_val:
        return stoploss_val
    return 0
def stoploss_exit_price_short(row, stoploss_val):
    price = row['Close']
    low = row['Low']
    high = row['High']
    if high > stoploss_val:
        return stoploss_val
    return 0


def normalize(df):
    df = df.copy()
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def find(possible_names):
        for name in possible_names:
            lname = name.lower()
            if lname in lower_map:
                return lower_map[lname]
        return None

    # --- datetime column ---
    dt_col = find(['datetime', 'date', 'timestamp', 'time'])
    if dt_col is None:
        raise ValueError("No datetime/date/timestamp column found in CSV.")
    # Rename to 'datetime'
    df = df.rename(columns={dt_col: 'datetime'})

    # --- rename core financial columns ---
    rename_map = {}

    col_open = find(['open'])
    col_high = find(['high'])
    col_low  = find(['low'])
    col_close = find(['close', 'price', 'last'])
    col_vol = find(['volume', 'vol'])
    col_oi  = find(['oi', 'open interest', 'open_interest'])

    if col_open:  rename_map[col_open] = 'Open'
    if col_high:  rename_map[col_high] = 'High'
    if col_low:   rename_map[col_low] = 'Low'
    if col_close: rename_map[col_close] = 'Close'
    if col_vol:   rename_map[col_vol] = 'Volume'
    if col_oi:    rename_map[col_oi] = 'OI'

    df = df.rename(columns=rename_map)

    # ensure required columns exist
    missing = [c for c in ['Open','High','Low','Close'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    # Sort by datetime and set index
    # df = df.sort_values('datetime').set_index('datetime')
    print(df.tail(5))
    return df
def wait_until_next_15min_plus30():
    """Wait until next 15-minute candle + 30 seconds mark."""
    now = datetime.now() + timedelta(hours=5, minutes=30)
    # now = datetime.now()
    print(now)
    # Find the next 15-minute multiple
    min = 15
    next_minute = (now.minute // min + 1) * min
    log(next_minute)
    next_time = now.replace(minute=0, second=30, microsecond=0) + timedelta(minutes=next_minute)
    log(next_time)
    if next_minute >= 60:
        next_time = now.replace(hour=(now.hour + 1) % 24, minute=0, second=30, microsecond=0)
    wait_seconds = (next_time - now).total_seconds()
    log(wait_seconds)
    if wait_seconds < 0:
        wait_seconds += (60*min)  # just in case of rounding errors
    log(f"⏳ Waiting {int(wait_seconds)} sec until next candle time {next_time.strftime('%H:%M:%S')}...")

    time.sleep(wait_seconds)


def log(msg):
    with open("static/logs.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")