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
    df['line_gap_range'] = df['line_gap'].rolling(5).max()
    df['cloud_gap'] = abs(df['senkou_a'] - df['senkou_b'])
    df['cloud_gap_range'] = df['cloud_gap'].rolling(5).max()
    df['candel_size'] = abs(df['Close'] - df['Open'])
    df['price_tenkan_gap'] = abs(df['Close'] - df['tenkan'])
    df['tenkan_kijun_gap'] = abs(df['tenkan'] - df['kijun'])
    df['price_cloud_gap'] = abs(df['Close'] - df[['senkou_a','senkou_b']].max(axis=1))
    df['DI_gap'] = abs(df['+DI'] - df['-DI'])
    df['DI_gap_range'] = df['DI_gap'].rolling(5).max()
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
    df['exit_long_avoid'] = (((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*0.8)))
    df['exit_long_avoid1'] = (((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*0.9))&
                              ((df['rolling_max_10'] - df['rolling_min_10']) > (df['ATR']*0.4))&
                              (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['-DI_move_up'])) &
                            #   (df['RSI'] < 50) &
                              (df['ADX'] > 40) &
                              df['price_move_down'])
    
    
    

    
    
    extra_condition = ((df['Open'] - df['Close']) >= 50).rolling(2).sum() >= 2

    
    


    # Exit Short
    # Exit Short
    # Exit Short
    # Exit Short
    df = df.copy()
    extra_condition1 = ((df['Close'] - df['Open']) >= 20).rolling(2).sum() >= 2
    
    
    df['price_in_lines'] = (
        (df['Close'] > df[['tenkan','kijun']].min(axis=1)) &
        (df['Close'] < df[['tenkan','kijun']].max(axis=1))
    )
    df['price_above_kijun'] = (df['Close'] > df['kijun'])
    df['line_gap_reduce'] = (df['line_gap'] <= df['line_gap'].shift(1))
    df['DI_gap_reduce'] = (df['DI_gap'] < df['DI_gap'].shift(1))

    ###################################
    df['avoid_entry1'] = (
        (df[['tenkan','kijun']].max(axis=1) > df[['senkou_a','senkou_b']].max(axis=1) ) &
        (df[['tenkan','kijun']].min(axis=1) < df[['senkou_a','senkou_b']].min(axis=1) ) &
        df['green_cloud']
    )
    df['avoid_entry_short1'] = (
        (df['tenkan_above_kijun'] & 
        ((df['tenkan'] - df['kijun']) > df['ATR']*2)) |
        (df['+DI_up'])
    )
    df['avoid_entry_short2'] = (
        ((abs(df['tenkan'] - df['kijun']) < df['ATR']*0.1).rolling(3).sum()>=2)
    )
    df['avoid_entry_short3'] = (
        (df['line_gap'] < df['ATR']) &
        ((df['cloud_gap'] < df['ATR']*2) | (df['ADX'] < df['ADX'].shift(1))) &
        df['price_in_lines']
    )
    df['avoid_entry_short4'] = (
        ((df['Open'] - df['Close'])*3 < (df['Close'] - df['Low']))
    )
    df['avoid_entry_short_5'] = (
        ((df['DI_gap'] < 5).rolling(5).sum() >= 3) & 
        ((df['line_gap'] < df['ATR']*0.6).rolling(5).sum() >= 3) & 
        df['tenkan_above_kijun'] &
        df['price_above_cloud']
    )
    df['avoid_entry_short6'] = (
        (df['cloud_gap_range'] < df['ATR']*1.5) &
        ((abs(df['High'] - df['Low']) > abs(df['Close'] - df['Open'])*2).rolling(3).sum() >= 2)
    )
    # df['avoid_entry_short4'] = (
    #     (df['range1'] < df['ATR']) &
    #     (df['cloud_gap'] < df['ATR']*2) &
    #     df['price_in_lines']
    # )
    df['entry_gap_short'] = ((
                                ((df['Close'] < df['Open']).rolling(3).sum() >= 2) & 
                                (((df['Close']+50) < df['Close'].shift(1)).rolling(3).sum() >= 2)&
                                # (df['Close'].shift(1) < df['Open'].shift(1)) & 
                                ((df['Open']+50) < df['Open'].shift(1)) &
                                df['price_below_cloud'] &
                                (df['cloud_gap'] > df['ATR']*0.3) &
                                # (df['Close'] < df['rolling_max_5'] - df['ATR']) &
                                (df['range1'] > df['ATR']*1.5)
                            ) & (~df['avoid_entry_short1']) & 
                            (~df['avoid_entry_short2']) & 
                            (~df['avoid_entry_short6']) & 
                            (~df['avoid_entry_short_5']) & 
                            (~df['avoid_entry_short3']))
    df['entry_gap_short1'] = ((
                                ((df['Close'] < df['Open']).rolling(3).sum() >= 3) & 
                                (((df['Close']+50) < df['Close'].shift(1)).rolling(3).sum() >= 3)&
                                # (df['Close'].shift(1) < df['Open'].shift(1)) & 
                                ((df['Open']+50) < df['Open'].shift(1)) &
                                df['price_below_cloud'] &
                                (df['cloud_gap'] > df['ATR']*0.3) &
                                # (df['Close'] < df['rolling_max_5'] - df['ATR']) &
                                (df['range1'] > df['ATR']*1.5)
                                ) & (~df['avoid_entry_short1']) & 
                                (~df['avoid_entry_short2']) & 
                                (~df['avoid_entry_short3']))
    
    df['entry_short_price_drop'] = ((((df['line_gap'] <= df['line_gap'].shift(1)).rolling(3).sum() >= 2) &
                                  (((df['Close']+20) < df['Close'].shift(1)).rolling(3).sum() >= 2) &
                                  (((df['Close']-0) < df['tenkan'].shift(0)).rolling(3).sum() >= 2) &
                                  (((df['Close']+0) < df['Open'].shift(0)).rolling(3).sum() >= 3) &
                                #   (df['range1'] > df['ATR']*0.5) &
                                  (df['Close'] < (df['rolling_max_10'].shift(1) - df['ATR']*1.5)) &
                                  (df['price_above_cloud'])) & (~df['avoid_entry_short1']) & (~df['avoid_entry_short2']) & (~df['avoid_entry_short3']) &
                            (~df['avoid_entry1']) & 
                            (~df['avoid_entry_short3']))
    
    df['entry_short_oi_vol'] = (
                                (df['OI'] > df['oi_ma']) &
                                (df['Volume'] > df['vol_ma']) &
                                df['tenkan_below_kijun'] &
                                (df['Close'] < df['tenkan'] )&
                                df['price_move_down']  &
                                (((df['Close']+50) < df['Open']).rolling(3).sum() >= 2) &
                                (~df['avoid_entry1']))& (~df['avoid_entry_short2']) & (~df['avoid_entry_short3']) & (~df['avoid_entry_short4'])
    
    df['trailing_entry_short0'] = np.where(df['price_above_cloud'],
                                (
                                    (df['Close'] > df['rolling_min_60']* 1.0+df['ATR']) &
                                    (df['Close'] < (df['rolling_max_5']-200)) &
                                    ((df['Open']-50) < df['Open'].shift(1)) &
                                    ((df['Close']-10) > df['Open']) &
                                    ((df['Close']-150) < df['Open']) &
                                    (df['ADX'] < 50)),
                        
                                    # ((df['Close'] < df['trailing_entry_short']* 0.96) &
                                    (
                                    (df['Close'] > df['rolling_min_60']* 1.0 + df['ATR']*1.3) &
                                    (df['Close'] <= df['rolling_max_10'] - df['ATR']*0.1) &
                                    ((df['Open']+150) > df['Open'].shift(1)) &
                                    ((df['Open']-150) < df['Open'].shift(1)) &
                                    (((df['Close']-50) > df['Open'].shift(0))) &
                                    (((df['Close']-200) < df['Open'].shift(0))) &
                                    (df['-DI'] > 40) &
                                    # (df['-DI_move_up']) &
                                    (((df['Close'] + 50) < df['tenkan']) | ((df['Close'] - 80) > df['tenkan'])) &
                                    (df['Close'] < df['kijun']) &
                                    (df['RSI'] < 50) &
                                    (df['ADX'] > 25) &
                                    (df['OI'] > (df['oi_ma'])*0.9) )
                                ) & (~df['avoid_entry_short1'])& (~df['avoid_entry_short2']) & (~df['avoid_entry_short3']) & (~df['avoid_entry1'])
    
    df['trailing_entry_short1'] = np.where(df['price_above_cloud'],
                                (
                                    ((df['Close']-0) < (df['rolling_max_10']-df['ATR']*1)) &
                                    (((df['Close']) < df['kijun']).rolling(2).sum() >= 2 )&
                                    ((df['Open']-10) < df['Open'].shift(1)) &
                                    (df['ADX'] > df['ADX'].shift(1)*0.92) &
                                    (df['+DI'] < 20) &
                                    # df['-DI_move_up'] &
                                    ((df['Close']+100) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.07))
                                ),
                                (
                                    (((df['Close'])< df['kijun']).rolling(3).sum() >= 2 )&
                                    ((df['Open']-10) < df['Open'].shift(1)) &
                                    (df['ADX'] > df['ADX'].shift(1)*0.92) &
                                    (df['-DI'] > 30) &
                                    df['-DI_up'] &
                                    (((df['Close']+0) < df['Open']).rolling(3).sum() >= 2)& 
                                    (df['OI'] < (df['oi_ma']*1.1))
                                )) & (~df['avoid_entry_short1']) & (~df['avoid_entry_short3'])& (~df['avoid_entry1']) & (~df['avoid_entry_short4'])
    
    df['entry_short_above_tk'] = (
        df['price_above_tk'].shift(2) &
        (((df['Close'] + 10) < df['Open']).rolling(2).sum() >= 2) &
        (df['red_cloud'] &
         (df['cloud_gap'] > df['cloud_gap'].shift(1)) &
         df['price_below_cloud'] &
         (df['tenkan_below_kijun']) &
          (df['-DI'] > 40) &
         ((df['Close'] - 50) < df[['tenkan','kijun']].max(axis=1)) &
         (df['Close'] < df['rolling_min_5'].shift(1) - df['ATR']*0.2) &
         (df['candel_size'] > 100) &
         (df['price_move_down']) ) |
        
        ((df['green_cloud']) &
         df['price_below_cloud'] &
        (df['range1'] > df['ATR']*2) &
         (df['candel_size'] > 100) &
          (df['-DI'] > 40) &
         (df['price_tenkan_gap'] < df['price_tenkan_gap'].shift(1)) &
         ((((df['Close'] + 100) < df[['tenkan','kijun']].max(axis=1)) & (df['RSI'] > df['rsi_ma']) ) ) 
         
        ) |
        (
         ((df['Close'] + 50) < df[['tenkan','kijun']].min(axis=1)) &
         ((df['Open'] - 20) > df[['tenkan','kijun']].min(axis=1)) &
         ((df['Close'] < df['Open']).rolling(3).sum() >= 2) &
         (df['tenkan_below_kijun']) &
         (df['candel_size'] > 100) &
         (df['price_move_down']) &
         (df['RSI'] < df['rsi_ma']) &
         (df['Close'] <= df['rolling_min_5'].shift(1) + df['ATR']*0.5) &
          (df['-DI'] > 40) &
         (df['ADX'] < 35) &
         (df['-DI_up']) 
        ) |
        (
          (df['cloud_gap'] > df['ATR']*0.8) &
          (df['-DI_up']) &
          ((df['Close'] + 100) < df[['tenkan','kijun']].min(axis=1)) &
          (df['Close'] < df['rolling_min_5'].shift(1) -df['ATR']*0.2) &
          (df['price_move_down'] & ((df['Close'] + 100) < df['Open'])) &
          (((df['High'] - df['Open']) >(df['Open'] - df['Low'])*3).rolling(2).sum() >= 1) &
          (df['RSI'] < df['rsi_ma']) &
          (df['-DI'] > 35) &
          (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['-DI_move_up'])) 
        )
    ) & (~df['avoid_entry_short1']) & (~df['avoid_entry_short3'])
    
    df['exit_long_tkcross'] = (
                                
                                (((df['Close'].shift(1) > df['tenkan']) | (df['Open'].shift(1) > df['kijun'])) &
                                (df['Open'] > (df['Close']+200)) &
                                ((df['Close']+50) < df['tenkan']) &
                                (df['-DI'] > 27) &
                                (df['DI_gap'] > 11) &
                                ((df['Close'] + 100) < df['kijun']) &
                                (df['Close'] <= (df['rolling_min_10']+df['ATR']*1)) &
                                (df['Volume'] < df['vol_ma']*1) ) | 
                                (
                                df['price_above_cloud'] &
                                (df['-DI'] > 23) &
                                (~(df['avoid_entry_short_5'])) &
                                (df['tenkan_below_kijun'] | df['-DI_up']) &
                                (df['Close'] <= (df['rolling_min_10']+df['ATR']*0.2)) &
                                ((df['line_gap'] <= df['line_gap'].shift(1)).rolling(2).sum() == 2) &
                                ((df['Close']+100) < df['tenkan']) &
                                ((df['Close']+0) < df['kijun']) &
                                (df['Open'] > (df['Close']+100)) &
                                (((df['Close'].shift(1)+50) > df['tenkan']) | ((df['Open'].shift(1)+50) > df['kijun'].shift(1))) 
                                )
                                ) & (~df['avoid_entry_short1'])& (~df['avoid_entry_short2']) & (~df['avoid_entry_short6']) & (~df['avoid_entry_short3'])& (~df['avoid_entry1'])
    df['exit_long_tkcross1'] = ((((df['Close'].shift(1) > df['tenkan']) ) &
                                (df['Open'] > (df['Close']+200)) &
                                # ((df['Open'] > (df['Close']+100)).rolling(3).sum() >= 2) &
                                ((df['Close']+50) < df['tenkan']) &
                                (df['-DI'] > 32) &
                                (df['+DI'] < 18) &
                                (df['DI_gap'] > 15) &
                                ((df['Close'] + 100) < df['kijun']) &
                                (df['Close'] <= (df['rolling_min_10']+df['ATR']*1)) &
                                (df['Volume'] < df['vol_ma']*1) ) ) & (~df['avoid_entry_short1'])& (~df['avoid_entry_short2']) & (~df['avoid_entry_short3'])& (~df['avoid_entry1'])
    
    df['exit_long_price_kijun'] = (
        
        ((df['Close'] < df['kijun']).rolling(2).sum() >= 2) &
        (df['price_move_down']) &
        (((df['Close']+50) < df['Open']).rolling(2).sum() >= 2) &
        (df['+DI_move_down']) &
        (df['Close'] < df['rolling_max_10'] - df['ATR']*1) &
        
        (df['tenkan_above_kijun'] &
         (df['line_gap'] < df['line_gap'].shift(1) ) &
         (df['range1'] > df['ATR']*1.5) ) |
        
        (df['tenkan_below_kijun'] &
         df['price_above_cloud'] &
         (df['price_above_tk'].shift(3)) &
         ((df['line_gap'] < df['line_gap'].shift(1)) | (df['line_gap'] < df['ATR'])) &
         (df['Close'] <= df['rolling_min_10'] + df['ATR']*0.5) &
         (df['ADX'] < df['ADX'].shift(1)) &
         (df['range1'].shift(1) < df['ATR']) &
         (df['Close'] < df['kijun'])
        ) 
    ) & (~df['avoid_entry_short1'])& (~df['avoid_entry_short2']) & (~df['avoid_entry_short3'])

    df['entry_short_price_bottom'] = (
        ((df['Close'] < df['kijun']).rolling(3).sum()==3) &
        df['tenkan_below_kijun'] &
        df['-DI_up'] &
        (df['Close'] <= df['rolling_min_5']) &
        (df['RSI'] < 50) &
        (df['red_cloud'] | 
         (df['green_cloud'] & (df['cloud_gap'] < df['ATR']*2) & 
          (df['cloud_gap'] < df['cloud_gap'].shift(1)))) &
        (df['OI'] > df['oi_ma']*1.03) &
        (((df['Close']-50) < df['Open']).rolling(3).sum() >= 2) &
        (df['candel_size'] > 80)
    ) & (~df['avoid_entry_short1'])& (~df['avoid_entry_short2']) & (~df['avoid_entry_short3'])
    
    df['exit_long_price_tk_cross3'] = (
        (df['ADX'] < 30) &
        df['price_below_tk'] &
        (df['+DI_move_down']) &
        (df['Close'] < (df['rolling_min_10']+df['ATR']*0.2)) &
        (df['range1'] > df['ATR']*1.5) &
        (((df['Close'].shift(1)-20) > df[['kijun','tenkan']].min(axis=1)).rolling(2).sum() >= 1) &
        df['tenkan_below_kijun'] &
        ((df['Close'] + 50) < df['tenkan']) &
        ((df['RSI'] < df['rsi_ma']).rolling(4).sum() >= 2) &
        df['price_move_down'] & 
        (((df['Close'] - 10) < df['Open']).rolling(3).sum()>=2)
    ) & (~df['avoid_entry_short1']) & (~df['avoid_entry_short3'])
    
    df['exit_long_price_tenkan'] = (
        df['price_move_down'] &
        (((df['Close'] - 100) < df['tenkan']).rolling(2).sum() >= 2) &
        ((df['RSI'] < df['rsi_ma']*1.2).rolling(2).sum() >= 2) &
        (((df['Close'] + 20) < df['Open']).rolling(2).sum() >= 2) &
        (df['Close'] < df['rolling_max_5'] - df['ATR']*1) &
        (df['-DI_up'] | ((df['DI_gap'] < df['DI_gap'].shift(1)) & df['+DI_move_down'])) &
        
        ((df['tenkan_kijun_gap'] > df['ATR']*2.5) &
         (df['price_above_cloud']) &
         (df['-DI'] > 25) &
         (df['RSI'] < df['rsi_ma']) &
         (df['Close'] < df['rolling_max_5'] - df['ATR']*2) &
         (df['price_above_tk'].shift(3)) &
         (df['price_cloud_gap'] > df['ATR']*3)) |
        
        ((((df['Close'] + 200) < df['tenkan']).rolling(2).sum() >= 1) &
         (df['price_above_cloud']) &
         (df['-DI'] > 37) &
         (df['+DI'] < 30) &
         (df['Close'] < df['rolling_max_5'] - df['ATR']*1.5) &
         (df['price_above_tk'].shift(3)) &
         (df['line_gap'] < df['line_gap'].shift(1)) &
         (df['range1'] > df['ATR']*1.5) &
         (df['price_cloud_gap'] > df['ATR']*2) &
         ((df['ADX'] < df['ADX'].shift(1)).rolling(2).sum() == 2))
    ) & (~df['avoid_entry_short3'])
    
    df['entry_short_price_top'] = (
        df['price_above_tk'] &
        df['tenkan_above_kijun'] &
        (df['price_cloud_gap'] > df['ATR']*2) &
        ((df['ADX'] < df['ADX'].shift(1)).rolling(2).sum() >= 2) &
        ((df['Close'] - 50) < df['tenkan']) &
        ((df['Close'] + 200) < df['Open']) 
    ) & (~df['avoid_entry_short1']) & (~df['avoid_entry_short3'])

    df['exit_long_cloud_exit'] = (
        df['red_cloud'] &
        ((df['Close'].shift(1) + 20)> df['senkou_a']) &
        (df['Close'] < df['senkou_a']) &
        (df['+DI_move_down']) &
        df['DI_gap_reduce'] &
        (df['ADX'] < 35) &
        (df['Close'] < df['rolling_max_5'] - df['ATR'])
    ) & (~df['avoid_entry_short3'])
    ###################################
    df['exit_short_indicators'] = (
        df['price_above_tk'] &
        df['+DI_up'] &
        (df['-DI'] < 20) &
        (df['+DI'] > 28) &
        (df['Close'] > df['Open']) &
        df['price_above_cloud']
    )
    df['entry_long_below_cloud'] = (df['price_below_cloud'] &
                                    (df['Close'] > (df['rolling_min_10'].shift(1)+df['ATR']*0.8)) &
                                    ((df['Close']+250) > df['tenkan']) & 
                                    (df['Close'] > df['Close'].shift(1)) & 
                                    ((df['Close']-250) > df['Open']))
    
    df['exit_short_avoid'] = (
                              (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1)) &
                              ((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*1)))
    
    df['exit_short_price_cloud2'] = (
                                    (((df['Close']-50) < df['Open']).rolling(3).sum() >= 3) & 
                                    (((df['Close']+100) > df['Open']).rolling(3).sum() >= 1) & 
                                    (df['RSI'] < df['rsi_ma']) &
                                    (df['RSI'] > 28) & 
                                    df['price_below_tk'] &
                                    ((df['DI_gap'] < df['DI_gap'].shift(1)*1.15)) &
                                    
                                    (((df['High'] <= cloud_top) & df['tenkan_above_kijun'] & ((df['Close'] - 100) < df['Open'])) | 
                                     ((df['Close'] <= cloud_top) & (df['tenkan_below_kijun']) ))
                                    )
    
    df['exit_short_above_cloud'] = ((df['Close'] >= df[['senkou_a','senkou_b']].max(axis=1)) &
                                    ((df['Close']-100) > df['Close'].shift(1)) &
                                    (((df['Close']).shift(1) + 200) > df['Open'].shift(1)) &
                                    (((df['Close']) > df['tenkan']).rolling(1).sum() == 1) &
                                    (df['RSI'] < 70) &
                                    (df['RSI'] > 30) &
                                    (df['-DI'] < 39)
                                    ) 
    
    df['exit_short_tenkan'] = (((df['Close'] > df['Open']).rolling(3).sum() >= 3) &
                               (((df['Close']-50) > (df['tenkan'])).rolling(3).sum() >= 2))
    
    df['exit_short_cloud_enter'] = (
        (((df['Close'].shift(1)-10) < df[['senkou_a','senkou_b']].min(axis=1)).rolling(2).sum() >=1) &
        ((df['Close']+20) > df[['senkou_a','senkou_b']].min(axis=1)) &
        
        (df['green_cloud'] &
        ((df['Close'] + 10) > df['senkou_b']) &
        (df['price_move_up']) &
        (df['candel_size'] > 150)) 
        
        # (df['green_cloud'] &
        #  (df['price_green_candle']) &
        #  (((df['Close'] - 20) > df['Open']).rolling(3).sum() >= 2) &
        #  (df['price_move_up']) &
        #  (df['candel_size'] > 100) &
        #  (df['RSI'] > df['rsi_ma']*0.95) &
        #  (df['Volume'] > df['vol_ma']) &
        #  (df['cloud_gap'] > df['ATR']) &
        #  ((df['Close'] + 50) > df['senkou_b']) &
        #  ((df['Close'] + 50) > df['tenkan'])) |
         
        #  (df['red_cloud'] &
        #  ((df['Close'] - 0) > df['senkou_a']) &
        #  (df['cloud_gap'] > df['ATR']) &
        #  (df['range1'] > df['ATR']*1) &
        #  (df['candel_size'] > 100) &
        # #  (df['RSI'] > df['rsi_ma']) &
        #  (df['Close'] > df['rolling_max_10'].shift(1) - df['ATR']) &
        # #  (df['price_move_up']) &
        #  entry_condition
        # ) |
        # ((df['cloud_gap'].shift(1) < df['ATR']*1.5) &
        #  (df['candel_size'] > 50) &
        #  (df['price_move_up']) &
        #  ((df['Close']) > df['tenkan']) &
        #  (df['RSI'] > df['rsi_ma']) &
        #  (df['+DI_move_up']) &
        #  (df['Volume'] > df['vol_ma']*0.8) &
        #  (df['Close'] >= (df['rolling_max_5'].shift(1) + df['ATR']*0.1)) 
        # ) 
    )

    df['exit_short_price_cloud'] = ((((df['Close']+0) < df['Open']).rolling(4).sum() == 4) & 
                                    (df['Close'] < (df['rolling_max_5'] - df['ATR'])) &
                                    (df['RSI'] < df['rsi_ma']) &
                                    (df['RSI'] < 30) &
                                    (df['ADX'] > 25) &
                                    ((df['Open'] - df['Low']) > (df['High'] - df['Open'] - 0)) &
                                    (((df['tenkan'] + 400) < df['tenkan'].shift(1)) |
                                    ((df['tenkan'] + 300) > df['tenkan'].shift(1))) &
                                    (df['-DI'] > 40) &
                                    (df['+DI'] < 10) &
                                    (df['High'] <= cloud_top))
    
    df['exit_short_tenkan_cross'] = (
        # df['price_move_up'] &
        # df['price_above_tenkan'] &
        # (((df['Close'] - 0) > df['Open']).rolling(2).sum() >= 2) &
        # (df['Close'] > df['rolling_min_5'] + df['ATR']*0.3) &
        # df['-DI_move_down'] &
        
        # ((((df['Close'] - 20) > df['tenkan']).rolling(2).sum() >= 2) &
        #  (df['price_below_cloud']) &
        #  df['price_below_tk'].shift(2)) |
        
        (df['tenkan_below_kijun'] &
         (df['tenkan_kijun_gap'] > df['ATR']*2) &
         ((df['Close'] + 10) > df['tenkan']) &
         df['DI_gap_reduce'] &
         (df['-DI'] < 38) &
         (df['+DI'] > 10) &
         (df['RSI'] > 35) &
         df['price_move_up'] &
         (df['Close'] > df['rolling_min_5'] + df['ATR']*0.) &
         df['price_below_tk'].shift(1)) |
        
        # (df['tenkan_above_kijun'] &
        #  (df['+DI_up']) &
        #  (df['price_below_tenkan'].shift(2)) &
        #  ((df['Close'] + 150) > df['tenkan']) &
        #  ((df['candel_size'] > 100).rolling(2).sum() >= 2)&
        #  (((df['Close'] > df['Open']).rolling(2).sum() == 2) |
        #  df['+DI_move_up']) &
        #  ((df['RSI'] > df['RSI'].shift(1)).rolling(2).sum()>= 1)) |
        
        (df['price_in_cloud'] &
         (df['price_above_tenkan']) &
         df['green_cloud'] &
         (df['-DI'] < 35) &
         (df['+DI'] > 20) &
         (df['RSI'] > 40) &
         (df['cloud_gap'] < df['cloud_gap'].shift(1)*1) &
         (df['Close'] > df['rolling_min_5'] + df['ATR']) &
         (df['DI_gap'] < df['DI_gap'].shift(1)*1.1) &
         (df['High'] > df['kijun']) &
         (df['candel_size'] > 200))
        
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
        )
    
    df['exit_short_rsi_di'] = (
        (df['RSI'] <= 22) &
        (df['-DI'] > 42) &
        (df['+DI'] < 9) &
        (df['OI'] < df['OI'].shift(1)) &
        (df['ADX'] > 35) &
        (df['ADX'] > df['ADX'].shift(1)*1.01) &
        (df['Close'] <= df['rolling_min_60'])
    )
    df['exit_short_price_tk_cross1'] = (
        df['price_above_tk'] & df['price_below_tk'].shift(1) &
        df['+DI_up'] &
        (df['price_below_cloud'] &
         df['tenkan_above_kijun'] &
        # (df['line_gap']  df['ATR'] * 1.5) &
        (df['candel_size'] > 200) ) 
    )

    df['exit_short_candle_green'] = (
        ((((df['Close']-100) > df['Open']).rolling(3).sum()>=2) &
        (df['price_above_cloud']) &
        df['tenkan_above_kijun'] &
        df['green_cloud'] &
        (df['+DI'] > 27) &
        (df['-DI'] < 44) &
        (df['RSI'] > 40) &
        (df['DI_gap'] < 10) &
        (df['-DI_move_down']))
    )
    ###################################
    
    
    
    
    
    
    df['avoid_entry_long1'] = (
        (df['Close'] <  df[['senkou_a','senkou_b']].max(axis=1)) &
        (df['Close'] >  df[['tenkan','kijun']].min(axis=1)) &
        (abs(df[['senkou_a','senkou_b']].min(axis=1) - df[['tenkan','kijun']].max(axis=1)) < df['ATR']*0.5)
    )
    df['avoid_entry_long2'] = (
        df['price_in_lines'] &
        (df['Close'] < df['kijun']) &
        (df['cloud_gap'] < df['ATR']*1.5) &
        df['green_cloud'] &
        ((df[['senkou_a','senkou_b']].max(axis=1) > df['tenkan']))
    )
    df['avoid_entry_long3'] = (
        df['red_cloud'] &
        # (df['cloud_gap'] < df['ATR']*0.2) &
        (df['cloud_gap'] < 50) &
        df['price_above_cloud'] &
        (df[['senkou_a','senkou_b']].min(axis=1) > df[['tenkan','kijun']].min(axis=1))
    )
    df['avoid_entry_long4'] = (
        ((df['Close'] < df['Open']).rolling(4).sum()>=4)
    )
    df['avoid_entry_long4'] = (
        # df['price_in_cloud'] &
        (df['line_gap'] < df['ATR']*0.5) &
        df['-DI_up']
    )
    df['avoid_entry_long5'] = (
        df['-DI_up'] &
        df['price_below_cloud'] &
        df['green_cloud'] &
        df['tenkan_below_kijun'] 
    )
    df['avoid_entry_long6'] = (
        df['-DI_up'] &
        (df['DI_gap'] > 15) &
        df['price_below_cloud'] &
        df['red_cloud'] &
        (df['cloud_gap'] > df['cloud_gap'].shift(1)) &
        df['tenkan_below_kijun'] 
    )
    df['avoid_entry_long7'] = (
        (df['-DI_up'].rolling(4).sum() >= 2) &
        ((df['DI_gap'] < 4).rolling(4).sum() >= 3) &
        ((df['kijun']- df['ATR']) < df['senkou_a'])
    )
    df['avoid_entry_long8'] = (
        ((df['line_gap'] < df['ATR']).rolling(10).sum() >= 5) & 
        (df['kijun'] < df[['senkou_a','senkou_b']].max(axis=1)) &
        (df['kijun'] > df[['senkou_a','senkou_b']].min(axis=1)) &
        df['price_above_cloud']
    )
    df['avoid_entry_long9'] = (
        (df['+DI_move_down'].rolling(2).sum()==2) &
        (df['cloud_gap'] < df['ATR']) &
        ((abs(df['High'] - df['Low']) > abs(df['Close'] - df['Open'])*3.5).rolling(3).sum() >= 2)
    )
    ##################

    df['entry_long_indicators'] = (
        (df['+DI'] > df['-DI']) &
        (~(df['RSI'] < df['rsi_ma']*0.92)) &
        (((df['Close']+ 100) > df['Open']).rolling(2).sum()>= 1) &
        ((df['tenkan'] + df['ATR']*0.1) > df['kijun']) &
        (df['tenkan'] >= df['kijun']) &
        (df['Close'] > df['kijun']) &
        (((df['Close']) > df['Open']) | ((df['Low'] + 250)> df['Open'])) &
        (~df['avoid_entry_long1']) &
        (~df['avoid_entry_long3']) &
        (~df['avoid_entry_long4']) &
        (~df['avoid_entry_long7']) &
        (~df['avoid_entry_long8']) &
        (~df['avoid_entry_long9'])
    )
    df['entry_long_indicators0'] = (
        (df['+DI'] > df['-DI']) &
        # df['tenkan_above_kijun'] &
        ((df['tenkan'] + df['ATR']*0.1) > df['kijun']) &
        (df['tenkan'] >= df['kijun']) &
        (df['Close'] > df['kijun']) &
        ((df['Close'] > df['Open']).rolling(3).sum()>=2) &
        ((df['Close'] + 100) > df['tenkan']) &
        (~df['avoid_entry_long1']) &
        (~df['avoid_entry_long3']) &
        (~df['avoid_entry_long4']) &
        (~df['avoid_entry_long7']) &
        (~df['avoid_entry_long8'])
    )
    df['entry_long_indicators1'] = (
        ((df['DI_gap'] < df['DI_gap'].shift(1)).rolling(3).sum() == 3) &
        ((df['Close'] - df['ATR']) > df['tenkan']) &
        # (df['ADX'] > 15) &
        (df['Close'] >= df['rolling_max_5']) & (~ df['avoid_entry_long2']) &
        (~df['avoid_entry_long3']) & 
        (~df['avoid_entry_short2'])& 
        (~df['avoid_entry_short3']) &
        (~df['avoid_entry_long4']) &
        (~df['avoid_entry_long6'])
    )
    df['entry_long_price_move'] = (
        ((df['Close'] > df['Open']).rolling(5).sum() >= 4) &
        (df['price_above_tenkan'].rolling(3).sum() >= 3) &
        ((df['Close'] - 100) <  df[['senkou_a','senkou_b']].min(axis=1)) &
        df['+DI_up'] &
        ((df['+DI']-5) > df['-DI']) &
        # ((df['+DI']-2) > df['+DI'].shift(1)) &
        (df['RSI'] > 45) &
        (~df['avoid_entry_long4']) &
        (~df['avoid_entry_long5']) &
        (~df['avoid_entry_long6']) &
        (~df['avoid_entry_long8'])
    )
    df['entry_long_cloud_cross'] = (
        (df['Close'].shift(1) < df['senkou_a']) &
        (df['Close'].shift(1) < df['kijun']) &
        (df['Close'] > df['senkou_a']) &
        df['price_above_tk'] &
        (df['+DI_up'] | (df['DI_gap_reduce'])) &
        (((df['Close'] - df['ATR']*0.5) > df['Open']).rolling(2).sum()==2) &
        (~df['avoid_entry_long7'])
        # (~df['avoid_entry_long5']) &
        # (~df['avoid_entry_long6']) &
        # (~df['avoid_entry_long8'])

    )
    
    ################################
    df['avoid_exit_long'] = (
        (((df['Close']+100) > df['Open']).rolling(5).sum()>= 3) &
        ((df['Close'] > df['rolling_min_5']) | (df['price_above_tenkan'].rolling(10).sum()>= 10))&
        (df['price_above_tenkan'].rolling(3).sum()>= 3)
    )
    df['avoid_exit_long1'] = (
        (df['price_above_tenkan'].rolling(5).sum()>= 5)
    )
    df['avoid_exit_long2'] = (
        df['-DI_move_down'] &
        (df['+DI'] > 15) &
        df['green_cloud'] &
        ((df['Close'] - 100) > df['Open']) &
        ((df['Close'] > df['Open']).rolling(2).sum()==2) &
        (df['Close'] >= df['rolling_max_10']) &
        ((df['Close']-150) > df['tenkan']) 
    )
    df['avoid_exit_long3'] = (
        ((df['Close'] > df['Open']).rolling(4).sum() == 4) &
        (df['candel_size'] > 100) &
        df['price_above_cloud']
    )
    df['avoid_exit_long4'] = (
        (df['range2'] < df['ATR']*1) &
        (df['cloud_gap'] > df['ATR']*2) &
        df['green_cloud'] &
        ((df['DI_gap'] < 3).rolling(4).sum()==4) &
        df['price_above_cloud']
    )
    df['avoid_exit_long5'] = (
        df['tenkan_below_kijun'] &
        df['line_gap_reduce'] &
        (df['line_gap'] < df['ATR']*0.2) &
        (df['Close'] > df['senkou_b']) &
        (df['+DI'] > 20)
    )
    ######
    df['exit_long_move_down'] = (
        (((df['Low'] + 200) < df['High']).rolling(3).sum()==3) &
        (((df['Close']) < df['Open']).rolling(3).sum()>=2) &
        (df['candel_size'] > 100) &
        (df['-DI_up'] | (df['DI_gap'] < 5)) &
        (((df['High'] - df['Open']) > (df['Open'] - df['Close'])*2).rolling(3).sum()>=2) &
        (((df['High'] - df['Open']) > (df['Open'] - df['Low'])).rolling(3).sum()>=2) &
        (df['Close'] <= df['rolling_min_5']+20) &
        df['DI_gap_reduce'] &
        df['+DI_move_down']

    )
    df['entry_exit_indicator1'] = (
        (df['Close'] < df['kijun']) &
        df['price_above_cloud'] &
        ((df['+DI'] -1)< df['-DI']) & (~df['avoid_exit_long'])  & (~df['avoid_exit_long4']) 
    )
    df['entry_exit_indicator11'] = (
        (df['Close'] < df['kijun']) &
        df['tenkan_above_kijun'] &
        (df['+DI'] <= df['-DI']) & (~df['avoid_exit_long'])  & (~df['avoid_exit_long4']) 
    )
    df['entry_exit_indicator2'] = (
        (df['tenkan'] < df['kijun']) &
        (df['+DI'] < df['-DI']) & (~df['avoid_exit_long']) & 
        (~df['avoid_exit_long2']) & (~df['avoid_exit_long4'])  & 
        (~df['avoid_exit_long5']) 
    )
    df['entry_exit_indicator21'] = (
        (df['tenkan'] < df['kijun']) &
        ((df['-DI_up']) ) & (~df['avoid_exit_long']) & (~df['avoid_exit_long4'])  & 
        (~df['avoid_exit_long5']) 
    )
    df['entry_exit_indicator22'] = (
        (df['Close'] < df['tenkan']) &
        (df['Low'] < df['kijun']) &
        (df['ADX'] < df['ADX'].shift(2)) &
        (((df['Close']+ 50) < df['Open']).rolling(3).sum()==3) &
        (((df['Low']+ 100) < df['High']).rolling(3).sum()==3) &
        ((df['-DI_up']) )  & (~df['avoid_exit_long4']) 
    )
    df['entry_exit_indicator3'] = (
        ((df['Close']) < df['tenkan']) &
        ((df['Low'] - 0) < df['kijun']) &
        (df['+DI'] < df['-DI']) &
        (df['ADX'] < 20) &
        (df['candel_size'] > 250) &
        (df['Close'] < df['Open']) &
        df['price_below_cloud'] &
        df['red_cloud'] & (~df['avoid_exit_long4'])  
    )
    df['exit_long_gap_down'] = (
        (((df['line_gap'] < df['line_gap'].shift(1)).rolling(2).sum()>= 2) &
        ((df['DI_gap'] < df['DI_gap'].shift(1)).rolling(2).sum()>= 2) &
        (df['Close'] < df['tenkan']) &
        (df['red_cloud']) &
        ((df['Close'] < df['Open']).rolling(4).sum()>= 4) &
        (df['ADX'] < df['ADX'].shift(1))) |
        
        (df['-DI_up'] &
         (df['price_below_tenkan'].rolling(3).sum() == 3) &
         ((df['Close'] < df['Open']).rolling(3).sum() == 3) &
         (df['Close'] <= df['rolling_min_10']) &
         (df['Close'] <= df['rolling_max_5'] - df['ATR']*2) &
         (df['ADX'] < 30) &
         ((df['tenkan'] < df['tenkan'].shift(1))) &
         ((df['ADX'] < df['ADX'].shift(1))))
    ) & (~df['avoid_exit_long4'])  & (~df['avoid_exit_long5']) 
    df['exit_long_below_tenkan_DI'] = (
        (df['-DI_up'] &
         (df['DI_gap'] > 5) &
         ((df['Close'] < df['Open']).rolling(3).sum() >= 2) &
         (df['Close'] < df['rolling_min_5']+50) &
         (df['ADX'] < 12) &
         ((df['Close']+50) < df['tenkan'])
         ) |
        ((df['+DI'] < 30) &
         df['price_below_tk'] &
         (df['Close'] < df['senkou_b']) &
         df['red_cloud'] &
         (df['Close'] <= df['rolling_min_5']) &
         (df['+DI_move_down']) &
         df['price_move_down'])
    ) & (~df['avoid_exit_long4']) 
    df['exit_long_below_tk'] = (
        (df['Close'] < df['tenkan']) &
        (df['Close'] < df['kijun']) &
        (df['line_gap_reduce'].rolling(2).sum() >=2) &
        (df['DI_gap_reduce'].rolling(2).sum() >=2) &
        (df['ADX'] < 20)
    ) & (~df['avoid_exit_long4']) 
    df['exit_long_price_drop'] = (
        ((df['Close'] < df['Open']).rolling(3).sum()>=2) &
        (df['Close'] <= df['rolling_min_5']) &
        (df['Close'] <= (df['rolling_max_5'] - df['ATR']*2)) &
        (df['Low'] < df['kijun']) &
        df['price_below_tenkan'] &
        df['+DI_move_down'] &
        df['-DI_move_up'] &
        df['-DI_up'] &
        (df['Open'] > df['tenkan']) & 
        (~df['avoid_exit_long1']) & 
        (~df['avoid_exit_long2']) & 
        (~df['avoid_exit_long5']) 
    )
    df = df.copy()
    #### df['exit_short_cloud_tenkan'] | df['entry_kumo_break_long'] | df['entry_long'] | df['entry_pullback_long'] | df['trailing_entry_long2'] | 
    
    
    df['final_entry_long'] = df['entry_long_indicators'] | df['entry_long_indicators0'] | df['entry_long_indicators1'] | df['entry_long_price_move'] | df['entry_long_cloud_cross']
    # df['final_entry_long'] = False
    
    
    
    df['exit_long_final'] = (df['exit_long_move_down'] | df['exit_long_below_tenkan_DI'] | df['exit_long_cloud_exit'] | df['exit_long_below_tk'] | df['entry_exit_indicator1'] | df['entry_exit_indicator11'] | df['entry_exit_indicator2'] | df['entry_exit_indicator22'] | df['entry_exit_indicator3'] | df['exit_long_gap_down'] | df['exit_long_price_drop'])
    ## df['exit_long_below_tenkan_DI'] | 
    # final_unique_entry_short = (df['entry_gap_short'] | df['trailing_entry_short01'] | df['exit_long_price_cloud2'] | df['entry_short_price_drop'] | 
    #                             df['entry_short_oi_vol']  | df['exit_long_adx'] | df['entry_short_above_cloud1'] | df['entry_short_above_cloud'] | df['entry_pullback_short1'] | 
    #                             df['trailing_entry_short0'] | df['entry_short_below_cloud'] | df['trailing_entry_short1'] | df['exit_long_price_cloud'] | df['trailing_entry_short02'] | df['exit_long_price_down'] | 
    #                             df['entry_short_above_tk'] | df['exit_long_tkcross'] | df['exit_long_price_tk_cross1'] | df['exit_long_price_tenkan1'] | df['exit_long_price_tk_cross2'] | 
    #                             df['entry_short_cloud_exit'] | df['exit_long_price_kijun'] | df['entry_short_price_bottom'] | df['exit_long_avoid1'] | df['exit_long_price_tk_cross3'] | df['entry_short_above_cloud2'])
    
    final_unique_entry_short = (df['entry_gap_short'] | df['entry_gap_short1'] | df['entry_short_price_drop'] | 
                                df['entry_short_oi_vol'] | df['exit_long_tkcross1'] | df['exit_long_price_tenkan'] | df['entry_short_price_top'] | 
                                df['trailing_entry_short0'] | df['trailing_entry_short1'] | 
                                df['entry_short_above_tk'] | df['exit_long_tkcross'] | df['exit_long_cloud_exit'] | 
                                df['exit_long_price_kijun'] | df['entry_short_price_bottom'] | df['exit_long_price_tk_cross3'] )
    
    
    df['final_entry_short'] = (np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [   
            ((df['entry_gap_short1'] | df['exit_long_tkcross1'] | df['entry_short_price_bottom'] | df['exit_long_price_tk_cross3'] | df['trailing_entry_short0'] )& (~df['entry_short_avoid'])) , 
            df['exit_long_tkcross'] |  (( df['entry_gap_short'] | df['entry_short_price_bottom'] |df['trailing_entry_short1'] ) & (~df['entry_short_avoid'])) ,
            ((df['exit_long_price_kijun'] | df['exit_long_tkcross'] )& (~df['entry_short_avoid'])& (~df['entry_long_avoid'])),  
            ((df['exit_long_cloud_exit'] |  df['entry_short_oi_vol']  ))  
        ],
        default=  ((df['entry_short_price_bottom'] | df['entry_short_price_drop']) ) 
    )| ( df['exit_long_price_tenkan'] |  
        df['entry_short_price_top'] | df['entry_short_above_tk'] 
          )) 
    # df['final_entry_short'] = False
    ## look for correction   exit_long_cloud_exit
    # final_unique_exit_short = (df['entry_long_cloud_enter'] | df['entry_long_below_cloud'] | df['exit_short_price_cloud'] | df['exit_short_price_cloud1'] | 
    #                            df['entry_long_below_cloud1'] | df['trailing_stop_short01']  | df['trailing_entry_long0'] | 
    #                            df['exit_short_avoid'] | df['exit_short_tenkan'] | df['exit_long_price_drop1'] | df['exit_short_kijun'] | df['exit_short_tenkan'] | df['exit_short_above_cloud'] |
    #                            df['exit_short_cloud_enter'] | df['trailing_stop_short'] | df['trailing_stop_short1']  | df['trailing_stop_short3'] |
    #                            df['exit_short_price_cloud2'] | df['exit_short_above_cloud'] | df['exit_short_tenkan_cross'] | df['exit_short_price_tk_cross2'] | 
    #                            df['exit_short_price_tk_cross1'] | df['exit_short_at_top'] | df['exit_short_avoid_range'] | df['exit_short_price_kijun'] | df['exit_short_price_move_up'] | 
    #                            df['exit_short_rsi_di'] | df['entry_long_below_tk_cloud'] | df['exit_short_move_up'] | df['exit_short_pullback']
    # ) 
    final_unique_exit_short = (df['entry_long_below_cloud'] | df['exit_short_avoid'] | df['exit_short_price_cloud2'] |
                               df['exit_short_above_cloud'] | df['exit_short_tenkan'] | df['exit_short_cloud_enter'] | 
                               df['exit_short_price_cloud'] | df['exit_short_tenkan_cross'] | df['exit_short_avoid_range'] | 
                               df['exit_short_rsi_di'] | df['exit_short_candle_green'] | df['exit_short_price_tk_cross1'] | df['exit_short_indicators'] )
    df['final_exit_short'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [

            ((df['entry_long_below_cloud']   )& (~df['entry_short_avoid'])) ,  
            df['entry_long_below_cloud']  , 
            df['exit_short_avoid']  ,
            ((df['exit_short_price_cloud2'] | df['exit_short_above_cloud'] )& (~df['entry_long_avoid']))
        ],
        default = df['exit_short_tenkan'] |  (( df['exit_short_cloud_enter'] |  df['exit_short_price_cloud'])& (~df['entry_long_avoid'])) 
    ) | (  df['exit_short_candle_green']  | 
        df['exit_short_tenkan_cross'] | df['exit_short_price_tk_cross1'] | df['exit_short_avoid_range'] | df['exit_short_rsi_di'] | df['exit_short_indicators']  )
    
    # df['exit_long'] | df['exit_long_below_cloud'] | df['exit_long_tkcross'] | df['exit_long_price_cloud'] | df['exit_long_tenkan'] | df['exit_long_kijun'] | df['trailing_stop_long']
    # df['exit_short'] | df['exit_short_above_cloud'] | df['exit_short_tkcross'] | df['exit_short_price_cloud'] | df['exit_short_tenkan'] | df['exit_short_kijun'] | df['trailing_stop_short']
    
    # df.to_csv('strategy_file.csv', index=False)
    return df

def stopless_point(row, position, entry_price, prow):
    price = row['Close']
    open = row['Open']
    high = row['High']
    low = row['Low']
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
    tenkan_kijun_diff = abs(tenkan - kijun)
    tenkan_senkou_a_diff = tenkan - senkou_a
    tenkan_senkou_b_diff = tenkan - senkou_b
    kijun_senkou_a_diff = kijun - senkou_a
    kijun_senkou_b_diff = kijun - senkou_b
    senkou_a_b_diff = abs(senkou_a - senkou_b)
    price_senkou_a_diff = price - senkou_a
    min5 = row['rolling_min_5']
    max5 = row['rolling_max_5']
    pmin5 = prow['rolling_min_5']
    pmax5 = prow['rolling_max_5']
    open_min5 = row['open_min_5']
    multiplier = 3
    stoploss_value = 0
    reverse_sl = 0
    if position == 1:
        
        
        stoploss_value = kijun - atr
        if not senkou_a_b and price_kijun:
            stoploss_value = kijun - atr*0.5
        if (tenkan_kijun_diff > atr*3):
            stoploss_value = kijun + (tenkan_kijun_diff/2)
            if (kijun > senkou_a) and row['price_above_cloud'] and (row['cloud_gap'] > atr*3):
                stoploss_value = stoploss_value - atr*0.8
            elif (kijun > senkou_a) and row['price_above_cloud'] and (row['cloud_gap'] > atr*1) and row['red_cloud'] and (not price_tenkan) and row['+DI_move_down']:
                stoploss_value = stoploss_value + atr*0.4
        elif price_cloud and price_tenkan and tenkan_kijun and row['+DI_up']:
            stoploss_value = min(min5,stoploss_value)
        if price_tenkan and ((price-open) > 100) and (row['DI_gap'] < 3) and (tenkan_kijun_diff > atr*2) and (row['+DI_move_down']):
            stoploss_value = open - atr*0.1
        
        
        
        if (price > (tenkan + atr*1.5)) and (price > senkou_a)  and (min5 < senkou_a) and ((price-min5) > 700):
            # stoploss_value = open_min5 + (price - open_min5)/2 - 100
            stoploss_value = open_min5 + atr
        elif (price_tenkan_diff > atr*0.1) and (price_senkou_a_diff > atr*3) and (row['ADX'] > 60) and (tenkan_kijun_diff > atr*2) and row['RSI'] > 65:
            stoploss_value = tenkan - atr*0.6
        elif price_tenkan and tenkan_kijun and (price_tenkan_diff > atr) and (tenkan_kijun_diff > atr*2):
            stoploss_value = tenkan - atr*1
        elif price_tenkan and tenkan_kijun and (price_tenkan_diff > atr*0.7) and (tenkan_kijun_diff > atr*1.5) and (row['ADX'] < 50):
            stoploss_value = tenkan - atr*1.5
        
        if (row['range2'] < atr*1) and price_cloud and price_tenkan and tenkan_kijun and (row['+DI'] > 32) and (row['ADX'] < 55) and (row['+DI_move_up']) :
            stoploss_value = tenkan - atr*0.5
        elif (row['range2'] < atr*1) and price_cloud and price_tenkan and tenkan_kijun and (row['+DI'] > 34) and (row['ADX'] < 60)and (row['RSI'] < row['rsi_ma']):
            stoploss_value = tenkan - atr*0.5
        
        if price_tenkan and tenkan_kijun and (price_tenkan_diff > atr*0.2) and (tenkan_kijun_diff > atr*1.5) and (row['ADX'] < 50):
            if price_cloud and (row['+DI'] < 35) and (row['-DI'] > 15) and (row['+DI_move_down']):
                stoploss_value = tenkan - atr*1
            elif (price_cloud1 or price_incloud):
                stoploss_value = tenkan - atr*1
        elif not tenkan_kijun and price_cloud1 and not senkou_a_b:
            stoploss_value = tenkan
        if (row['range1'] < atr*1) and row['+DI_move_down'] and price_kijun and not price_tenkan:
            stoploss_value = min5 - atr*1.5
        elif (row['range1'] < atr*1) and (row['DI_gap'] < 3) and (prow['DI_gap'] < 3) and price_kijun:
            stoploss_value = min5 - atr*1.5
        
        if ((price-min5) > atr*2) and (price_senkou_a_diff > atr) and (min5 < (senkou_a - atr*0.5)) and (row['ADX'] < 25):
            stoploss_value = min5 + atr*1.2


        if stoploss_value == 0:
            stoploss_value = price - atr*2



        if ((min5 - stoploss_value) > 100):
            reverse_sl = stoploss_value - 50
        elif ((min5 - stoploss_value) > 50):
            reverse_sl = stoploss_value - 70
        if reverse_sl == 0:
            if price_cloud:
                if price_tenkan and (stoploss_value <= tenkan):
                    reverse_sl = stoploss_value - 150
                elif price_tenkan and (stoploss_value > tenkan) and (price_tenkan_diff > atr*2):
                    reverse_sl = stoploss_value - atr
                    if row['+DI_up']:
                        reverse_sl = tenkan - 10
                        
                elif price_tenkan and (stoploss_value > tenkan) and (price_tenkan_diff > atr*1.5):
                    reverse_sl = stoploss_value - atr*1.1
                elif price_tenkan and (stoploss_value > tenkan) and (price_tenkan_diff > atr*0):
                    reverse_sl = stoploss_value - 100
                else:
                    reverse_sl = stoploss_value - 100
                # reverse_sl = stoploss_value - 200
            elif price_incloud:
                reverse_sl = stoploss_value - 150
            else:
                reverse_sl = stoploss_value - 150
        if not price_tenkan and tenkan_kijun and row['+DI_up'] and (tenkan_kijun_diff > atr*1) and (kijun > senkou_b) and (not senkou_a_b):
            reverse_sl = stoploss_value - atr*2
        # elif price_tenkan and tenkan_kijun and row['+DI_up'] and (kijun > senkou_b) and (not senkou_a_b):
        #     reverse_sl = min(stoploss_value, tenkan) - atr*1
        # elif row['-DI_up'] and (not tenkan_kijun) and senkou_a_b and price_cloud1 and row['price_in_lines'] and (tenkan_kijun_diff > atr):
        #     reverse_sl = stoploss_value - atr
        elif (row['+DI'] > 50) and (row['-DI'] < 11) and price_kijun and price_cloud:
            reverse_sl = stoploss_value - atr*2
        elif (row['line_gap'] < atr*0.2) and (tenkan_senkou_a_diff < atr*2):
            # if (stoploss_value - reverse_sl) < 250:
            reverse_sl = stoploss_value - atr*2
        # if price_senkou_a_diff > atr*5 and senkou_a_b and (row['cloud_gap'] > atr*3) and (row['cloud_gap'] > prow['cloud_gap']*1.1):
        #     reverse_sl = stoploss_value - atr
        # elif (row['line_gap_range'] < atr) and senkou_a_b and (senkou_a_b_diff > atr*2):
        #     reverse_sl = stoploss_value - atr
                
        # reverse_sl = reverse_sl - 100
            # stoploss_value = kijun - atr
        # if (low < (tenkan - atr*2)):
        #     stoploss_value = low - 50
        # if (price - stoploss_value) > 1000:
        # print(f'{row['date']} -- {price} -- {stoploss_value} -- {price - stoploss_value}')
        if (reverse_sl >= stoploss_value):
            reverse_sl = 0
        if stoploss_value == 0:
            reverse_sl = 0
        # reverse_sl = 0
        # stoploss_value = 0
        # if stoploss_value == 0:
        #     stoploss_value = price - atr*5
        # print(f"{stoploss_value} --- low {low}")
        if stoploss_value < price:
            # print(row['date'], stoploss_value)
            if stoploss_value > (price - 100):
                return int(stoploss_value) - 80, (int(reverse_sl) - 120)
            return int(stoploss_value), int(reverse_sl)
    return 0,0

def stopless_point_short(row, position):
    price = row['Close']
    open = row['Open']
    high = row['High']
    low = row['Low']
    atr = row['ATR']
    tenkan = row['tenkan']
    kijun = row['kijun']
    senkoua = row['senkou_a']
    senkoub = row['senkou_b']
    price_tenkan = price > tenkan
    price_kijun = price > kijun
    tenkan_kijun = tenkan > kijun
    price_kijun_diff = abs(price - kijun)
    price_tenkan_diff = abs(price - tenkan)
    price_senkoua_diff = abs(price - senkoua)
    tenkan_kijun_diff = abs(tenkan - kijun)
    max_cloud = max(senkoua, senkoub)
    min_cloud = min(senkoua, senkoub)
    max_line = max(tenkan, kijun)
    min_line = min(tenkan, kijun)
    stoploss_val = 0
    if (not price_kijun) and not tenkan_kijun:
        if not price_tenkan:
            if (tenkan_kijun_diff > atr*2):
                stoploss_val = tenkan + 200
                if row['exit_long_final']:
                    if row['+DI'] > 15:
                        # print(row['date'], tenkan)
                        # stoploss_val = tenkan + atr*0.2
                        stoploss_val = tenkan + atr*0.3
                        if row['+DI'] > 24:
                            stoploss_val = tenkan + atr*0.2
                        elif (tenkan_kijun_diff > atr*4):
                            stoploss_val = tenkan + atr*1
                        # if row['-DI'] > 20:
                        #     stoploss_val = kijun
                        # if (tenkan_kijun_diff > atr*3):
                        #     stoploss_val = tenkan + atr
                    elif row['RSI'] > 40:
                        stoploss_val = tenkan + 100
                    else:
                        stoploss_val = kijun + 100
                    


            elif (tenkan_kijun_diff > atr*1):
                if row['+DI_up']:
                    stoploss_val = kijun + atr*0.2
                elif (row['+DI_move_up']):
                    stoploss_val = kijun + atr*0.5
                elif (row['DI_gap'] < 5):
                    stoploss_val = kijun + atr*0.5
                else:
                    stoploss_val = kijun + atr*0.5
                if row['exit_long_final']:
                    if row['+DI'] > 20:
                        stoploss_val = min(price + atr,kijun)
                    elif row['RSI'] > 40:
                        # stoploss_val = kijun + atr*0.2
                        stoploss_val = min(price + atr*1.2,kijun)
                    else:
                        stoploss_val = kijun + atr*1.5
            else:
                if row['+DI_up'] or (row['+DI_move_up'] and (row['DI_gap'] < 10)):
                    stoploss_val = kijun + atr*0.2
                elif row['DI_gap_reduce'] and (row['DI_gap'] < 5):
                    stoploss_val = kijun + atr*0.2
                elif row['price_below_cloud'] and (row['RSI'] > 30) and (row['-DI'] < 40):
                    stoploss_val = kijun + atr*0.5
                    if (row['DI_gap'] > 20) and (row['+DI'] < 12):
                        stoploss_val = kijun + atr*1



        elif tenkan_kijun_diff > atr*2:
            # print(row['date'])
            stoploss_val = kijun + atr*0.2
            if price_kijun_diff < atr:
                stoploss_val = kijun + 50
        else:
            
            if row['DI_gap_reduce'] and (row['RSI'] > 40) and row['price_below_cloud']:
                
                if row['DI_gap_reduce'] and (row['RSI'] > 45):
                    stoploss_val = price + atr*0.6
                elif row['-DI_move_down']:
                    stoploss_val = price + atr*0.8
                else:
                    stoploss_val = price + atr*1

            elif (min_line > max_cloud) and (tenkan_kijun_diff > atr*1):
                stoploss_val = kijun + atr*0.3
            elif (max_line < max_cloud):
                if price_kijun_diff > atr*2:
                    stoploss_val = kijun
            else:
                # print(row['date'])
                if (tenkan_kijun_diff < atr*0.4) and (row['DI_gap'] < 4):
                    stoploss_val = high + atr
                else:
                    stoploss_val = kijun + atr*0.2
    elif (not price_kijun) and tenkan_kijun:
        # stoploss_val = tenkan
        if row['price_above_cloud'] and (row['+DI'] < 15) and (tenkan_kijun_diff < atr*0.5):
            stoploss_val = kijun + atr*0
            if (tenkan_kijun_diff < atr*0.8) and (row['RSI'] > 43):
                stoploss_val = max_line + 50
        elif row['price_above_cloud'] and (tenkan_kijun_diff > atr*0.5):
            stoploss_val = tenkan + atr*3
        elif (price < max_cloud):
            if price_tenkan_diff < atr:
                # print(row['date'])
                stoploss_val = tenkan + atr
                if (row['+DI_up']):
                    stoploss_val = tenkan + atr*0.5
            elif price_tenkan_diff > atr:
                stoploss_val = tenkan + atr
    elif price_kijun and tenkan_kijun:
        # print(row['date'])
        if price_tenkan_diff > atr:
            stoploss_val = tenkan + atr*1

        elif (price_tenkan_diff <= atr) and (row['+DI'] > 25):
            stoploss_val = tenkan + atr*1
        elif (price_tenkan_diff <= atr) and (row['+DI'] <= 25):
            stoploss_val = tenkan + atr*2
        else:
            stoploss_val = tenkan + atr*2
    elif price_kijun and not tenkan_kijun:
        if (tenkan_kijun_diff < atr*1) and (row['+DI'] > 15):
            stoploss_val = kijun + atr*1.2
            if (row['candel_size'] < 100) and (price_senkoua_diff < atr):
                stoploss_val = high + atr*0.8
            if row['price_below_cloud'] and ((min_cloud - max_line) < atr*2):
                stoploss_val = min_cloud
        else:
            stoploss_val = kijun + atr*2.5
            
    if (min_line < max_cloud) and (max_line > max_cloud) & (row['DI_gap'] < 5):
        stoploss_val = min_cloud + atr*4
    if (row['line_gap_range'] < atr) and (row['DI_gap_range'] < 5):
        stoploss_val = price + atr*1
        
    if stoploss_val > price:
        return int(stoploss_val), int(stoploss_val + 5)
    return int(0), int(0)

def stoploss_entry_point(row, prow):
    price = row['Close']
    open = row['Open']
    high = row['High']
    low = row['Low']
    atr = row['ATR']
    tenkan = row['tenkan']
    kijun = row['kijun']
    senkoua = row['senkou_a']
    senkoub = row['senkou_b']
    max5 = row['rolling_max_5']
    price_tenkan = price > tenkan
    price_kijun = price > kijun
    tenkan_kijun = tenkan > kijun
    price_kijun_diff = abs(price - kijun)
    price_tenkan_diff = abs(price - tenkan)
    price_senkoua_diff = abs(price - senkoua)
    tenkan_kijun_diff = abs(tenkan - kijun)
    max_cloud = max(senkoua, senkoub)
    min_cloud = min(senkoua, senkoub)
    max_line = max(tenkan, kijun)
    min_line = min(tenkan, kijun)
    stoploss_val = 0
    # print(row['date'], "--0")
    if (not price_kijun) and not tenkan_kijun:
        if not price_tenkan:
            if (tenkan_kijun_diff > atr*2):
                stoploss_val = tenkan + atr*0.5
                if row['exit_long_final']:
                    if row['+DI'] > 15:
                        stoploss_val = tenkan + atr*0.3
                        if (tenkan_kijun_diff > atr*4):
                            stoploss_val = tenkan + atr*1
                    elif row['RSI'] > 40:
                        stoploss_val = tenkan + 100
                    else:
                        stoploss_val = kijun + 100
                if (row['cloud_gap'] < atr*0.2):
                    stoploss_val = kijun

            elif (tenkan_kijun_diff > atr*1):
                # print(row['date'], "---1")
                if row['+DI_up']:
                    stoploss_val = kijun + atr*0.2
                elif (row['+DI_move_up']):
                    stoploss_val = kijun + atr*0.5
                elif (row['DI_gap'] < 5):
                    stoploss_val = kijun + atr*0.5
                else:
                    stoploss_val = kijun + atr*0.5
                if row['exit_long_final']:
                    if row['+DI'] > 20:
                        stoploss_val = min(price + atr,kijun)
                    elif row['RSI'] > 40:
                        # stoploss_val = kijun + atr*0.2
                        stoploss_val = min(price + atr*1.2,kijun)
                    else:
                        stoploss_val = kijun + atr*1.5
            else:
                # print(row['date'], "---2")
                if row['+DI_up'] or (row['+DI_move_up'] and (row['DI_gap'] < 10)):
                    stoploss_val = kijun + atr*0.2
                    if (row['line_gap'] < atr*0.5) and row['price_below_cloud']:
                        stoploss_val = kijun + atr*2
                elif row['DI_gap_reduce'] and (row['DI_gap'] < 5):
                    stoploss_val = kijun + atr*0.2
                elif row['price_below_cloud'] and (row['RSI'] > 30) and (row['-DI'] < 40):
                    stoploss_val = kijun + atr*0.5
                    
                    
        elif tenkan_kijun_diff > atr*2:
            stoploss_val = kijun + atr*0.2
            if price_kijun_diff < atr:
                stoploss_val = kijun + 50
        else:
            
            if row['DI_gap_reduce'] and (row['RSI'] > 40) and row['price_below_cloud']:
                
                if row['DI_gap_reduce'] and (row['RSI'] > 45):
                    stoploss_val = price + atr*0.6
            
                    if row['green_cloud']:
                        stoploss_val = kijun

                elif row['-DI_move_down']:
                    stoploss_val = price + atr*0.8
                else:
                    stoploss_val = price + atr*1

            elif (min_line > max_cloud) and (tenkan_kijun_diff > atr*1):
                
                stoploss_val = kijun + atr*0.3
            elif (max_line < max_cloud):
                if price_kijun_diff > atr*2:
                    stoploss_val = kijun
            else:
                # print(row['date'])
                if (tenkan_kijun_diff < atr*0.4) and (row['DI_gap'] < 4):
                    stoploss_val = high + atr
                else:
                    stoploss_val = kijun + atr*0.2
                    # if row['price_above_cloud']:
                    #     stoploss_val = kijun + atr*0.5

    elif (not price_kijun) and tenkan_kijun:
        # print(row['date'], "--1")
        # stoploss_val = tenkan
        if row['price_above_cloud'] and (row['+DI'] < 15) and (tenkan_kijun_diff < atr*0.5):
            stoploss_val = kijun + atr*0
            if (tenkan_kijun_diff < atr*0.8) or row['-DI_up']:
                stoploss_val = max_line + 50
        elif row['price_above_cloud'] and (tenkan_kijun_diff > atr*0.5):
            stoploss_val = tenkan + atr*3
            if (tenkan_kijun_diff < atr*0.8):
                stoploss_val = max_line + 50
        elif (price < max_cloud):
            if price_tenkan_diff < atr:
                stoploss_val = tenkan + atr
            elif price_tenkan_diff > atr:
                stoploss_val = tenkan + atr
        # if (tenkan_kijun_diff < atr*0.8) and row['-DI_up']:
        #     stoploss_val = max_line + atr
    elif price_kijun and tenkan_kijun:
        # print(row['date'])
        if price_tenkan_diff > atr:
            stoploss_val = tenkan + atr*1
        elif (price_tenkan_diff <= atr) and (row['+DI'] > 25):
            stoploss_val = tenkan + atr*1
        elif (price_tenkan_diff <= atr) and (row['+DI'] <= 25):
            stoploss_val = tenkan + atr*2
            if price < min_cloud:
                stoploss_val = min(min_cloud+atr*0.7, stoploss_val)
        else:
            stoploss_val = tenkan + atr*2
    elif price_kijun and not tenkan_kijun:
        # print(row['date'], "--2")
        if (tenkan_kijun_diff < atr*1) and (row['+DI'] > 15):
            stoploss_val = kijun + atr*1.2
            if (row['DI_gap'] < 4) and (price_kijun_diff > atr):
                stoploss_val = high + atr*0.8
            elif (row['DI_gap'] < 3):
                # print(row['date'], "--2")
                stoploss_val = high + atr*0.5
            elif row['-DI_up'] and (price_kijun_diff > atr) and (row['DI_gap'] > 5):
                stoploss_val = high + atr*0.5

            if (row['candel_size'] < 100) and (price_senkoua_diff < atr):
                stoploss_val = high + atr*0.8
            if row['price_below_cloud'] and ((min_cloud - max_line) < atr*2):
                stoploss_val = min_cloud
        else:
            stoploss_val = kijun + atr*2.5
            
    if (min_line < max_cloud) and (max_line > max_cloud) & (row['DI_gap'] < 5):
        stoploss_val = min_cloud + atr*4
    
    if stoploss_val > price:
        return int(stoploss_val)
    return int(0) 


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
    """Wait until next 15-minute candle + 15 seconds mark."""
    now = datetime.now() + timedelta(hours=5, minutes=30)
    # now = datetime.now()
    log(now)
    # Find the next 15-minute multiple
    min = 30
    next_minute = (now.minute // min + 1) * min
    # log(next_minute)
    next_time = now.replace(minute=0, second=20, microsecond=0) + timedelta(minutes=next_minute)
    # log(next_time)
    if next_minute >= 60:
        next_time = now.replace(hour=(now.hour + 1) % 24, minute=0, second=20, microsecond=0)
    wait_seconds = (next_time - now).total_seconds()
    # log(wait_seconds)
    if wait_seconds < 0:
        wait_seconds += (60*min)  # just in case of rounding errors
    log(f"⏳ Waiting {int(wait_seconds)} sec until next candle time {next_time.strftime('%H:%M:%S')}...")

    time.sleep(wait_seconds)


def log(msg):
    with open("static/logs.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")