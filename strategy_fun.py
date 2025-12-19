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
    df['tenkan_above_kijun'] = df['tenkan'] > df['kijun']
    df['tenkan_below_kijun'] = df['tenkan'] < df['kijun']
    df['price_below_tenkan'] = df['Close'] < df['tenkan']
    df['price_above_tenkan'] = df['Close'] > df['tenkan']
    df['line_gap'] = abs(df['tenkan'] - df['kijun'])
    df['cloud_gap'] = abs(df['senkou_a'] - df['senkou_b'])

    df['price_above_tk'] = (df['Close'] > df['tenkan']) & (df['Close'] > df['kijun'])
    df['price_below_tk'] = (df['Close'] < df['tenkan']) & (df['Close'] < df['kijun'])

    df['tk_cross_up'] = df['tenkan_above_kijun'] & (~df['tenkan_above_kijun']).shift(1).fillna(False)
    df['tk_cross_down'] = df['tenkan_below_kijun'] & (~df['tenkan_below_kijun']).shift(1).fillna(False)
    # df['Price_above_tk'] = (((df['Close'] > df['tenkan']).rolling(5).sum() >= 3) &
    #                         ((df['Close'] > df['Close'].shift(1).rolling(5).sum() >= 3) ))
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
    df['price_in_cloud'] = ((~df['price_above_cloud']) & (~df['price_below_cloud']))
    df['price_in_bt_lines'] = ((~df['Price_above_tk']) & (~df['Price_below_tk']))
    df['Market_Regime'] = df.apply(classify_trend_ichimoku, axis=1)
    
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
    df['rolling_min_120'] = df['Close'].rolling(window=120).min()
    df['rolling_max_120'] = df['Close'].rolling(window=120).max()
    df['rolling_min_60'] = df['Close'].rolling(window=60).min()
    df['rolling_max_60'] = df['Close'].rolling(window=60).max()
    # Combine all into long entry
    

    ####################################################################################
    ####################################################################################
    ####################################################################################
    df = df.copy()
    df['entry_long'] = (
        df['tk_cross_up'] &
        df['price_above_cloud'] &
        (df['Close'] > df['Close'].shift(1)) &     # bullish momentum
        # (df['RSI'] > 50) &                         # bullish RSI
        (df['ADX'] > 20) &                         # strong trend
        (df['Volume'] > df['vol_ma'])              # volume confirmation
    )
    df['entry_long_below_cloud'] = (df['price_below_cloud'] &
                                    (df['Close'] > (df['rolling_min_10'].shift(1)+df['ATR']*0)) &
                                    ((df['Close']+150) > df['tenkan']) & 
                                    (df['Close'] > df['Close'].shift(1)) & 
                                    # (df['Close'].shift(1) > df['Open'].shift(1)) & 
                                    ((df['Close']-200) > df['Open']))
    df['entry_long_price']=(
                                (((df['Close'] > df['Open']).rolling(3).sum() >= 3) | (df['Close'] > df['rolling_max_10'].shift(1))) &
                                (df['Close'] > cloud_top) &
                                ((df['Close'] < df['tenkan']).rolling(3).sum() >= 1)  &
                                price_above_tk
                            )
    df['entry_long_price_cloud'] = (
                                        entry_condition & 
                                        (df['High'] <= cloud_top) &
                                        ((df['Close']-50) > df['Close'].shift(1)) &
                                        (df['Close'] > (df['rolling_min_10'] + df['ATR']*0.5))
                                    )
    df['entry_long_oi_vol'] = (
                                    df['tenkan_above_kijun'] &
                                    ((df['Close'] > df['Open']).rolling(3).sum() >= 2))
    df['entry_long_oi_vol1'] = (
                                    df['tenkan_above_kijun'] &
                                    ((df['Close'] > df['Open']).rolling(3).sum() >= 2)& 
                                    (df['OI'] < (df['oi_ma']*1.1)))
    df['entry_kumo_break_long'] = (
                                    (df['Close'] > df[['senkou_a','senkou_b']].max(axis=1)) &  # price breaks cloud top
                                    (df['Close'].shift(1) <= df[['senkou_a','senkou_b']].max(axis=1).shift(1)) &  # previous bar was below
                                    (df['Volume'] > df['vol_ma']) &
                                    (df['Close'] > df['Close'].shift(1)) 
                                )
    df['entry_long_price_enter_cloud'] = ((((df['Close']+10) > df['Open']).rolling(3).sum()>=3) &
                                          (((df['Close']+10) > df['tenkan']).rolling(2).sum()>=2) &
                                          (((df['Close']+10) > df['kijun']).rolling(2).sum()>=2) )
    df['entry_long_high'] = (
                              df['tenkan_above_kijun'] &
                              ((df['Close'] > df['Open']).rolling(3).sum() >= 2) &
                              ((df['Close']) >= df['rolling_max_10']) &
                              (df['ADX'] > df['ADX'].shift(2))&
                              (df['OI'] > (df['oi_ma'])))    
    df['exit_long_price_drop1'] = (((df['line_gap'] < df['line_gap'].shift(1)).rolling(3).sum() >= 2) &
                                  ((df['Close'] < df['Close'].shift(1)).rolling(3).sum() >= 3) &
                                #   ((df['Close'] < df['kijun'].shift(1)).rolling(3).sum() >= 3) &
                                  ((df['Close'] < df['Open'].shift(1)).rolling(3).sum() >= 3))
    df['entry_gap_long'] = (
                                # (df['Close'] > df['Open']) & 
                                (((df['Close']+0) > df['Close'].shift(1)).rolling(3).sum() >= 2)&
                                (((df['Close']+100) > df['Open'].shift(1)).rolling(2).sum() >= 2)&
                                (((df['Close']+100) > df['tenkan'].shift(0)).rolling(2).sum() >= 1)&
                                df['price_above_cloud']
                            )
    df['entry_pullback_long'] = (
                                    break_above_range &
                                    (df['Close'] > df[['senkou_a','senkou_b']].max(axis=1)) &  # trend up
                                    # (df['Close'] > df['kijun']) &                              # rebound from Kijun
                                    ((df['Low'].shift(1) < df['kijun'].shift(1)) | 
                                    (df['Low'].shift(2) < df['tenkan'].shift(2))) &              # yesterday dipped below kijun
                                    (df['Volume'] > df['vol_ma']*0.8)
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
    df['trailing_entry_long'] = (df['Close'].expanding().max() + 1 * df['ATR'])
    df['trailing_entry_long0'] = (
                                (df['Close'] > (df['rolling_min_10'].shift(1))) &
                                (df['Close'] > df['trailing_entry_long']* 0.96) &
                                ((df['Open']+300) > df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']-50) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.1))
                                )
    df['trailing_entry_long3'] = (
                                    ((df['Close'] > (df['rolling_max_10'] - df['ATR']*0)) |
                                    (df['Close'] > (df['rolling_min_10'] + df['ATR']*0.5))) &
                                    ((df['Close']+100) > df['Open'].shift(0))
                                )
    df['trailing_entry_long2'] = (
                                (df['Close'] > df['trailing_entry_long']* 0.97) &
                                ((df['Close'] > (df['rolling_max_10'] -df['ATR'])) |
                                (df['Close'] > (df['rolling_min_10'] + df['ATR']*1)) )&
                                (df['OI'] > df['oi_ma']) &
                                ((df['Open']-50) < df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']-50) > df['Open'].shift(0))
                                )
    df = df.copy()

    df['entry_short_kumo_break'] = (((df['Close']+100) < df['rolling_max_10'].shift(1)) &
                                  (df['Close'].shift(1) > df[['senkou_a','senkou_b']].min(axis=1)) &
                                  (df['Close'] < df[['senkou_a','senkou_b']].min(axis=1)) &
                                  (df['Close'] < df['tenkan']) &
                                  ((df['Close'] < df['Close'].shift(1)).rolling(2).sum() >= 2) &
                                  (df['Volume'] > (df['vol_ma']*1)))
    df['entry_gap_short'] = (
                                ((df['Close'] < df['Open']).rolling(3).sum() >= 2) & 
                                (((df['Close']+0) < df['Close'].shift(1)).rolling(3).sum() >= 2)&
                                (df['Close'].shift(1) < df['Open'].shift(1)) & 
                                (df['Open'] < df['Open'].shift(1)) &
                                df['price_below_cloud']
                            )
    df['entry_short_above_cloud'] = np.where(df['price_above_cloud'], (
                                    ((df['Close']+250) < df['tenkan']) & 
                                    ((df['tenkan'] < df['tenkan'].shift(1)).rolling(2).sum() >= 2) &
                                    ((df['Close']) < df['Close'].shift(1)) &
                                    (df['Close'].shift(1) < df['Open'].shift(1)) 
                                    ),False)
    df['entry_short_above_cloud1'] = ((df['Close'] <= df[['senkou_a','senkou_b']].min(axis=1)) &
                                   ((df['Close']-0) < df['Close'].shift(1)) &
                                   (df['Close'] < (df['rolling_min_10']+df['ATR'])) &
                                   ((df['Close']-100) < df['tenkan']))

    df['entry_short_oi_vol'] = (
                                (df['OI'] > df['oi_ma']) &
                                (df['Volume'] > df['vol_ma']) &
                                df['tenkan_below_kijun'] &
                                (df['Close'] < df['tenkan'] )&
                                # (df['Low'] < df['senkou_b'] )&
                                ((df['Close'] < df['Open']).rolling(3).sum() >= 2))
    df['entry_short_oi_vol1'] = (
                                (df['OI'] > (df['oi_ma']*1.2)) &
                                # (df['Volume'] > df['vol_ma']) &
                                df['tenkan_below_kijun'] &
                                # (df['Close'] < df['tenkan'] )&
                                # (df['Low'] < df['senkou_b'] )&
                                ((df['Close'] < df['Open']).rolling(3).sum() >= 2))
    df['entry_pullback_short'] = (
                                    # break_below_range &
                                    ((df['Close']+100) < df[['senkou_a','senkou_b']].min(axis=1)) &  # trend down
                                    # (df['Close'] < df['kijun']) &
                                    ((df['High'].shift(1) > df['kijun'].shift(1)) | 
                                    (df['Open'].shift(2) > df['tenkan'].shift(2))) &
                                    (df['Volume'] > df['vol_ma']*0.8) 
                                )


    df['trailing_entry_short'] = (df['Close'].expanding().max() - 2 * df['ATR'])

    df['trailing_entry_short0'] = np.where(df['price_above_cloud'],
                                (
                                    (df['Close'] < df['trailing_entry_short']* 0.95) &
                                    (df['Close'] < (df['rolling_max_10']-100)) &
                                    ((df['Open']-100) < df['Open'].shift(1)) &
                                    ((df['Close']-50) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.1)) ),

                                    ((df['Close'] < df['trailing_entry_short']* 0.96) &
                                    ((df['Open']+300) > df['Open'].shift(1)) &
                                    (((df['Close']-50) > df['Open'].shift(0))) &
                                    (df['OI'] < (df['oi_ma']*1.1)) )
                                )
    df['trailing_entry_short1'] = np.where(df['price_above_cloud'],
                                (
                                    ((df['Close']-0) < (df['rolling_max_10']-df['ATR']*1)) &
                                    (((df['Close'])< df['kijun']).rolling(2).sum() >= 2 )&
                                    ((df['Open']-10) < df['Open'].shift(1)) &
                                    (df['ADX'] > df['ADX'].shift(1)*0.92) &
                                    ((df['Close']+100) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.075))
                                ),
                                (
                                    (((df['Close'])< df['kijun']).rolling(3).sum() >= 2 )&
                                    ((df['Open']-10) < df['Open'].shift(1)) &
                                    (df['ADX'] > df['ADX'].shift(1)*0.92) &
                                    ((df['Close']+0) < df['Open'])& (df['OI'] > (df['oi_ma']*0.98)) &
                                    (df['OI'] < (df['oi_ma']*1.1))
                                ))
    df['trailing_entry_short2'] = np.where(df['price_above_cloud'],
                                    (
                                        ((df['Close'] - 250)> df['Open']) & 
                                        (df['Close'] < df['trailing_entry_short']* 1) &
                                        (df['OI'] > (df['oi_ma']*1.015)) &
                                        df['price_above_cloud'].shift(1) 
                                    ),
                                    (
                                        ((df['Close'] + 200) < df['Open']) & 
                                        (df['Close'] < (df['rolling_min_10']+df['ATR']*1)) &
                                        (df['Volume'] > (df['vol_ma']*0.9)) &
                                        (df['OI'] > (df['oi_ma']*1.1)) 
                                    ))
    df['trailing_entry_short3'] = np.where(df['price_above_tenkan'],
                                ((df['Close'] < df['trailing_entry_long']* 0.96) &
                                ((df['Open']-0) < df['Open'].shift(1)) &
                                (((df['Close']-300) < df['Open']) & (df['Close']+300) > df['Open'])& 
                                (df['OI'] < (df['oi_ma']*1.1))),
                                (df['Close'] < df['trailing_entry_long']* 0.96) &
                                ((df['Open']-0) < df['Open'].shift(1)) &
                                ((df['Close']-100) > df['Open'].shift(0))& 
                                (df['OI'] < (df['oi_ma']*1.1)) ) 
    df['trailing_stop_long0'] = (df['Close'].expanding().max() - 2 * df['ATR'])
    df['trailing_stop_long5'] = np.where(df['price_above_cloud'],
                                        (((df['Close'] - 250)> df['Open']) & 
                                        (df['Close'] < df['trailing_stop_long0']* 1) &
                                        (df['OI'] > (df['oi_ma']*1.015))
                                        ),
                                        False)
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
    df['exit_long_price_turn'] = ((((df['High'] - df['Open']) > (df['Open'] - df['Low'])*1.5).rolling(2).sum()>=1) &
                                   (df['price_above_cloud']) &
                                   (((df['Close']+50) < df['Open']).rolling(4).sum()>=3) &
                                   (((df['Close']+10) < df['Close'].shift(1)).rolling(2).sum()>=2) &
                                   (df['Close'] < (df['rolling_max_10'] - df['ATR']*0)) &
                                   ((df['Close']+100) < df['tenkan'])) 
    df['exit_long_price_turn1'] =  ((((df['High'] - df['Open']) > (df['Open'] - df['Low'])*1).rolling(3).sum()>=2) &
                                    (df['price_above_cloud']) &
                                   (((df['Close']+20) < df['Open']).rolling(3).sum()>=2) &
                                   (df['Close'] < (df['rolling_max_10'] - df['ATR']*1)) &
                                   (((df['Close']) < df['kijun']).rolling(2).sum()>=2) &
                                   ((df['RSI'] < df['rsi_ma']).rolling(3).sum() >= 3)
                                   )
    
    df['exit_long_avoid'] = (((df['rolling_max_10'] - df['rolling_min_10']) < (df['ATR']*0.8)))
    df['exit_long_price_drop'] = (((df['line_gap'] <= df['line_gap'].shift(1)).rolling(3).sum() >= 2) &
                                  ((df['Close'] < df['Close'].shift(1)).rolling(3).sum() >= 2) &
                                  ((df['Close'] < df['tenkan'].shift(0)).rolling(3).sum() >= 3) &
                                  ((df['Close'] < df['Open'].shift(0)).rolling(3).sum() >= 3) &
                                  (df['Close'] < (df['rolling_max_10'].shift(1) - df['ATR']*1)) &
                                  (df['price_above_cloud']))
    df['exit_long_price_down'] = (df['price_above_cloud'] &
                                  ((df['Close'] < df['Close'].shift(1)).rolling(2).sum() >= 2)&
                                  ((df['Close'] < df['tenkan'].shift(1)).rolling(2).sum() >= 1)&
                                #   ((df['tenkan'] < df['tenkan'].shift(1)).rolling(3).sum() >= 2)&
                                  (df['Close'] < (df['rolling_max_10']-(df['ATR']*2))) &
                                  (df['ADX']< df['ADX'].shift(1)) &
                                  (df['Volume'] < (df['vol_ma']*1.1))
                                  )
    df['exit_long_kijun'] = (((df['Close'] < df['kijun']).rolling(3).sum() >= 3) &
                             (((df['Close']-100) < df['Open'].shift(0)).rolling(2).sum()>=1) &
                             (((df['tenkan']) < df['kijun']).rolling(2).sum()>=2) &
                             ((df['OI'] < df['oi_ma']*1.07).rolling(2).sum()>=2)
                             )
    df['exit_long_kijun1'] = (((df['Close'] < df['kijun']).rolling(3).sum() >= 3) &
                             ((df['Close'] < df['tenkan']).rolling(3).sum() >= 1))
    df['exit_long_kumo_break'] = (((df['Close']+100) < df['rolling_max_10'].shift(1)) &
                                  (df['Close'].shift(1) > df[['senkou_a','senkou_b']].min(axis=1)) &
                                  (df['Close'] < df[['senkou_a','senkou_b']].min(axis=1)) &
                                  (df['Close'] < df['tenkan']) &
                                  ((df['Close'] < df['Close'].shift(1)).rolling(3).sum() >= 2) &
                                  (df['Volume'] > (df['vol_ma']*1)))
    

    df['exit_long_tkcross'] = ((df['tk_cross_down'].rolling(3).sum() >= 2) | 
                                (((df['Close'].shift(1) > df['tenkan']) | (df['Open'].shift(1) > df['kijun'])) &
                                (df['Open'] > (df['Close']+200)) &
                                ((df['Close']+100) < df['tenkan']) &
                                (df['Close'] < df['kijun']) &
                                (df['Close'] < (df['rolling_min_10']+df['ATR']*0.4)) &
                                (df['Volume'] < df['vol_ma']*1) ) | (
                                df['price_above_cloud'] &
                                (df['Close'] < (df['rolling_min_10']+df['ATR']*0.4)) &
                                (df['line_gap'] < df['line_gap'].shift(1)) &
                                (df['line_gap'].shift(1) <= df['line_gap'].shift(2)) &
                                ((df['Close']+100) < df['tenkan']) &
                                (df['Close'] < df['kijun']) &
                                (df['Open'] > (df['Close']+200)) &
                                (((df['Close'].shift(1)+50) > df['tenkan']) | ((df['Open'].shift(1)+50) > df['kijun'].shift(1))) 
                                ))
    df['exit_long_tkcross1'] = (((df['Close'] <= df['kijun']).rolling(3).sum() >= 3) &
                                (df['OI'] < df['oi_ma']*1.05) &
                                (((df['tenkan']-df['ATR']) <= df['kijun']).rolling(3).sum() >= 2) &
                                ((df['Close'] <= df['Close'].shift(1)).rolling(3).sum() >= 2) &
                                (df['Close'] < (df['rolling_max_10']-df['ATR'])) )
    df['exit_long_price_reversion'] = ((((df['Close']+0) < df['Open'].shift(0)).rolling(3).sum() >= 3) &
                                       (((df['Close']+10) < df['Close'].shift(1)).rolling(2).sum() >= 2) &
                                       (((df['Close']+50) < df['tenkan']).rolling(2).sum() >= 2) &
                                       ((df['RSI'] < df['RSI'].shift(1)).rolling(3).sum()>=3))
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
    
    
    df['trailing_stop_long'] = (
                                # (df['Close'] < df['trailing_stop_long0']* 1) &
                                (df['Close'] < (df['rolling_max_10']-df['ATR'])) &
                                ((df['Open']-10) < df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']-50) > df['Open'].shift(0))& (df['OI'] < (df['oi_ma']*1.1))
                                )
    df['trailing_stop_long11'] = (
                                # (df['Close'] < df['trailing_stop_long0']* 1) &
                                (df['Close'] < (df['rolling_min_10']+df['ATR'])) &
                                ((df['Open']+50) < df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                (((df['Close']+100) < df['Open']).rolling(2).sum() >= 2)& 
                                (df['OI'] < (df['oi_ma']*1.1))
                                )
    df['trailing_stop_long1'] = (
                                (df['Close'] < df['trailing_stop_long']* 1) &
                                ((df['Open']-10) < df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']-40) > df['Open'].shift(0))
                                )
    df['trailing_stop_long10'] = (
                                (df['Close'] < df['rolling_max_60'] - df['ATR']*2) &
                                (((df['Open']+50) < df['Open'].shift(1)).rolling(2).sum() >=2) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                (((df['Close']-100) < df['Open']).rolling(3).sum() >=2)
                                )
    
    
    df['trailing_stop_long2'] = (
                                (df['Close'] < df['trailing_stop_long0']* 0.95) &
                                # (df['Close'] < (df['rolling_max_10']-(df['ATR']*3))) &
                                ((df['Open']-10) < df['Open'].shift(1)) &
                                # (df['Close'] < df['Close'].shift(1)) &
                                ((df['Close']-50) > df['Open'].shift(0)) & (df['OI'] < (df['oi_ma']*1.1))
                                )
    df['trailing_stop_long4'] = np.where(df['price_above_cloud'],
                                         (((df['Close'] - 150) > df['Open']) & 
                                          (df['Close'] < df['trailing_stop_long0']* 0.99) &
                                          (df['Close'] < (df['rolling_max_10'].shift(0)-df['ATR']*0)) &                                          (df['OI'] < (df['oi_ma']*1.1))
                                          ),
                                         False)
    df['trailing_stop_long3'] = np.where(df['price_above_cloud'],
                                         (((df['Close'] - 150)< df['Open']) & df['trailing_stop_long'] &  (df['OI'] < (df['oi_ma']*0.9))),
                                         df['trailing_stop_long11'])


    # Exit Short
    # Exit Short
    # Exit Short
    # Exit Short
    df = df.copy()
    extra_condition1 = ((df['Close'] - df['Open']) >= 10).rolling(2).sum() >= 2
    df['exit_short_price_cloud1'] = np.where(
                                            extra_condition1,
                                            (df['Close'] <= cloud_bottom) & (df['High'] <= cloud_top),   # WHEN ABOVE CLOUD + TENKAN
                                            entry_condition & (df['High'] <= cloud_top)                   # OTHERWISE
                                        ) 
    df['exit_short_high'] = (((df['Close']-100) > df['rolling_min_10'].shift(0)) &
                            ((df['Close'] > df['Open']).rolling(3).sum() >= 2) &
                            ((df['Close'] > df['Close'].shift(1)).rolling(3).sum() >= 2) &
                            ((df['High'] > df['tenkan'].shift(1)).rolling(3).sum() >= 1) &
                            (df['Volume'] > (df['vol_ma']*1))
                            )
    df['exit_short_cloud_tenkan'] = (df['price_below_cloud'] &
                                     df['price_above_tenkan'] &
                                     df['tenkan_above_kijun'] &
                                     ((df['Close'] > df['Open']).rolling(3).sum() >= 2) )
    

    
    df['exit_short_avoid'] = (
                              (df['Close'] > df[['senkou_a','senkou_b']].min(axis=1)) &
                              ((df['rolling_max_10'].shift(0) - df['rolling_min_10'].shift(0)) < (df['ATR']*1)))
    
    df['exit_short_price_cloud'] = entry_condition & (df['High'] <= cloud_top)
    df['exit_short_tenkan'] = (((df['Close'] > df['Open']).rolling(3).sum() >= 3) &
                               ((df['Close'] > (df['tenkan'])).rolling(3).sum() >= 2))
    df['exit_short_above_cloud'] = ((df['Close'] >= df[['senkou_a','senkou_b']].max(axis=1)) &
                                    ((df['Close']-100) > df['Close'].shift(1)) &
                                    ((df['Close']) > df['tenkan']) 
                                    ) 
    df['exit_short_kijun'] = ((df['Close'] > df['kijun']).rolling(3).sum() >= 3)
    df['exit_short_price_turn'] = ((((df['Open'] - df['Low']) > (df['High'] - df['Close'])*7).rolling(2).sum()>=1) &
                                #    (df['price_below_cloud']) &
                                   (((df['Close']-50) > df['Open']).rolling(2).sum()>=2) &
                                   (((df['Close']-10) > df['Close'].shift(1)).rolling(2).sum()>=2) &
                                   (df['Close'] > (df['rolling_min_10'] + df['ATR']*1)) &
                                   ((df['Close']-100) > df['tenkan']))
    df['exit_short_tkcross'] = ((df['tk_cross_up'].rolling(3).sum() >= 2) | (
                                df['price_below_cloud'] &
                                (df['Close'] > (df['rolling_max_10']-df['ATR']*1)) &
                                (df['line_gap'] < df['line_gap'].shift(1)) &
                                (df['line_gap'].shift(1) <= df['line_gap'].shift(2)) &
                                ((df['Close']+100) > df['tenkan']) &
                                (df['Close'] > df['kijun']) &
                                (df['Open'] < (df['Close']+200)) &
                                (((df['Close'].shift(1)+50) < df['tenkan']) | ((df['Open'].shift(1)+50) < df['kijun'].shift(1))) 
                            ))


    df['trailing_stop_short0'] = df['Close'].expanding().max() - 2 * df['ATR']
    df['trailing_stop_short'] = (
                                (df['Close'] > (df['trailing_stop_short0']*0.89)) &
                                (df['Close'] > (df['rolling_min_10']+(df['ATR']*1))) &
                                ((df['Open']-50) > df['Open'].shift(1)) &
                                ((df['Open']-0) > df['Close']) 
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


    df = df.copy()

    final_unique_long_condition = (df['entry_long_below_cloud'] | df['entry_long_price'] | df['entry_long_price_cloud'] | df['entry_long_oi_vol'] | 
                                   df['entry_long_oi_vol1'] | df['entry_long_price_enter_cloud'] | df['entry_kumo_break_long'] | df['entry_long_high'] | 
                                   df['exit_long_price_drop1'] | df['entry_gap_long'] | df['entry_long'] | df['entry_pullback_long'] | 
                                   df['trailing_entry_long0'] | df['trailing_entry_long1'] | df['trailing_entry_long2'] | df['trailing_entry_long3']
                                )

    df['final_entry_long'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [
            df['trailing_entry_long1'] |   df['entry_long_below_cloud']  | ((df['trailing_stop_short3']| df['exit_short_price_turn'] | df['trailing_entry_long0'] | df['entry_long_price'])& (~df['entry_long_avoid'])),
            (( df['entry_long_price_cloud'] | df['trailing_entry_long3']) & (~df['entry_long_avoid1']) & (~df['entry_short_avoid'])),  ##.    ###df['entry_long_oi_vol'] | 
            df['entry_long_price_enter_cloud']|df['exit_short_price_cloud1'] |  ((df['entry_long_price_cloud'] |df['entry_long_oi_vol1'] | df['entry_kumo_break_long']) )  ,#
            df['entry_long_below_cloud']| df['trailing_entry_long2']| df['entry_long_high'] | df['exit_short_tenkan'] |  ((df['exit_short_cloud_tenkan'] | df['exit_short_price_turn'] | df['exit_long_price_drop1'] | df['entry_long_price_cloud'] | df['entry_gap_long']) & (~df['entry_long_avoid']))
        ],
        default= df['exit_short_price_cloud'] | df['exit_short_tkcross'] | df['entry_pullback_long'] |df['entry_long_price_cloud'] | df['entry_long_below_cloud'] 
    )
    # df['final_entry_long'] = False

    final_unique_exit_long = (df['exit_long_price_turn'] | df['exit_long_price_turn1'] | df['entry_short_above_cloud'] | 
                              df['exit_long_tkcross'] | df['exit_long_tkcross1'] | df['exit_long_kijun'] | df['exit_long_kijun1'] | df['exit_long_price_reversion'] | df['exit_long_price_cloud'] | 
                              df['exit_long_price_drop'] | df['exit_long_price_down'] | df['exit_long_kumo_break'] | 
                              df['trailing_stop_long1'] | df['trailing_stop_long2'] | df['trailing_stop_long3'] | df['trailing_stop_long4'] 
                            )

    
    df['exit_long_final'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [   
            df['exit_long_price_turn']| df['exit_long_avoid'] | ((df['trailing_stop_long4']| df['trailing_stop_long2'])& (~df['entry_long_avoid'])& (~df['entry_short_avoid'])) ,  #
            df['exit_long_price_turn']| df['exit_long_price_turn1']| df['exit_long_price_drop']|  ((df['exit_long_tkcross']|  df['exit_long_price_down'])& (~df['entry_long_avoid1'])& (~df['entry_short_avoid'])) ,    ###df['exit_long_price_below_cloud']| df['trailing_stop_long1']  |
            df['exit_long_price_reversion'] | ((df['trailing_stop_long1'] | df['exit_long_price_cloud']| df['exit_long_kijun1'])) ,  ###df['exit_long_avoid']|  df['exit_long_tenkan'] | df['exit_long_tkcross']  | 
            df['exit_long_price_turn']| ((df['trailing_stop_long4'] | df['exit_long_price_cloud'] | df['exit_long_price_down'] | df['exit_long_kijun'] | df['trailing_stop_long3']| df['exit_long_kumo_break'])& (~df['entry_short_avoid1']))    ###df['exit_long_price_drop'] |
        ],
        default= df['entry_short_above_cloud'] | df['exit_long_price_turn']| df['exit_long_avoid']|  ((df['exit_long_tkcross1'] | df['exit_long_tkcross'] | df['trailing_stop_long4'] | df['trailing_stop_long3'] )& (~df['entry_long_avoid']))     ###df['exit_long_kumo_break']  |
    )

    final_unique_entry_short = (df['entry_short_kumo_break'] | df['entry_gap_short'] | df['entry_pullback_short'] | 
                                df['entry_short_oi_vol'] | df['entry_short_oi_vol1'] | df['entry_short_above_cloud1'] | df['entry_short_above_cloud'] | 
                                df['trailing_entry_short0'] | df['trailing_entry_short1'] | df['trailing_entry_short2'] | df['trailing_entry_short3'])


    df['final_entry_short'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [   
            ((df['entry_short_kumo_break'] |  df['trailing_entry_short0'] )& (~df['entry_short_avoid'])) ,    ###df['entry_short_price'] | df['entry_short_kumo_break'] |  
            df['entry_gap_short']  | ((df['trailing_entry_short1'] |df['trailing_entry_short0'] | df['entry_short_above_cloud1']) & (~df['entry_short_avoid'])) ,    ####df['exit_long_price_down'] | 
            ((df['trailing_stop_long2'] | df['exit_long_tkcross'] | df['trailing_entry_short3'] |  df['entry_short_oi_vol1'])& (~df['entry_short_avoid'])& (~df['entry_long_avoid'])),    ####df['entry_gap_short']  |
            ((df['entry_pullback_short'] |  df['trailing_entry_short2'] | df['entry_short_oi_vol']  | df['trailing_stop_long5'] | df['entry_short_above_cloud']))     ##df['entry_short_kumo_break1'] |
        ],
        default=  ((df['exit_long_price_down'] | df['exit_long_price_drop'] | df['entry_pullback_short'] | df['entry_short_kumo_break'] | df['trailing_entry_short1']| df['entry_short_above_cloud']))
    )
    # df['final_entry_short'] = False

    final_unique_exit_short = (df['exit_short_price_cloud'] | df['exit_short_price_cloud1'] | df['exit_short_price_turn'] | 
                               df['exit_short_avoid'] | df['exit_short_high'] | df['exit_short_cloud_tenkan'] | df['exit_short_tkcross'] | 
                               df['exit_short_kijun'] | df['exit_short_tenkan'] | df['exit_short_above_cloud'] |
                               df['trailing_stop_short'] | df['trailing_stop_short1'] | df['trailing_entry_long2'] | df['trailing_stop_short3']
    ) 
    
    
    df['final_exit_short'] = np.select(
        [
            df['Market_Regime'] == 'Strong Trend',
            df['Market_Regime'] == 'Weak Trend / Possible Breakout',
            df['Market_Regime'] == 'Range / Mean Reversion',
            df['Market_Regime'] == 'Volatility Breakout / Regime Shift'
        ],
        [

            ((df['trailing_entry_long0'] | df['entry_long_below_cloud'] | df['exit_short_price_cloud1']  | df['trailing_stop_short'] )& (~df['entry_short_avoid'])) ,
            ((df['exit_short_avoid'] | df['exit_short_price_cloud'] | df['exit_short_tenkan'] | df['exit_short_above_cloud'])& (~df['entry_short_avoid'])), # | df['exit_short_kijun'] df['exit_short_tkcross'] | 
            df['exit_short_price_turn']| df['exit_short_avoid']|  ((  df['exit_short_kijun']  | df['exit_short_tenkan'])& (~df['entry_short_avoid'])& (~df['entry_long_avoid'])) ,#   ###df['exit_short_above_cloud'] |df['exit_short_tkcross']| df['exit_short_price_cloud1']    |
            ((df['exit_short_price_cloud']   | df['entry_long_price'] |  df['exit_short_high'] | df['exit_short_cloud_tenkan'] )& (~df['entry_long_avoid']))# | df['exit_short_kijun'] |  df['exit_short_above_cloud'] || df['exit_short_kumo_break']    ###df['exit_short_tkcross'] | df['trailing_stop_short1'] |
        ],
        default = df['exit_short_avoid'] | df['exit_short_tenkan']| df['trailing_stop_short1'] | ((df['exit_short_tkcross'] |  df['exit_short_price_cloud1']   | df['trailing_stop_short3']))  ##df['trailing_entry_long2'] | 
    )
    
    
    
    # df['exit_long'] | df['exit_long_below_cloud'] | df['exit_long_tkcross'] | df['exit_long_price_cloud'] | df['exit_long_tenkan'] | df['exit_long_kijun'] | df['trailing_stop_long']
    # df['exit_short'] | df['exit_short_above_cloud'] | df['exit_short_tkcross'] | df['exit_short_price_cloud'] | df['exit_short_tenkan'] | df['exit_short_kijun'] | df['trailing_stop_short']
    
    # df.to_csv('strategy_file.csv', index=False)
    return df



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