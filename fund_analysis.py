def safe(x, default=None):
    return x if x is not None else default


def score_low_better(value, ideal, max_val):
    """
    Lower value is better (PE, PB, EV/EBITDA)
    """
    if value is None:
        return 0
    if value <= ideal:
        return 1
    if value >= max_val:
        return 0
    return (max_val - value) / (max_val - ideal)


def score_peg(value):
    """
    PEG specific logic
    """
    if value is None:
        return 0
    if value < 1:
        return 1
    elif value < 2:
        return 0.5
    else:
        return 0

def score_growth(value):
    """
    Growth metrics are usually in decimal (e.g., 0.15 = 15%)
    """
    if value is None:
        return 0
    if value >= 0.20:   # 20%+
        return 1
    elif value >= 0.10:
        return 0.7
    elif value > 0:
        return 0.4
    else:
        return 0
    
def eps_growth_score(forward_eps, current_eps):
    if forward_eps is None or current_eps is None or current_eps == 0:
        return 0
    
    growth = (forward_eps - current_eps) / abs(current_eps)

    return score_growth(growth)

def peg_adjustment(pe, growth):
    """
    Approx PEG logic using PE and earnings growth
    """
    if pe is None or growth is None or growth <= 0:
        return 0
    
    peg = pe / (growth * 100)  # growth assumed decimal

    if peg < 1:
        return 1
    elif peg < 2:
        return 0.5
    else:
        return 0

def score_margin(value):
    """
    value is in decimal (e.g., 0.25 = 25%)
    """
    if value is None:
        return 0
    if value >= 0.30:
        return 1
    elif value >= 0.20:
        return 0.8
    elif value >= 0.10:
        return 0.6
    elif value > 0:
        return 0.3
    else:
        return 0
    
def score_roe(value):
    if value is None:
        return 0
    if value >= 0.20:   # 20%+
        return 1
    elif value >= 0.15:
        return 0.8
    elif value >= 0.10:
        return 0.6
    elif value > 0:
        return 0.3
    else:
        return 0


def score_roa(value):
    if value is None:
        return 0
    if value >= 0.10:
        return 1
    elif value >= 0.07:
        return 0.8
    elif value >= 0.04:
        return 0.6
    elif value > 0:
        return 0.3
    else:
        return 0
    

def score_debt_to_equity(value):
    if value is None:
        return 0
    
    if value <= 0.5:
        return 1      # very safe
    elif value <= 1:
        return 0.8
    elif value <= 2:
        return 0.5
    elif value <= 3:
        return 0.2
    else:
        return 0      # high risk

def score_current_ratio(value):
    if value is None:
        return 0
    
    if value >= 2:
        return 1
    elif value >= 1.5:
        return 0.8
    elif value >= 1:
        return 0.5
    else:
        return 0   # liquidity risk

def score_quick_ratio(value):
    if value is None:
        return 0
    
    if value >= 1.5:
        return 1
    elif value >= 1:
        return 0.8
    elif value >= 0.7:
        return 0.5
    else:
        return 0

def score_total_debt(total_debt, market_cap):
    if total_debt is None or market_cap is None or market_cap == 0:
        return 0
    
    ratio = total_debt / market_cap

    if ratio < 0.2:
        return 1
    elif ratio < 0.5:
        return 0.7
    elif ratio < 1:
        return 0.4
    else:
        return 0

def calculate_upside(target_mean, current_price):
    if target_mean is None or current_price is None or current_price == 0:
        return 0
    return (target_mean - current_price) / current_price

def score_upside(upside):
    """
    upside is decimal (e.g., 0.25 = 25%)
    """
    if upside >= 0.30:
        return 1
    elif upside >= 0.15:
        return 0.8
    elif upside >= 0.05:
        return 0.6
    elif upside > 0:
        return 0.4
    else:
        return 0   # negative upside
    
def score_recommendation(rec):
    if rec is None:
        return 0

    rec = rec.lower()

    mapping = {
        "strong_buy": 1,
        "buy": 0.8,
        "outperform": 0.7,
        "hold": 0.5,
        "neutral": 0.5,
        "underperform": 0.2,
        "sell": 0
    }

    return mapping.get(rec, 0.5)

def score_target_range(high, low, current_price):
    if high is None or low is None or current_price is None:
        return 0
    
    spread = (high - low) / current_price

    # smaller spread = higher confidence
    if spread < 0.2:
        return 1
    elif spread < 0.4:
        return 0.7
    else:
        return 0.4
    

def valuation_analysis(result):
    
    # pe = safe(result.get("trailingPE"))
    pe = float(result['trailingPE'])
    fpe = safe(result.get("forwardPE"))
    peg = safe(result.get("pegRatio"))
    pb = safe(result.get("priceToBook"))
    ev_ebitda = safe(result.get("enterpriseToEbitda"))
    ev_rev = safe(result.get("enterpriseToRevenue"))

    # --- Individual Scores ---
    low_pe_score = score_low_better(pe, ideal=15, max_val=100)
    low_fpe_score = score_low_better(fpe, ideal=15, max_val=100)
    low_pb_score = score_low_better(pb, ideal=1.5, max_val=50)
    low_ev_ebitda_score = score_low_better(ev_ebitda, ideal=10, max_val=100)
    low_ev_rev_score = score_low_better(ev_rev, ideal=2, max_val=20)
    low_peg_score = score_peg(peg)

    # --- Final Score ---
    valuation_score = (
        low_pe_score +
        low_fpe_score +
        low_pb_score +
        low_ev_ebitda_score +
        low_ev_rev_score +
        low_peg_score
    ) / 6  # normalize to 0–1

    # --- Interpretation ---
    if valuation_score > 0.7:
        label = "Undervalued"
    elif valuation_score > 0.4:
        label = "Fairly Valued"
    else:
        label = "Overvalued"

    return {
        "valuation_score": round(valuation_score, 2),
        "valuation_label": label,
        "val_components": {
            "pe": pe,
            "forward_pe": fpe,
            "pb": pb,
            "ev_ebitda": ev_ebitda,
            "ev_revenue": ev_rev,
            "peg": peg
        }
    }

def growth_analysis(result):

    revenue_growth = safe(result.get("revenueGrowth"))
    earnings_growth = safe(result.get("earningsGrowth"))
    quarterly_growth = safe(result.get("earningsQuarterlyGrowth"))

    forward_eps = safe(result.get("forwardEPS"))
    current_eps = safe(result.get("epsCurrentYear"))

    pe = safe(result.get("trailingPE"))

    # --- Individual Scores ---
    revenue_score = score_growth(revenue_growth)
    earnings_score = score_growth(earnings_growth)
    quarterly_score = score_growth(quarterly_growth)
    eps_score = eps_growth_score(forward_eps, current_eps)

    # PEG logic using earnings growth
    peg_score = peg_adjustment(pe, earnings_growth)

    # --- Final Growth Score ---
    growth_score = (
        revenue_score +
        earnings_score +
        quarterly_score +
        eps_score +
        peg_score
    ) / 5

    # --- Label ---
    if growth_score > 0.7:
        label = "High Growth"
    elif growth_score > 0.4:
        label = "Moderate Growth"
    else:
        label = "Low Growth"

    return {
        "growth_score": round(growth_score, 2),
        "growth_label": label,
        "growth_components": {
            "revenue_growth": revenue_score,
            "earnings_growth": earnings_score,
            "quarterly_growth": quarterly_score,
            "eps_growth": eps_score,
            "peg_factor": peg_score
        }
    }

def profitability_analysis(result):

    profit_margin = safe(result.get("profitMargins"))
    gross_margin = safe(result.get("grossMargins"))
    operating_margin = safe(result.get("operatingMargins"))
    ebitda_margin = safe(result.get("ebitdaMargins"))

    roe = safe(result.get("returnOnEquity"))
    roa = safe(result.get("returnOnAssets"))

    # --- Individual Scores ---
    profit_score = score_margin(profit_margin)
    gross_score = score_margin(gross_margin)
    operating_score = score_margin(operating_margin)
    ebitda_score = score_margin(ebitda_margin)

    roe_score = score_roe(roe)
    roa_score = score_roa(roa)

    # --- Final Score ---
    profitability_score = (
        profit_score +
        gross_score +
        operating_score +
        ebitda_score +
        roe_score +
        roa_score
    ) / 6

    # --- Label ---
    if profitability_score > 0.7:
        label = "High Quality Business"
    elif profitability_score > 0.4:
        label = "Average Quality"
    else:
        label = "Low Quality"

    return {
        "profitability_score": round(profitability_score, 2),
        "profitability_label": label,
        "prof_components": {
            "profit_margin": profit_score,
            "gross_margin": gross_score,
            "operating_margin": operating_score,
            "ebitda_margin": ebitda_score,
            "roe": roe_score,
            "roa": roa_score
        }
    }

def financial_health_analysis(result):

    debt_to_equity = safe(result.get("debtToEquity"))
    total_debt = safe(result.get("totalDebt"))
    current_ratio = safe(result.get("currentRatio"))
    quick_ratio = safe(result.get("quickRatio"))
    market_cap = safe(result.get("marketCap"))

    # --- Individual Scores ---
    debt_score = score_debt_to_equity(debt_to_equity)
    current_score = score_current_ratio(current_ratio)
    quick_score = score_quick_ratio(quick_ratio)
    total_debt_score = score_total_debt(total_debt, market_cap)

    # --- Final Risk Score ---
    risk_score = (
        debt_score +
        current_score +
        quick_score +
        total_debt_score
    ) / 4

    # --- Label ---
    if risk_score > 0.7:
        label = "Low Risk (Strong Balance Sheet)"
    elif risk_score > 0.4:
        label = "Moderate Risk"
    else:
        label = "High Risk"

    return {
        "risk_score": round(risk_score, 2),
        "risk_label": label,
        "health_components": {
            "debt_to_equity": debt_score,
            "current_ratio": current_score,
            "quick_ratio": quick_score,
            "total_debt": total_debt_score
        }
    }

def sentiment_analysis(result, current_price):

    target_high = safe(result.get("targetHighPrice"))
    target_mean = safe(result.get("targetMeanPrice"))
    target_low = safe(result.get("targetLowPrice"))
    recommendation = safe(result.get("recommendationKey"))

    # --- Core calculations ---
    upside = calculate_upside(target_mean, current_price)

    upside_score = score_upside(upside)
    rec_score = score_recommendation(recommendation)
    range_score = score_target_range(target_high, target_low, current_price)

    # --- Final Score ---
    sentiment_score = (
        upside_score * 0.5 +
        rec_score * 0.3 +
        range_score * 0.2
    )

    # --- Label ---
    if sentiment_score > 0.7:
        label = "Bullish"
    elif sentiment_score > 0.4:
        label = "Neutral"
    else:
        label = "Bearish"

    return {
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_label": label,
        "upside": round(upside, 2),
        "sent_components": {
            "upside": upside_score,
            "recommendation": rec_score,
            "confidence": range_score
        }
    }

def composite_score(result):

    v = result.get("valuation_score", 0)
    g = result.get("growth_score", 0)
    p = result.get("profitability_score", 0)
    r = result.get("risk_score", 0)
    s = result.get("sentiment_score", 0)

    # --- Weights (balanced model) ---
    composite = (
        v * 0.25 +   # valuation
        g * 0.25 +   # growth
        p * 0.20 +   # quality
        r * 0.15 +   # risk
        s * 0.15     # sentiment
    )

    # --- Label ---
    if composite > 0.75:
        label = "Strong Buy"
    elif composite > 0.6:
        label = "Buy"
    elif composite > 0.45:
        label = "Hold"
    else:
        label = "Avoid"

    return {
        "composite_score": round(composite, 2),
        "composite_label": label
    }


def fundamental_analysis(symbol, info):

    result = {
        'industry': info.get("industry", None),
        'sector': info.get("sector", None),
        'business': info.get("longBusinessSummary", None),
        'dividendYield': info.get("dividendYield", None),
        'payoutRatio': info.get("payoutRatio", None),
        'beta': info.get("beta", None),
        'trailingPE': info.get("trailingPE", None),
        'forwardPE': info.get("forwardPE", None),
        'trailingEPS': info.get("trailingEps", None),
        'forwardEPS': info.get("forwardEps", None),
        'epsTrailingTwelveMonths': info.get("epsTrailingTwelveMonths", None),
        'epsForward': info.get("epsForward", None),
        'epsCurrentYear': info.get("epsCurrentYear", None),
        'pegRatio': info.get("pegRatio", None),
        'marketCap': info.get("marketCap", None),   #
        'enterpriseValue': info.get("enterpriseValue", None), #
        'profitMargins': info.get("profitMargins", None), ##
        'bookValue': info.get("bookValue", None),
        'priceToBook': info.get("priceToBook", None),
        'earningsQuarterlyGrowth': info.get("earningsQuarterlyGrowth", None),
        'enterpriseToRevenue': info.get("enterpriseToRevenue", None),
        'enterpriseToEbitda': info.get("enterpriseToEbitda", None),
        'targetHighPrice': info.get("targetHighPrice", None), #
        'targetLowPrice': info.get("targetLowPrice", None), ###
        'targetMeanPrice': info.get("targetMeanPrice", None),  ##
        'recommendationKey': info.get("recommendationKey", None),  ##
        'totalCashPerShare': info.get("totalCashPerShare", None),
        'ebitda': info.get("ebitda", None),
        'totalRevenue': info.get("totalRevenue", None),
        'totalDebt': info.get("totalDebt", None),
        'quickRatio': info.get("quickRatio", None),  ##
        'currentRatio': info.get("currentRatio", None),  ##
        'debtToEquity': info.get("debtToEquity", None),  ##
        'revenuePerShare': info.get("revenuePerShare", None),
        'returnOnAssets': info.get("returnOnAssets", None),
        'returnOnEquity': info.get("returnOnEquity", None),
        'grossProfits': info.get("grossProfits", None),  
        'freeCashflow': info.get("freeCashflow", None),
        'operatingCashflow': info.get("operatingCashflow", None),
        'earningsGrowth': info.get("earningsGrowth", None),
        'revenueGrowth': info.get("revenueGrowth", None),
        'grossMargins': info.get("grossMargins", None),  ##
        'ebitdaMargins': info.get("ebitdaMargins", None),  ##
        'operatingMargins': info.get("operatingMargins", None),  ##
        'customPriceAlertConfidence': info.get("customPriceAlertConfidence", None),
        'fiftyTwoWeekRange': info.get("fiftyTwoWeekRange", None),
    }
    return result
