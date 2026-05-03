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
    

def valuation_analysis(result):
    
    pe = safe(result.get("trailingPE"))
    fpe = safe(result.get("forwardPE"))
    peg = safe(result.get("pegRatio"))
    pb = safe(result.get("priceToBook"))
    ev_ebitda = safe(result.get("enterpriseToEbitda"))
    ev_rev = safe(result.get("enterpriseToRevenue"))

    # --- Individual Scores ---
    low_pe_score = score_low_better(pe, ideal=15, max_val=40)
    low_fpe_score = score_low_better(fpe, ideal=15, max_val=40)
    low_pb_score = score_low_better(pb, ideal=1.5, max_val=6)
    low_ev_ebitda_score = score_low_better(ev_ebitda, ideal=10, max_val=30)
    low_ev_rev_score = score_low_better(ev_rev, ideal=2, max_val=10)
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
        "label": label,
        "components": {
            "pe": low_pe_score,
            "forward_pe": low_fpe_score,
            "pb": low_pb_score,
            "ev_ebitda": low_ev_ebitda_score,
            "ev_revenue": low_ev_rev_score,
            "peg": low_peg_score
        }
    }