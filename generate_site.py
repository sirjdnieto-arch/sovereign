import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests

try:
    import pandas_datareader.data as web
except Exception:
    web = None

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "LOOKBACK_SIGNAL": 5,
    "PVI_MA": 120,
    "MCG_REGIME_N": 20,
    "MCG_EXIT_N": 45,
    "LATERAL_LOOKBACK": 20,
    "PRICE_PERIOD": "2y",
    "MAX_WORKERS": 8,
}


UNIVERSE = {
    "S&P 500": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "META", "BRK-B", "TSLA",
        "JNJ", "V", "PG", "XOM", "UNH", "JPM", "HD", "LLY", "MA", "CVX",
        "ABBV", "KO", "PEP", "COST", "BAC", "CRM", "NFLX", "ABT", "MCD",
        "LMT", "EL", "NEE", "CAT", "MRK", "TPL"
    ],

    "NASDAQ 100": [
        "ASML", "ADBE", "AVGO", "CSCO", "CMCSA", "AMD", "TXN", "QCOM",
        "AMAT", "INTU", "VRTX", "ZS", "PLTR", "CSU.TO", "MU"
    ],

    "Euro Stoxx 50": [
        "LVMUY", "SAP", "OR.PA", "TTE", "MC.PA", "SIE.DE", "ENGI.PA",
        "AIR.PA", "ALV.DE", "EL.PA", "AI.PA", "BNP.PA", "SAN.PA",
        "KER.PA", "SU.PA", "NESN.SW"
    ],

    "DAX 40": [
        "LIN.DE", "VOW3.DE", "BMW.DE", "ADS.DE", "IFX.DE", "MUV2.DE",
        "FRE.DE", "DTE.DE", "RWE.DE"
    ],

    "IBEX 35": [
        "ITX.MC", "BBVA.MC", "SAN.MC", "TEF.MC", "IBE.MC", "REP.MC",
        "FER.MC", "ACX.MC", "ACS.MC", "AENA.MC", "ANA.MC", "IAG.MC",
        "LOG.MC", "MAP.MC", "PUIG.MC", "NTGY.MC", "ELE.MC", "IDR.MC"
    ],

    "China": [
        "PDD", "NIO", "TCEHY", "BZUN", "FUTU", "MOMO", "MNSO",
        "TAL", "EDU", "WB", "XPEV"
    ],

    "Commodities": [
        "GC=F", "SI=F"
    ],

    "Crypto": [
        "BTC-USD", "ETH-USD", "XRP-USD"
    ]
}


BAD_TICKERS = {
    "GHG", "DSWL", "AHCO", "LFVN", "TCMD", "KWEB", "1810.HK", "1211.HK"
}

TECHNICAL_ONLY_BUCKETS = {
    "Euro Stoxx 50",
    "DAX 40",
    "IBEX 35",
    "Commodities",
    "Crypto",
}

TECHNICAL_ONLY_SUFFIXES = (
    ".PA", ".DE", ".MC", ".SW", ".TO"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(x):
    try:
        if isinstance(x, pd.Series):
            x = x.iloc[0]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[0, 0]
        if x is None:
            return np.nan
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return x
    except Exception:
        return np.nan


def valid_number(*xs):
    return all(np.isfinite(safe_float(x)) for x in xs)


def clean_json_value(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if pd.isna(x):
        return None
    return x


def records_for_json(df):
    if df is None or df.empty:
        return []
    out = []
    for rec in df.to_dict(orient="records"):
        out.append({k: clean_json_value(v) for k, v in rec.items()})
    return out


def build_universe_df():
    rows = []
    seen = set()

    for bucket, tickers in UNIVERSE.items():
        for ticker in tickers:
            if ticker in BAD_TICKERS:
                continue
            if ticker in seen:
                continue

            seen.add(ticker)

            technical_only = (
                bucket in TECHNICAL_ONLY_BUCKETS
                or ticker.endswith("-USD")
                or ticker.endswith("=F")
                or any(ticker.endswith(suf) for suf in TECHNICAL_ONLY_SUFFIXES)
            )

            rows.append({
                "ticker": ticker,
                "bucket": bucket,
                "technical_only": technical_only
            })

    return pd.DataFrame(rows)


def flatten_yf(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def yf_download_prices(ticker, period="2y"):
    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        df = flatten_yf(df)

        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Close", "Volume"]].dropna(subset=["Close"])
        return df

    except Exception:
        return pd.DataFrame()


def get_name_currency_sector(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("longName") or info.get("shortName") or info.get("displayName") or ticker
        currency = info.get("currency") or "—"
        sector = info.get("sector") or "—"
        return name, currency, sector
    except Exception:
        return ticker, "—", "—"


# ============================================================
# TÉCNICO: PVI + MCGINLEY
# ============================================================

def calculate_mcginley(close, period):
    vals = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    if len(vals) == 0:
        return pd.Series(index=close.index, dtype=float)

    mcg = np.zeros(len(vals), dtype=float)
    mcg[0] = vals[0]

    for i in range(1, len(vals)):
        last = mcg[i - 1]
        price = vals[i]

        if last <= 0 or price <= 0:
            mcg[i] = price
            continue

        ratio = price / last
        denom = period * (ratio ** 4)

        if denom == 0 or not np.isfinite(denom):
            mcg[i] = last
        else:
            mcg[i] = last + (price - last) / denom

    return pd.Series(mcg, index=close.index)


def calculate_pvi(df, ma_period):
    df = df.copy()
    df["ROC"] = df["Close"].pct_change().fillna(0)

    vols = df["Volume"].fillna(0).values
    rocs = df["ROC"].fillna(0).values

    pvi = [1000.0]

    for i in range(1, len(df)):
        if vols[i] > vols[i - 1]:
            pvi.append(pvi[-1] * (1 + rocs[i]))
        else:
            pvi.append(pvi[-1])

    df["PVI"] = pvi
    df["PVI_Signal"] = df["PVI"].rolling(ma_period, min_periods=ma_period).mean()

    return df


def bars_ago_for_signal(signal_series, lookback):
    tail = signal_series.tail(lookback)
    idx = np.where(tail.fillna(False).values)[0]

    if len(idx) == 0:
        return None

    last_pos = idx[-1]
    return int(len(tail) - 1 - last_pos)


def ago_txt(n):
    if n is None or pd.isna(n):
        return "—"
    n = int(n)
    if n == 0:
        return "vela actual"
    if n == 1:
        return "hace 1 vela"
    return f"hace {n} velas"


def analyze_technical(row):
    ticker = row["ticker"]
    bucket = row["bucket"]

    name, currency, sector = get_name_currency_sector(ticker)
    df = yf_download_prices(ticker, CONFIG["PRICE_PERIOD"])

    if df.empty or len(df) < CONFIG["PVI_MA"] + 10:
        return {
            "ticker": ticker,
            "name": name,
            "bucket": bucket,
            "currency": currency,
            "sector": sector,
            "technical_only": bool(row["technical_only"]),
            "has_signal": False,
            "error": "Sin datos suficientes"
        }

    df = calculate_pvi(df, CONFIG["PVI_MA"])
    df["McG_Regime"] = calculate_mcginley(df["Close"], CONFIG["MCG_REGIME_N"])
    df["McG_Exit"] = calculate_mcginley(df["Close"], CONFIG["MCG_EXIT_N"])

    df["PVI_Cross_Up"] = (
        (df["PVI"] > df["PVI_Signal"])
        & (df["PVI"].shift(1) <= df["PVI_Signal"].shift(1))
    )

    df["PVI_Cross_Down"] = (
        (df["PVI"] < df["PVI_Signal"])
        & (df["PVI"].shift(1) >= df["PVI_Signal"].shift(1))
    )

    df["McG_Cross_Down"] = (
        (df["Close"] < df["McG_Exit"])
        & (df["Close"].shift(1) >= df["McG_Exit"].shift(1))
    )

    buy_ago = bars_ago_for_signal(df["PVI_Cross_Up"], CONFIG["LOOKBACK_SIGNAL"])
    pvi_sell_ago = bars_ago_for_signal(df["PVI_Cross_Down"], CONFIG["LOOKBACK_SIGNAL"])
    mcg_sell_ago = bars_ago_for_signal(df["McG_Cross_Down"], CONFIG["LOOKBACK_SIGNAL"])

    has_buy = buy_ago is not None
    has_pvi_sell = pvi_sell_ago is not None
    has_mcg_sell = mcg_sell_ago is not None

    above = df["Close"] > df["McG_Regime"]
    recent_crosses = int(
        above.tail(CONFIG["LATERAL_LOOKBACK"])
        .astype(int)
        .diff()
        .abs()
        .fillna(0)
        .sum()
    )

    c = df.iloc[-1]

    close_now = safe_float(c["Close"])
    mcg_regime_now = safe_float(c["McG_Regime"])
    mcg_exit_now = safe_float(c["McG_Exit"])
    pvi_now = safe_float(c["PVI"])
    pvi_sig_now = safe_float(c["PVI_Signal"])

    if recent_crosses > 2:
        regime = "🟡 LATERAL"
    elif close_now > mcg_regime_now:
        regime = "🟢 ALCISTA"
    else:
        regime = "🔴 BAJISTA"

    events = []

    if has_buy:
        events.append(("🟢 COMPRA", buy_ago))

    if has_pvi_sell:
        events.append(("🟠 VENTA 50% PVI", pvi_sell_ago))

    if has_mcg_sell:
        events.append(("🟠 VENTA 50% McGINLEY", mcg_sell_ago))

    if has_buy and (has_pvi_sell or has_mcg_sell):
        main_signal = "⚠️ MIXTA"
    elif has_buy:
        main_signal = "🟢 COMPRA"
    elif has_pvi_sell and has_mcg_sell:
        main_signal = "🔴 VENTA 100%"
    elif has_pvi_sell:
        main_signal = "🟠 VENTA 50% PVI"
    elif has_mcg_sell:
        main_signal = "🟠 VENTA 50% McGINLEY"
    else:
        main_signal = "—"

    bars_min = min([e[1] for e in events], default=None)
    events_text = " · ".join([f"{name_} ({ago_txt(ago)})" for name_, ago in events])

    dist_to_mcg_exit = np.nan
    if valid_number(close_now, mcg_exit_now) and mcg_exit_now != 0:
        dist_to_mcg_exit = close_now / mcg_exit_now - 1

    return {
        "ticker": ticker,
        "name": name,
        "bucket": bucket,
        "currency": currency,
        "sector": sector,
        "technical_only": bool(row["technical_only"]),
        "has_signal": bool(events),
        "main_signal": main_signal,
        "events_text": events_text,
        "buy_ago": buy_ago,
        "pvi_sell_ago": pvi_sell_ago,
        "mcg_sell_ago": mcg_sell_ago,
        "bars_min": bars_min,
        "regime": regime,
        "recent_crosses": recent_crosses,
        "close": close_now,
        "mcg_regime": mcg_regime_now,
        "mcg_exit": mcg_exit_now,
        "dist_to_mcg_exit": dist_to_mcg_exit,
        "pvi": pvi_now,
        "pvi_signal": pvi_sig_now,
        "pvi_status": "POSITIVO" if pvi_now > pvi_sig_now else "NEGATIVO",
        "price_below_mcg_exit": bool(close_now < mcg_exit_now) if valid_number(close_now, mcg_exit_now) else False,
        "last_date": str(df.index[-1].date()),
        "error": ""
    }


# ============================================================
# FUNDAMENTALES
# ============================================================

def get_statement(ticker_obj, names):
    for name in names:
        try:
            df = getattr(ticker_obj, name)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def get_row(df, aliases, col=0):
    if df is None or df.empty or len(df.columns) <= col:
        return np.nan

    for alias in aliases:
        if alias in df.index:
            try:
                obj = df.loc[alias]
                if isinstance(obj, pd.DataFrame):
                    return safe_float(obj.iloc[0, col])
                return safe_float(obj.iloc[col])
            except Exception:
                continue

    return np.nan


def get_ttm(df, aliases, fallback_annual_df=None, min_quarters=4):
    if df is not None and not df.empty:
        for alias in aliases:
            if alias in df.index:
                try:
                    obj = df.loc[alias]
                    vals = obj.iloc[:4] if not isinstance(obj, pd.DataFrame) else obj.iloc[0, :4]
                    vals = pd.to_numeric(vals, errors="coerce").dropna()
                    if len(vals) >= min_quarters:
                        return safe_float(vals.sum())
                except Exception:
                    pass

    if fallback_annual_df is not None and not fallback_annual_df.empty:
        return get_row(fallback_annual_df, aliases, col=0)

    return np.nan


def piotroski_annual(is_a, bs_a, cf_a):
    if is_a.empty or bs_a.empty or cf_a.empty or len(is_a.columns) < 2 or len(bs_a.columns) < 2:
        return np.nan

    points = 0
    possible = 0

    def add(condition):
        nonlocal points, possible
        if condition is None:
            return
        possible += 1
        if bool(condition):
            points += 1

    ni0 = get_row(is_a, ["Net Income", "Net Income Common Stockholders"], 0)
    ni1 = get_row(is_a, ["Net Income", "Net Income Common Stockholders"], 1)
    cfo0 = get_row(cf_a, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)

    ta0 = get_row(bs_a, ["Total Assets"], 0)
    ta1 = get_row(bs_a, ["Total Assets"], 1)

    ltd0 = get_row(bs_a, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Total Debt"], 0)
    ltd1 = get_row(bs_a, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Total Debt"], 1)

    ca0 = get_row(bs_a, ["Current Assets", "Total Current Assets"], 0)
    ca1 = get_row(bs_a, ["Current Assets", "Total Current Assets"], 1)

    cl0 = get_row(bs_a, ["Current Liabilities", "Total Current Liabilities"], 0)
    cl1 = get_row(bs_a, ["Current Liabilities", "Total Current Liabilities"], 1)

    gp0 = get_row(is_a, ["Gross Profit"], 0)
    gp1 = get_row(is_a, ["Gross Profit"], 1)

    rev0 = get_row(is_a, ["Total Revenue", "Operating Revenue"], 0)
    rev1 = get_row(is_a, ["Total Revenue", "Operating Revenue"], 1)

    shares0 = get_row(bs_a, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 0)
    shares1 = get_row(bs_a, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 1)

    if valid_number(ni0):
        add(ni0 > 0)
    if valid_number(cfo0):
        add(cfo0 > 0)
    if valid_number(ni0, ni1, ta0, ta1) and ta0 != 0 and ta1 != 0:
        add((ni0 / ta0) > (ni1 / ta1))
    if valid_number(cfo0, ni0):
        add(cfo0 > ni0)
    if valid_number(ltd0, ltd1, ta0, ta1) and ta0 != 0 and ta1 != 0:
        add((ltd0 / ta0) < (ltd1 / ta1))
    if valid_number(ca0, ca1, cl0, cl1) and cl0 != 0 and cl1 != 0:
        add((ca0 / cl0) > (ca1 / cl1))
    if valid_number(shares0, shares1):
        add(shares0 <= shares1)
    if valid_number(gp0, gp1, rev0, rev1) and rev0 != 0 and rev1 != 0:
        add((gp0 / rev0) > (gp1 / rev1))
    if valid_number(rev0, rev1, ta0, ta1) and ta0 != 0 and ta1 != 0:
        add((rev0 / ta0) > (rev1 / ta1))

    if possible == 0:
        return np.nan

    return 9 * points / possible


def score_var(val, threshold_high, threshold_low, reverse=False):
    if pd.isna(val):
        return 0

    if not reverse:
        if val > threshold_high:
            return 0.5
        if val < threshold_low:
            return -0.5
        return 0

    if val < threshold_high:
        return 0.5
    if val > threshold_low:
        return -0.5
    return 0


def get_fundamental_raw(ticker):
    try:
        t = yf.Ticker(ticker)

        try:
            info = t.info or {}
        except Exception:
            info = {}

        bs_a = get_statement(t, ["balance_sheet", "balancesheet"])
        is_a = get_statement(t, ["income_stmt", "financials"])
        cf_a = get_statement(t, ["cashflow"])

        bs_q = get_statement(t, ["quarterly_balance_sheet", "quarterly_balancesheet"])
        is_q = get_statement(t, ["quarterly_income_stmt", "quarterly_financials"])
        cf_q = get_statement(t, ["quarterly_cashflow"])

        if bs_a.empty and bs_q.empty:
            return None

        current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
        target = safe_float(info.get("targetMeanPrice"))

        upside = np.nan
        if valid_number(current_price, target) and current_price > 0 and target > 0:
            upside = target / current_price - 1

        revenue_ttm = get_ttm(is_q, ["Total Revenue", "Operating Revenue"], is_a)
        op_income_ttm = get_ttm(is_q, ["Operating Income"], is_a)
        net_income_ttm = get_ttm(is_q, ["Net Income", "Net Income Common Stockholders"], is_a)
        tax_ttm = get_ttm(is_q, ["Tax Provision"], is_a)
        pretax_ttm = get_ttm(is_q, ["Pretax Income", "Income Before Tax"], is_a)
        interest_expense_ttm = get_ttm(is_q, ["Interest Expense", "Interest Expense Non Operating"], is_a)
        cfo_ttm = get_ttm(cf_q, ["Operating Cash Flow", "Total Cash From Operating Activities"], cf_a)

        ebitda_ttm = get_ttm(is_q, ["EBITDA", "Normalized EBITDA"], is_a)
        if pd.isna(ebitda_ttm):
            ebitda_ttm = safe_float(info.get("ebitda"))

        bs_ref = bs_q if not bs_q.empty else bs_a

        total_assets = get_row(bs_ref, ["Total Assets"], 0)
        current_assets = get_row(bs_ref, ["Current Assets", "Total Current Assets"], 0)
        current_liabilities = get_row(bs_ref, ["Current Liabilities", "Total Current Liabilities"], 0)
        total_liabilities = get_row(bs_ref, ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"], 0)

        equity = get_row(
            bs_ref,
            ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
            0
        )

        if pd.isna(equity) and valid_number(total_assets, total_liabilities):
            equity = total_assets - total_liabilities

        retained_earnings = get_row(bs_ref, ["Retained Earnings"], 0)

        total_debt = get_row(bs_ref, ["Total Debt"], 0)
        if pd.isna(total_debt):
            lt_debt = get_row(bs_ref, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
            st_debt = get_row(bs_ref, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], 0)
            total_debt = np.nansum([lt_debt, st_debt])

        cash = get_row(bs_ref, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], 0)
        if pd.isna(cash):
            cash = 0

        tax_rate = 0.21
        if valid_number(tax_ttm, pretax_ttm) and pretax_ttm > 0:
            tax_rate = np.clip(tax_ttm / pretax_ttm, 0, 0.35)

        nopat = op_income_ttm * (1 - tax_rate) if valid_number(op_income_ttm) else np.nan

        invested_capital = np.nan
        if valid_number(equity, total_debt):
            invested_capital = equity + total_debt - cash

        roic = np.nan
        if valid_number(nopat, invested_capital) and invested_capital > 0:
            roic = nopat / invested_capital

        pb = safe_float(info.get("priceToBook"))

        debt_ebitda = np.nan
        if valid_number(total_debt, ebitda_ttm) and ebitda_ttm > 0:
            debt_ebitda = total_debt / ebitda_ttm
        else:
            debt_ebitda = safe_float(info.get("debtToEbitda"))

        op_margin = np.nan
        if valid_number(op_income_ttm, revenue_ttm) and revenue_ttm != 0:
            op_margin = op_income_ttm / revenue_ttm

        asset_turnover = np.nan
        if valid_number(revenue_ttm, total_assets) and total_assets != 0:
            asset_turnover = revenue_ttm / total_assets

        current_ratio = np.nan
        if valid_number(current_assets, current_liabilities) and current_liabilities != 0:
            current_ratio = current_assets / current_liabilities

        cash_quality = np.nan
        if valid_number(cfo_ttm, net_income_ttm) and abs(net_income_ttm) > 1e-9:
            cash_quality = cfo_ttm / net_income_ttm

        interest_coverage = np.nan
        if valid_number(op_income_ttm, interest_expense_ttm):
            if abs(interest_expense_ttm) < 1e-9:
                interest_coverage = 10
            else:
                interest_coverage = op_income_ttm / abs(interest_expense_ttm)

        insider_pct = safe_float(info.get("heldPercentInsiders"))

        market_cap = safe_float(info.get("marketCap"))

        altman_z = np.nan
        if valid_number(total_assets, current_assets, current_liabilities, op_income_ttm, market_cap, total_liabilities, revenue_ttm):
            if total_assets != 0 and total_liabilities != 0:
                wc = current_assets - current_liabilities
                re = retained_earnings if valid_number(retained_earnings) else 0
                altman_z = (
                    1.2 * (wc / total_assets)
                    + 1.4 * (re / total_assets)
                    + 3.3 * (op_income_ttm / total_assets)
                    + 0.6 * (market_cap / total_liabilities)
                    + 1.0 * (revenue_ttm / total_assets)
                )

        piotroski = piotroski_annual(is_a, bs_a, cf_a)

        return {
            "ticker": ticker,
            "upside": upside,
            "pb": pb,
            "debt_ebitda": debt_ebitda,
            "altman_z": altman_z,
            "op_margin": op_margin,
            "asset_turnover": asset_turnover,
            "roic": roic,
            "piotroski": piotroski,
            "current_ratio": current_ratio,
            "cash_quality": cash_quality,
            "interest_coverage": interest_coverage,
            "insider_pct": insider_pct,
        }

    except Exception:
        return None


def score_fundamental(d):
    neutral = {
        "upside": 0.0,
        "pb": 3.0,
        "debt_ebitda": 2.5,
        "altman_z": 2.4,
        "op_margin": 0.10,
        "asset_turnover": 0.50,
        "roic": 0.08,
        "piotroski": 5.0,
        "current_ratio": 1.2,
        "cash_quality": 1.0,
        "interest_coverage": 3.0,
        "insider_pct": 0.02,
    }

    imputed = 0

    for k, v in neutral.items():
        if k not in d or pd.isna(d[k]):
            d[k] = v
            imputed += 1

    c_valor = (
        score_var(d["upside"], 0.20, 0)
        + score_var(d["pb"], 1.5, 6, reverse=True)
    )

    a_solvencia = (
        score_var(d["debt_ebitda"], 1.5, 4, reverse=True)
        + score_var(d["altman_z"], 3.0, 1.8)
    )

    m_margen = (
        score_var(d["op_margin"], 0.15, 0.05)
        + score_var(d["asset_turnover"], 0.8, 0.3)
    )

    e_eficiencia = (
        score_var(d["roic"], 0.15, 0.05)
        + score_var(d["piotroski"], 7, 3)
    )

    l_liquidez = (
        score_var(d["current_ratio"], 2.0, 0.8)
        + score_var(d["cash_quality"], 1.2, 0.8)
    )

    s_skin = (
        score_var(d["interest_coverage"], 6.0, 1.5)
        + score_var(d["insider_pct"], 0.05, 0.01)
    )

    total = c_valor + a_solvencia + m_margen + e_eficiencia + l_liquidez + s_skin

    d.update({
        "C_valor": c_valor,
        "A_solvencia": a_solvencia,
        "M_margen": m_margen,
        "E_eficiencia": e_eficiencia,
        "L_liquidez": l_liquidez,
        "S_skin": s_skin,
        "votos_netos": total,
        "imputed_fields": imputed
    })

    return d


def add_fundamentals(assets_df, universe_df):
    fund_rows = []

    candidates = universe_df[~universe_df["technical_only"]]["ticker"].tolist()

    print(f"🧮 Calculando fundamentales para {len(candidates)} activos elegibles...")

    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = {executor.submit(get_fundamental_raw, t): t for t in candidates}

        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                raw = fut.result()
                if raw:
                    fund_rows.append(score_fundamental(raw))
            except Exception:
                pass

    if not fund_rows:
        assets_df["has_fundamentals"] = False
        return assets_df

    fund_df = pd.DataFrame(fund_rows)

    if len(fund_df) >= 4:
        fund_df["cuartil"] = pd.qcut(
            fund_df["votos_netos"].rank(method="first"),
            4,
            labels=["Q4", "Q3", "Q2", "Q1"]
        ).astype(str)
    else:
        fund_df["cuartil"] = "N/A"

    fund_df["has_fundamentals"] = True

    merged = assets_df.merge(fund_df, on="ticker", how="left")
    merged["has_fundamentals"] = merged["has_fundamentals"].fillna(False)

    return merged


# ============================================================
# MACRO + FX
# ============================================================

def yf_close_series(ticker, period="1y"):
    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        df = flatten_yf(df)
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna()
    except Exception:
        return None


def fetch_fred(series_id, start, end):
    if web is None:
        return None
    try:
        df = web.DataReader(series_id, "fred", start, end)
        if df.empty:
            return None
        return df[series_id].ffill().bfill()
    except Exception:
        return None


def get_fng_data():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30", timeout=8)
        data = r.json().get("data", [])
        vals = [int(x["value"]) for x in data]
        return pd.Series(vals[::-1])
    except Exception:
        return pd.Series([50] * 30)


def macro_context():
    end = datetime.now()
    start = end - timedelta(days=400)

    data = {
        "VIX": yf_close_series("^VIX", "300d"),
        "MOVE": yf_close_series("^MOVE", "300d"),
        "BRENT": yf_close_series("BZ=F", "400d"),
        "JUNK_CDS": fetch_fred("BAMLH0A0HYM2", start, end),
        "TED_LIQ": fetch_fred("TEDRATE", start, end),
        "CRYPTO_F&G": get_fng_data(),
    }

    rows = []
    score = 0.0

    for name, series in data.items():
        if series is None or len(series.dropna()) < 21:
            rows.append({
                "id": name,
                "value": None,
                "roc5": None,
                "roc20": None,
                "impact": 0.0,
                "diag": "🟡 DATA_DELAY"
            })
            continue

        s = series.dropna()

        v_now = safe_float(s.iloc[-1])
        v_prev_5 = safe_float(s.iloc[-6])
        v_prev_20 = safe_float(s.iloc[-21])

        roc5 = v_now / v_prev_5 - 1 if v_prev_5 else 0
        roc20 = v_now / v_prev_20 - 1 if v_prev_20 else 0

        impact = 0.0
        diag = "Estable"

        if name == "VIX":
            if v_now > 28 and roc5 < -0.12:
                impact, diag = 4.0, "🔵 Giro comprador"
            elif roc5 > 0.15:
                impact, diag = -2.5, "🔴 Presión volatilidad"
            elif v_now < 21:
                impact, diag = 1.0, "🟢 Volatilidad contenida"

        elif name == "MOVE":
            if roc20 < -0.25:
                impact, diag = 2.5, "🟢 Calma bonos"
            elif roc5 > 0.15:
                impact, diag = -3.5, "💀 Tensión tipos"

        elif name == "JUNK_CDS":
            if roc5 > 0.08:
                impact, diag = -4.5, "⚠️ Riesgo crédito"
            elif v_now < 3.5:
                impact, diag = 1.5, "🟢 Crédito sano"

        elif name == "TED_LIQ":
            if v_now > 0.45:
                impact, diag = -4.0, "🚫 Iliquidez"
            elif v_now > 0.01:
                impact, diag = 1.0, "🟢 Liquidez OK"

        elif name == "BRENT":
            p80 = s.tail(252).quantile(0.80) if len(s) >= 100 else s.quantile(0.80)
            if v_now > p80 and roc5 > 0.03:
                impact, diag = -3.5, "⛽ Brent caro/acelerando"
            elif roc5 > 0.06:
                impact, diag = -3.0, "⛽ Shock coste"
            elif roc20 < -0.10:
                impact, diag = 1.0, "🟢 Energía aflojando"
            else:
                diag = "Percentil dinámico OK"

        elif name == "CRYPTO_F&G":
            if v_now < 25:
                impact, diag = 2.5, "🔵 Miedo extremo"
            elif v_now > 75:
                impact, diag = -2.5, "🔴 Euforia peligrosa"
            else:
                diag = "Neutral"

        score += impact

        rows.append({
            "id": name,
            "value": v_now,
            "roc5": roc5,
            "roc20": roc20,
            "impact": impact,
            "diag": diag
        })

    if score >= 4:
        label = "🟢 FAVORABLE"
    elif score >= 0:
        label = "🟡 NEUTRAL"
    else:
        label = "🔴 CAUTELOSO"

    return {
        "score": score,
        "label": label,
        "rows": rows
    }


def currency_context():
    s = yf_close_series("EURUSD=X", "1y")

    if s is None or len(s) < 60:
        return {
            "label": "🟡 DATA_DELAY",
            "score": 0.0,
            "eurusd": None,
            "roc5": None,
            "roc20": None,
            "sma50": None,
            "diag": "Sin datos suficientes de EUR/USD"
        }

    eurusd = safe_float(s.iloc[-1])
    roc5 = eurusd / safe_float(s.iloc[-6]) - 1
    roc20 = eurusd / safe_float(s.iloc[-21]) - 1
    sma50 = safe_float(s.rolling(50).mean().iloc[-1])

    score = 0.0

    if roc20 > 0.015:
        score -= 2.0

    if eurusd > sma50 and roc5 > 0:
        score -= 1.0

    if roc20 < -0.015:
        score += 1.5

    if score <= -2:
        label = "🔴 EUR FUERTE"
        diag = "Riesgo de erosión para inversor europeo con activos USD"
    elif score < 0:
        label = "🟡 EUR PRESIONA"
        diag = "El EUR/USD puede restar rentabilidad a activos USD"
    elif score > 0:
        label = "🟢 USD FAVORABLE"
        diag = "El cambio acompaña exposición USD"
    else:
        label = "🟡 NEUTRAL"
        diag = "Sin presión clara por divisa"

    return {
        "label": label,
        "score": score,
        "eurusd": eurusd,
        "roc5": roc5,
        "roc20": roc20,
        "sma50": sma50,
        "diag": diag
    }


# ============================================================
# HTML
# ============================================================

INDEX_HTML = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sovereign Daily Command Center</title>

<style>
:root {
  --bg: #020617;
  --panel: #0f172a;
  --panel2: #111827;
  --border: rgba(148,163,184,.22);
  --text: #e5e7eb;
  --muted: #94a3b8;
  --green: #86efac;
  --red: #fca5a5;
  --yellow: #fde68a;
  --blue: #93c5fd;
  --purple: #d8b4fe;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  background:
    radial-gradient(circle at top left, rgba(30,64,175,.42), transparent 28%),
    radial-gradient(circle at bottom right, rgba(7,89,133,.24), transparent 28%),
    var(--bg);
  color: var(--text);
}

.container {
  max-width: 1500px;
  margin: auto;
  padding: 28px;
}

h1 {
  font-size: 38px;
  margin: 0;
  letter-spacing: -.03em;
}

.subtitle {
  color: var(--muted);
  margin-top: 8px;
  margin-bottom: 24px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 18px;
}

.grid2 {
  display: grid;
  grid-template-columns: 1.1fr .9fr;
  gap: 14px;
  margin-bottom: 18px;
}

.card {
  background: rgba(15, 23, 42, .92);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 35px rgba(0,0,0,.25);
}

.card h2 {
  margin-top: 0;
}

.metric-title {
  color: var(--muted);
  font-size: 13px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .05em;
}

.metric-big {
  font-size: 28px;
  font-weight: 950;
  margin-top: 8px;
}

.small {
  font-size: 12px;
  color: var(--muted);
  margin-top: 5px;
}

.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 22px 0;
}

.tab {
  cursor: pointer;
  border: 1px solid var(--border);
  background: rgba(15,23,42,.72);
  color: var(--text);
  border-radius: 999px;
  padding: 10px 14px;
  font-weight: 800;
}

.tab.active {
  background: rgba(59,130,246,.25);
  color: var(--blue);
  border-color: rgba(59,130,246,.5);
}

.section {
  display: none;
}

.section.active {
  display: block;
}

table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 14px;
  border: 1px solid var(--border);
}

th {
  background: rgba(30,41,59,.96);
  color: #cbd5e1;
  text-align: left;
  padding: 11px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .04em;
  position: sticky;
  top: 0;
}

td {
  border-top: 1px solid rgba(148,163,184,.13);
  padding: 11px;
  vertical-align: top;
}

tr:hover {
  background: rgba(30,41,59,.5);
}

.badge {
  display: inline-block;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  border: 1px solid var(--border);
  white-space: nowrap;
}

.buy { color: var(--green); background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.35); }
.sell { color: var(--red); background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.35); }
.partial { color: var(--yellow); background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.35); }
.neutral { color: #cbd5e1; background: rgba(148,163,184,.12); border-color: rgba(148,163,184,.28); }
.mixed { color: var(--purple); background: rgba(168,85,247,.15); border-color: rgba(168,85,247,.35); }
.q { color: var(--blue); background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.35); }

.ticker {
  font-weight: 950;
  font-size: 16px;
}

.name {
  color: var(--muted);
  font-size: 12px;
  margin-top: 3px;
}

.controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

input, select, button {
  border: 1px solid var(--border);
  background: rgba(15,23,42,.95);
  color: var(--text);
  border-radius: 10px;
  padding: 10px;
}

button {
  cursor: pointer;
  font-weight: 800;
}

button.primary {
  background: rgba(37,99,235,.42);
  border-color: rgba(59,130,246,.55);
  color: white;
}

button.danger {
  background: rgba(220,38,38,.22);
  border-color: rgba(248,113,113,.4);
  color: var(--red);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 2fr auto;
  gap: 8px;
  margin-bottom: 14px;
}

.warning {
  color: var(--yellow);
}

.footer {
  color: var(--muted);
  margin-top: 28px;
  font-size: 12px;
}

@media (max-width: 900px) {
  .grid, .grid2, .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
</head>

<body>
<div class="container">
  <h1>🛰️ Sovereign Daily Command Center</h1>
  <div class="subtitle" id="subtitle">Cargando datos...</div>

  <div class="grid" id="summaryCards"></div>

  <div class="tabs">
    <button class="tab active" onclick="showTab('global')">🌍 Panel global</button>
    <button class="tab" onclick="showTab('signals')">🎯 Señales</button>
    <button class="tab" onclick="showTab('universe')">📡 Universo</button>
    <button class="tab" onclick="showTab('portfolio')">💼 Mi cartera</button>
    <button class="tab" onclick="showTab('rules')">📘 Reglas</button>
  </div>

  <section id="global" class="section active">
    <div class="grid2">
      <div class="card">
        <h2>🌍 Macro global</h2>
        <div id="macroTable"></div>
      </div>
      <div class="card">
        <h2>💶 EUR/USD para inversor europeo</h2>
        <div id="fxBox"></div>
      </div>
    </div>
  </section>

  <section id="signals" class="section">
    <div class="card">
      <h2>🎯 Señales recientes</h2>
      <div class="controls">
        <input id="signalSearch" placeholder="Buscar ticker o nombre..." oninput="renderSignals()">
        <select id="signalType" onchange="renderSignals()">
          <option value="">Todas las señales</option>
          <option value="COMPRA">Compras</option>
          <option value="VENTA">Ventas</option>
          <option value="MIXTA">Mixtas</option>
        </select>
        <select id="signalRegime" onchange="renderSignals()">
          <option value="">Todos los regímenes</option>
          <option value="ALCISTA">Alcista</option>
          <option value="LATERAL">Lateral</option>
          <option value="BAJISTA">Bajista</option>
        </select>
      </div>
      <div id="signalsTable"></div>
    </div>
  </section>

  <section id="universe" class="section">
    <div class="card">
      <h2>📡 Universo completo</h2>
      <div class="controls">
        <input id="universeSearch" placeholder="Buscar ticker, nombre, sector..." oninput="renderUniverse()">
        <select id="universeRegime" onchange="renderUniverse()">
          <option value="">Todos los regímenes</option>
          <option value="ALCISTA">Alcista</option>
          <option value="LATERAL">Lateral</option>
          <option value="BAJISTA">Bajista</option>
        </select>
      </div>
      <div id="universeTable"></div>
    </div>
  </section>

  <section id="portfolio" class="section">
    <div class="card">
      <h2>💼 Mi cartera privada</h2>
      <p class="small">
        Tus posiciones se guardan solo en este navegador mediante localStorage. No se suben a GitHub.
      </p>

      <div class="form-grid">
        <input id="pfTicker" placeholder="Ticker, ej. NVDA">
        <input id="pfQty" type="number" step="0.0001" placeholder="Cantidad">
        <input id="pfPrice" type="number" step="0.0001" placeholder="Precio compra">
        <input id="pfDate" type="date">
        <input id="pfNote" placeholder="Nota">
        <button class="primary" onclick="addPosition()">Añadir</button>
      </div>

      <div class="controls">
        <button onclick="exportPortfolio()">Exportar cartera JSON</button>
        <input type="file" id="importFile" accept=".json" onchange="importPortfolio(event)">
        <button class="danger" onclick="clearPortfolio()">Vaciar cartera</button>
      </div>

      <div id="portfolioSummary"></div>
      <div id="portfolioTable"></div>
    </div>
  </section>

  <section id="rules" class="section">
    <div class="card">
      <h2>📘 Reglas del sistema</h2>
      <p><b>Compra:</b> PVI cruza su media de abajo hacia arriba en las últimas 5 velas.</p>
      <p><b>Venta 50% PVI:</b> PVI cruza su media de arriba hacia abajo en las últimas 5 velas.</p>
      <p><b>Venta 50% McGinley:</b> el precio cruza McGinley de salida de arriba hacia abajo.</p>
      <p><b>Venta 100%:</b> se activan las dos patas de salida.</p>
      <p><b>Lateral:</b> demasiados cruces recientes alrededor de McGinley.</p>
      <p><b>Macro y EUR/USD:</b> notifican contexto, no bloquean señales.</p>
      <p class="warning"><b>Nota:</b> herramienta mecánica basada en reglas. No es asesoramiento financiero.</p>
    </div>
  </section>

  <div class="footer">
    Sovereign Daily Command Center · Datos vía yfinance/FRED/Fear & Greed · Actualización automática vía GitHub Actions.
  </div>
</div>

<script>
let allAssets = [];
let signals = [];
let summary = {};
let portfolio = JSON.parse(localStorage.getItem("sovereign_portfolio") || "[]");

function fmtNum(x, d=2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toLocaleString("es-ES", {maximumFractionDigits:d, minimumFractionDigits:d});
}

function fmtPct(x, d=1) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return (Number(x) * 100).toLocaleString("es-ES", {maximumFractionDigits:d, minimumFractionDigits:d}) + "%";
}

function clsFor(text) {
  text = String(text || "");
  if (text.includes("COMPRA") || text.includes("ALCISTA") || text.includes("MANTENER")) return "buy";
  if (text.includes("VENTA 100") || text.includes("BAJISTA") || text.includes("VENDER TODO")) return "sell";
  if (text.includes("VENTA") || text.includes("LATERAL") || text.includes("REDUCIR") || text.includes("STOP")) return "partial";
  if (text.includes("MIXTA")) return "mixed";
  return "neutral";
}

function badge(text, extra="") {
  return `<span class="badge ${extra || clsFor(text)}">${text || "—"}</span>`;
}

function qLabel(q) {
  if (q === "Q1") return "Q1 ⭐ TOP";
  if (q === "Q2") return "Q2 ✅";
  if (q === "Q3") return "Q3 ⚖️";
  if (q === "Q4") return "Q4 ⚠️";
  return "—";
}

function showTab(id) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  event.target.classList.add("active");
}

async function loadData() {
  allAssets = await fetch("data/all_assets.json").then(r => r.json());
  signals = await fetch("data/signals.json").then(r => r.json());
  summary = await fetch("data/summary.json").then(r => r.json());

  renderSummary();
  renderGlobal();
  renderSignals();
  renderUniverse();
  renderPortfolio();
}

function renderSummary() {
  document.getElementById("subtitle").innerText =
    `Última actualización: ${summary.generated_at || "—"} · Activos analizados: ${summary.total_assets || 0}`;

  const cards = [
    ["🌍 Macro", summary.macro?.label || "—", `Score: ${fmtNum(summary.macro?.score, 1)}`],
    ["💶 EUR/USD", summary.fx?.label || "—", summary.fx?.diag || "—"],
    ["🎯 Señales", String(summary.total_signals || 0), `Compras: ${summary.buy_signals || 0} · Ventas: ${summary.sell_signals || 0}`],
    ["⚙️ Parámetros", `${summary.config?.LOOKBACK_SIGNAL || 5} velas`, `PVI ${summary.config?.PVI_MA || 120} · McG ${summary.config?.MCG_EXIT_N || 45}`],
  ];

  document.getElementById("summaryCards").innerHTML = cards.map(c => `
    <div class="card">
      <div class="metric-title">${c[0]}</div>
      <div class="metric-big">${c[1]}</div>
      <div class="small">${c[2]}</div>
    </div>
  `).join("");
}

function renderGlobal() {
  const rows = (summary.macro?.rows || []).map(r => `
    <tr>
      <td>${r.id}</td>
      <td>${fmtNum(r.value, 2)}</td>
      <td>${fmtPct(r.roc5, 1)}</td>
      <td>${fmtPct(r.roc20, 1)}</td>
      <td>${fmtNum(r.impact, 1)}</td>
      <td>${r.diag || "—"}</td>
    </tr>
  `).join("");

  document.getElementById("macroTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Indicador</th><th>Valor</th><th>ROC 5D</th><th>ROC 20D</th><th>Impacto</th><th>Diagnóstico</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  const fx = summary.fx || {};
  document.getElementById("fxBox").innerHTML = `
    <p>${badge(fx.label || "—")}</p>
    <p>${fx.diag || "—"}</p>
    <table>
      <tbody>
        <tr><td>EUR/USD</td><td>${fmtNum(fx.eurusd, 4)}</td></tr>
        <tr><td>ROC 5D</td><td>${fmtPct(fx.roc5, 2)}</td></tr>
        <tr><td>ROC 20D</td><td>${fmtPct(fx.roc20, 2)}</td></tr>
        <tr><td>SMA 50</td><td>${fmtNum(fx.sma50, 4)}</td></tr>
        <tr><td>Score</td><td>${fmtNum(fx.score, 1)}</td></tr>
      </tbody>
    </table>
  `;
}

function signalRow(a) {
  const fund = a.has_fundamentals
    ? `
      <div><b>${fmtNum(a.votos_netos,1)}</b> ${badge(qLabel(a.cuartil), "q")}</div>
      <div class="small">
        C ${fmtNum(a.C_valor,1)} · A ${fmtNum(a.A_solvencia,1)} · M ${fmtNum(a.M_margen,1)} ·
        E ${fmtNum(a.E_eficiencia,1)} · L ${fmtNum(a.L_liquidez,1)} · S ${fmtNum(a.S_skin,1)}
      </div>
      <div class="small">
        ROIC ${fmtPct(a.roic,1)} · Altman ${fmtNum(a.altman_z,2)} · Piotroski ${fmtNum(a.piotroski,1)}
      </div>
    `
    : `<div class="small">Sin fundamentales / técnico-only</div>`;

  return `
    <tr>
      <td>
        <div class="ticker">${a.ticker}</div>
        <div class="name">${a.name || "—"}</div>
      </td>
      <td>${badge(a.main_signal)}<div class="small">${a.events_text || ""}</div></td>
      <td>${badge(a.regime)}<div class="small">Cruces: ${a.recent_crosses ?? "—"}</div></td>
      <td>
        ${fmtNum(a.close,2)}
        <div class="small">Dist. McG: ${fmtPct(a.dist_to_mcg_exit,1)}</div>
        <div class="small">PVI: ${a.pvi_status || "—"}</div>
      </td>
      <td>${fund}</td>
      <td>${a.bucket || "—"}</td>
    </tr>
  `;
}

function renderSignals() {
  const q = (document.getElementById("signalSearch")?.value || "").toUpperCase();
  const type = document.getElementById("signalType")?.value || "";
  const regime = document.getElementById("signalRegime")?.value || "";

  let data = signals.slice();

  if (q) {
    data = data.filter(a =>
      String(a.ticker).toUpperCase().includes(q) ||
      String(a.name).toUpperCase().includes(q)
    );
  }

  if (type) {
    data = data.filter(a => String(a.main_signal || "").includes(type));
  }

  if (regime) {
    data = data.filter(a => String(a.regime || "").includes(regime));
  }

  if (!data.length) {
    document.getElementById("signalsTable").innerHTML = `<p class="small">No hay señales con esos filtros.</p>`;
    return;
  }

  document.getElementById("signalsTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Activo</th><th>Señal</th><th>Régimen</th><th>Técnico</th><th>Fundamental</th><th>Universo</th>
        </tr>
      </thead>
      <tbody>${data.map(signalRow).join("")}</tbody>
    </table>
  `;
}

function renderUniverse() {
  const q = (document.getElementById("universeSearch")?.value || "").toUpperCase();
  const regime = document.getElementById("universeRegime")?.value || "";

  let data = allAssets.slice();

  if (q) {
    data = data.filter(a =>
      String(a.ticker).toUpperCase().includes(q) ||
      String(a.name).toUpperCase().includes(q) ||
      String(a.sector).toUpperCase().includes(q)
    );
  }

  if (regime) {
    data = data.filter(a => String(a.regime || "").includes(regime));
  }

  data = data.sort((a,b) => String(a.ticker).localeCompare(String(b.ticker)));

  const rows = data.map(a => `
    <tr>
      <td><div class="ticker">${a.ticker}</div><div class="name">${a.name || "—"}</div></td>
      <td>${badge(a.main_signal || "—")}</td>
      <td>${badge(a.regime || "—")}</td>
      <td>${fmtNum(a.close,2)}</td>
      <td>${a.pvi_status || "—"}</td>
      <td>${fmtPct(a.dist_to_mcg_exit,1)}</td>
      <td>${a.has_fundamentals ? fmtNum(a.votos_netos,1) : "—"}</td>
      <td>${a.has_fundamentals ? qLabel(a.cuartil) : "—"}</td>
      <td>${a.bucket || "—"}</td>
    </tr>
  `).join("");

  document.getElementById("universeTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Activo</th><th>Señal</th><th>Régimen</th><th>Precio</th><th>PVI</th><th>Dist. McG</th><th>Score</th><th>Q</th><th>Universo</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function findAsset(ticker) {
  return allAssets.find(a => String(a.ticker).toUpperCase() === String(ticker).toUpperCase());
}

function savePortfolio() {
  localStorage.setItem("sovereign_portfolio", JSON.stringify(portfolio));
}

function addPosition() {
  const ticker = document.getElementById("pfTicker").value.trim().toUpperCase();
  const quantity = parseFloat(document.getElementById("pfQty").value);
  const buyPrice = parseFloat(document.getElementById("pfPrice").value);
  const buyDate = document.getElementById("pfDate").value;
  const note = document.getElementById("pfNote").value;

  if (!ticker || !quantity || !buyPrice) {
    alert("Ticker, cantidad y precio de compra son obligatorios.");
    return;
  }

  const asset = findAsset(ticker);

  portfolio.push({
    ticker,
    name: asset ? asset.name : ticker,
    quantity,
    buy_price: buyPrice,
    buy_date: buyDate,
    note
  });

  savePortfolio();

  document.getElementById("pfTicker").value = "";
  document.getElementById("pfQty").value = "";
  document.getElementById("pfPrice").value = "";
  document.getElementById("pfNote").value = "";

  renderPortfolio();
}

function deletePosition(i) {
  portfolio.splice(i, 1);
  savePortfolio();
  renderPortfolio();
}

function clearPortfolio() {
  if (!confirm("¿Seguro que quieres vaciar la cartera local?")) return;
  portfolio = [];
  savePortfolio();
  renderPortfolio();
}

function systemAction(asset, pos) {
  if (!asset) {
    return ["⚪ SIN DATOS", "No hay datos diarios para este activo."];
  }

  const signal = asset.main_signal || "";
  const regime = asset.regime || "";
  const pvi = asset.pvi_status || "";
  const belowMcg = asset.price_below_mcg_exit === true;
  const dist = asset.dist_to_mcg_exit;

  if (signal.includes("VENTA 100")) {
    return ["🔴 VENDER TODO", "Han saltado las dos patas de salida: PVI y McGinley."];
  }

  if (signal.includes("VENTA 50% PVI")) {
    return ["🟠 VENDER 50%", "El PVI cruzó su media hacia abajo."];
  }

  if (signal.includes("VENTA 50% McGINLEY")) {
    return ["🟠 VENDER 50%", "El precio cruzó McGinley de arriba hacia abajo."];
  }

  if (belowMcg) {
    return ["🟠 REDUCIR / VIGILAR", "El precio está por debajo del McGinley de salida."];
  }

  if (regime.includes("LATERAL")) {
    return ["🟡 NO AUMENTAR", "Régimen lateral. Señales menos fiables."];
  }

  if (regime.includes("BAJISTA")) {
    return ["🔴 NO AUMENTAR", "Régimen bajista. No conviene aumentar riesgo."];
  }

  if (regime.includes("ALCISTA") && pvi === "POSITIVO") {
    return ["🟢 MANTENER", "Tendencia alcista viva y PVI positivo."];
  }

  if (signal.includes("COMPRA")) {
    return ["🟢 POSIBLE COMPRA / AÑADIR", "Compra reciente detectada por cruce alcista de PVI."];
  }

  if (dist !== null && dist !== undefined && Number(dist) < 0.015) {
    return ["🟠 STOP CERCA", "El precio está muy cerca de McGinley."];
  }

  return ["⚪ MANTENER / VIGILAR", "Sin señal operativa clara."];
}

function renderPortfolio() {
  if (!portfolio.length) {
    document.getElementById("portfolioSummary").innerHTML = "";
    document.getElementById("portfolioTable").innerHTML = `<p class="small">Aún no tienes posiciones guardadas.</p>`;
    return;
  }

  let totalValue = 0;
  let totalCost = 0;

  const rows = portfolio.map((p, i) => {
    const asset = findAsset(p.ticker);
    const price = asset ? Number(asset.close) : null;
    const qty = Number(p.quantity);
    const buy = Number(p.buy_price);

    const value = price ? qty * price : null;
    const cost = qty * buy;
    const pnl = value !== null ? value - cost : null;
    const pnlPct = value !== null && cost ? value / cost - 1 : null;

    if (value !== null) totalValue += value;
    totalCost += cost;

    const [action, reason] = systemAction(asset, p);

    return `
      <tr>
        <td>
          <div class="ticker">${p.ticker}</div>
          <div class="name">${p.name || (asset ? asset.name : "")}</div>
        </td>
        <td>${fmtNum(qty,4)}</td>
        <td>${fmtNum(buy,2)}</td>
        <td>${price ? fmtNum(price,2) : "—"}</td>
        <td>${pnlPct !== null ? fmtPct(pnlPct,2) : "—"}<div class="small">${pnl !== null ? fmtNum(pnl,2) : "—"}</div></td>
        <td>${asset ? badge(asset.main_signal || "—") : "—"}<div class="small">${asset ? (asset.events_text || "") : ""}</div></td>
        <td>${asset ? badge(asset.regime || "—") : "—"}</td>
        <td>${badge(action)}<div class="small">${reason}</div></td>
        <td>${p.buy_date || "—"}<div class="small">${p.note || ""}</div></td>
        <td><button class="danger" onclick="deletePosition(${i})">Borrar</button></td>
      </tr>
    `;
  }).join("");

  const totalPnl = totalValue - totalCost;
  const totalPnlPct = totalCost ? totalValue / totalCost - 1 : null;

  document.getElementById("portfolioSummary").innerHTML = `
    <div class="grid">
      <div class="card"><div class="metric-title">Valor cartera</div><div class="metric-big">${fmtNum(totalValue,2)}</div></div>
      <div class="card"><div class="metric-title">Coste</div><div class="metric-big">${fmtNum(totalCost,2)}</div></div>
      <div class="card"><div class="metric-title">PnL</div><div class="metric-big">${fmtNum(totalPnl,2)}</div><div class="small">${fmtPct(totalPnlPct,2)}</div></div>
      <div class="card"><div class="metric-title">Posiciones</div><div class="metric-big">${portfolio.length}</div></div>
    </div>
  `;

  document.getElementById("portfolioTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Activo</th><th>Cantidad</th><th>Compra</th><th>Actual</th><th>PnL</th>
          <th>Señal</th><th>Régimen</th><th>Acción sistema</th><th>Fecha/Nota</th><th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function exportPortfolio() {
  const blob = new Blob([JSON.stringify(portfolio, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sovereign_portfolio.json";
  a.click();
}

function importPortfolio(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = function(e) {
    try {
      portfolio = JSON.parse(e.target.result);
      savePortfolio();
      renderPortfolio();
    } catch {
      alert("No se pudo importar el JSON.");
    }
  };

  reader.readAsText(file);
}

loadData();
</script>
</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Sovereign Daily Command Center")
    site_dir = Path("site")
    data_dir = site_dir / "data"

    site_dir.mkdir(exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    universe_df = build_universe_df()

    print(f"📡 Universo: {len(universe_df)} activos")

    tech_rows = []

    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = {
            executor.submit(analyze_technical, row): row["ticker"]
            for _, row in universe_df.iterrows()
        }

        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                res = fut.result()
                tech_rows.append(res)
                if res.get("has_signal"):
                    print(f"✅ Señal: {ticker} -> {res.get('main_signal')}")
            except Exception as e:
                print(f"⚠️ Error técnico {ticker}: {e}")

    assets_df = pd.DataFrame(tech_rows)

    assets_df = add_fundamentals(assets_df, universe_df)

    if not assets_df.empty:
        assets_df["sort_signal"] = assets_df["bars_min"].fillna(999)
        assets_df = assets_df.sort_values(["sort_signal", "ticker"], ascending=[True, True])

    signals_df = assets_df[assets_df["has_signal"] == True].copy() if not assets_df.empty else pd.DataFrame()

    macro = macro_context()
    fx = currency_context()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buy_signals = 0
    sell_signals = 0

    if not signals_df.empty:
        buy_signals = int(signals_df["main_signal"].astype(str).str.contains("COMPRA").sum())
        sell_signals = int(signals_df["main_signal"].astype(str).str.contains("VENTA").sum())

    summary = {
        "generated_at": generated_at,
        "total_assets": int(len(assets_df)),
        "total_signals": int(len(signals_df)),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "macro": macro,
        "fx": fx,
        "config": CONFIG,
    }

    (data_dir / "all_assets.json").write_text(
        json.dumps(records_for_json(assets_df), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (data_dir / "signals.json").write_text(
        json.dumps(records_for_json(signals_df), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (data_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=clean_json_value),
        encoding="utf-8"
    )

    (site_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    print("✅ Site generado en /site")
    print(f"🕒 Última actualización: {generated_at}")


if __name__ == "__main__":
    main()
