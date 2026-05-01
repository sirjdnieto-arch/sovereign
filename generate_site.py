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
    "PVI_SIGNAL_TYPE": "EMA",

    "MCG_REGIME_N": 20,
    "MCG_EXIT_N": 45,

    "LATERAL_LOOKBACK": 20,

    "PRICE_PERIOD": "10y",
    "DROP_TODAY_CANDLE": True,
    "REQUIRE_CURRENT_SIGNAL_STATE": True,

    "MAX_WORKERS": 8,

    "RVOL_PERIOD": 30,
    "RVOL_HIGH": 1.5,
    "VOLUME_MIN_NONZERO_PCT": 0.60,

    "VOL_FILTER_MODE": "score",

    "ATR_PERIOD": 14,
    "CHOP_PERIOD": 14,

    "PVI_GAP_MIN_STRONG": 0.001,

    "ENTRY_QUALITY_A": 85,
    "ENTRY_QUALITY_B": 70,
    "ENTRY_QUALITY_C": 55,

    "EXIT_PRESSURE_VIGILAR": 25,
    "EXIT_PRESSURE_REDUCIR": 50,
    "EXIT_PRESSURE_FUERTE": 75,
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
    "Commodities",
    "Crypto",
}

TECHNICAL_ONLY_SUFFIXES = ()


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
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
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
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        df = flatten_yf(df)
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                if col == "Volume":
                    df[col] = 0
                else:
                    df[col] = df["Close"]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
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
# INDICADORES
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
    df["PVI_Signal"] = (
        df["PVI"]
        .ewm(span=ma_period, adjust=False, min_periods=ma_period)
        .mean()
    )
    return df


def calculate_atr(df, period=14):
    df = df.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def calculate_chop(df, period=14):
    df = df.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(period, min_periods=period).sum()
    high_max = high.rolling(period, min_periods=period).max()
    low_min = low.rolling(period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-9)) / np.log10(period)
    return chop.replace([np.inf, -np.inf], np.nan)


def volume_quality_check(df, min_nonzero_pct=0.60):
    if df is None or df.empty or "Volume" not in df.columns:
        return {
            "volume_quality": False,
            "volume_nonzero_pct": 0.0,
            "volume_zero_days": None,
            "volume_warning": "Sin volumen"
        }
    vol = df["Volume"].fillna(0)
    nonzero_pct = float((vol > 0).mean())
    zero_days = int((vol <= 0).sum())
    if nonzero_pct >= min_nonzero_pct:
        warning = ""
        ok = True
    else:
        warning = "Volumen poco fiable; validar PVI en TradingView"
        ok = False
    return {
        "volume_quality": bool(ok),
        "volume_nonzero_pct": nonzero_pct,
        "volume_zero_days": zero_days,
        "volume_warning": warning
    }


def calculate_rvol(df, period=30):
    vol = df["Volume"].fillna(0).astype(float)
    ma = vol.rolling(period, min_periods=max(5, period // 2)).mean()
    rvol = vol / ma.replace(0, np.nan)
    return rvol.replace([np.inf, -np.inf], np.nan)


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


def signal_freshness(bars_ago):
    if bars_ago is None or pd.isna(bars_ago):
        return "—"
    bars_ago = int(bars_ago)
    if bars_ago == 0:
        return "Hoy"
    if bars_ago <= 2:
        return "Reciente"
    if bars_ago <= 5:
        return "Esta semana"
    return "Antigua"


def entry_quality_label_from_score(score, volume_quality):
    if score is None or pd.isna(score):
        return "—"
    if not volume_quality:
        return "VOL NO FIABLE"
    s = float(score)
    if s >= CONFIG["ENTRY_QUALITY_A"]:
        return "ALTA"
    if s >= CONFIG["ENTRY_QUALITY_B"]:
        return "BUENA"
    if s >= CONFIG["ENTRY_QUALITY_C"]:
        return "MEDIA"
    return "BAJA"


def exit_pressure_label(score):
    if score is None or pd.isna(score):
        return "—"
    score = float(score)
    if score >= CONFIG["EXIT_PRESSURE_FUERTE"]:
        return "Salida fuerte"
    if score >= CONFIG["EXIT_PRESSURE_REDUCIR"]:
        return "Reducir"
    if score >= CONFIG["EXIT_PRESSURE_VIGILAR"]:
        return "Vigilar"
    return "Baja"


def technical_state_label(regime, pvi_status, price_below_mcg_exit):
    if price_below_mcg_exit and pvi_status == "NEGATIVO":
        return "Deteriorado"
    if price_below_mcg_exit:
        return "Bajo McGinley"
    if "ALCISTA" in str(regime) and pvi_status == "POSITIVO":
        return "Alcista confirmado"
    if "LATERAL" in str(regime):
        return "Lateral"
    if "BAJISTA" in str(regime):
        return "Bajista"
    return "Neutral"


def stop_status_label(dist_ratio, dist_atr):
    if not valid_number(dist_ratio):
        return "—"
    dist_pct = float(dist_ratio) * 100.0
    if dist_pct < 2:
        pct_status = "MUY CERCA"
    elif dist_pct < 5:
        pct_status = "AJUSTADO"
    elif dist_pct < 10:
        pct_status = "HOLGADO"
    else:
        pct_status = "LEJANO"
    if valid_number(dist_atr):
        da = float(dist_atr)
        if da < 1:
            atr_status = "(<1 ATR)"
        elif da < 2:
            atr_status = "(1-2 ATR)"
        elif da < 3:
            atr_status = "(2-3 ATR)"
        else:
            atr_status = "(>3 ATR)"
    else:
        atr_status = ""
    return f"{pct_status} {atr_status}".strip()


def calculate_entry_quality(
    has_buy, close_now, mcg_regime_now, mcg_exit_now,
    recent_crosses, pvi_gap, rvol_now, bullish_high_volume,
    volume_quality, dist_to_mcg_exit, dist_to_mcg_exit_atr,
):
    if not has_buy:
        return None, "—", ""
    score = 40
    notes = ["PVI compra reciente"]
    if valid_number(close_now, mcg_regime_now) and close_now > mcg_regime_now:
        score += 20
        notes.append("precio sobre McGinley regimen")
    else:
        notes.append("contra tendencia / bajo regimen")
    if valid_number(close_now, mcg_exit_now) and close_now > mcg_exit_now:
        score += 10
        notes.append("precio sobre McGinley salida")
    else:
        notes.append("cerca/bajo McGinley salida")
    if recent_crosses <= 2:
        score += 15
        notes.append("no lateral")
    else:
        notes.append("lateralidad elevada")
    if bullish_high_volume:
        score += 15
        notes.append(f"RVOL alto alcista")
    elif valid_number(rvol_now) and rvol_now >= 1.0:
        score += 8
        notes.append("volumen normal/positivo")
    else:
        notes.append("sin confirmacion de volumen")
    if valid_number(pvi_gap) and pvi_gap >= CONFIG["PVI_GAP_MIN_STRONG"]:
        score += 10
        notes.append("gap PVI no micro")
    else:
        notes.append("microcruce PVI / gap debil")
    if not volume_quality:
        score -= 10
        notes.append("volumen poco fiable")
    if valid_number(dist_to_mcg_exit_atr):
        d_atr = float(dist_to_mcg_exit_atr)
        if d_atr > 3:
            score -= 15
            notes.append("entrada muy extendida (>3 ATR)")
        elif d_atr > 2:
            score -= 7
            notes.append("entrada algo extendida (2-3 ATR)")
    if valid_number(dist_to_mcg_exit):
        d_pct = float(dist_to_mcg_exit) * 100
        if d_pct > 15:
            score -= 5
            notes.append("distancia >15% a McGinley salida")
    score = float(np.clip(score, 0, 100))
    label = entry_quality_label_from_score(score, volume_quality)
    return score, label, " · ".join(notes)


def calculate_exit_pressure(
    pvi_status, close_now, mcg_exit_now, mcg_regime_now,
    regime, recent_crosses, bearish_high_volume, pvi_gap
):
    score = 0
    notes = []
    if valid_number(close_now, mcg_exit_now) and close_now < mcg_exit_now:
        score += 35
        notes.append("precio bajo McGinley salida")
    if pvi_status == "NEGATIVO":
        score += 30
        notes.append("PVI negativo")
    if valid_number(close_now, mcg_regime_now) and close_now < mcg_regime_now:
        score += 15
        notes.append("precio bajo McGinley regimen")
    if "BAJISTA" in str(regime):
        score += 10
        notes.append("regimen bajista")
    elif "LATERAL" in str(regime):
        score += 8
        notes.append("regimen lateral")
    if recent_crosses > 2:
        score += 5
        notes.append("cruces recientes/lateralidad")
    if bearish_high_volume:
        score += 15
        notes.append("RVOL bajista alto")
    if valid_number(pvi_gap) and pvi_gap <= -CONFIG["PVI_GAP_MIN_STRONG"]:
        score += 10
        notes.append("gap PVI negativo claro")
    score = float(np.clip(score, 0, 100))
    label = exit_pressure_label(score)
    if not notes:
        notes.append("sin presion tecnica relevante")
    return score, label, " · ".join(notes)


def analyze_technical(row):
    ticker = row["ticker"]
    bucket = row["bucket"]
    name, currency, sector = get_name_currency_sector(ticker)
    df = yf_download_prices(ticker, CONFIG["PRICE_PERIOD"])
    try:
        if CONFIG.get("DROP_TODAY_CANDLE", True):
            today_utc = datetime.now(timezone.utc).date()
            if len(df) > 1 and pd.to_datetime(df.index[-1]).date() == today_utc:
                df = df.iloc[:-1]
    except Exception:
        pass
    if df.empty or len(df) < CONFIG["PVI_MA"] + 10:
        return {
            "ticker": ticker, "name": name, "bucket": bucket,
            "currency": currency, "sector": sector,
            "technical_only": bool(row["technical_only"]),
            "has_signal": False, "error": "Sin datos suficientes"
        }
    vol_q = volume_quality_check(df, CONFIG["VOLUME_MIN_NONZERO_PCT"])
    df = calculate_pvi(df, CONFIG["PVI_MA"])
    df["McG_Regime"] = calculate_mcginley(df["Close"], CONFIG["MCG_REGIME_N"])
    df["McG_Exit"] = calculate_mcginley(df["Close"], CONFIG["MCG_EXIT_N"])
    df["ATR"] = calculate_atr(df, CONFIG["ATR_PERIOD"])
    df["CHOP"] = calculate_chop(df, CONFIG["CHOP_PERIOD"])
    df["RVOL"] = calculate_rvol(df, CONFIG["RVOL_PERIOD"])
    df["PVI_Cross_Up"] = (
        (df["PVI"] > df["PVI_Signal"]) &
        (df["PVI"].shift(1) <= df["PVI_Signal"].shift(1))
    )
    df["PVI_Cross_Down"] = (
        (df["PVI"] < df["PVI_Signal"]) &
        (df["PVI"].shift(1) >= df["PVI_Signal"].shift(1))
    )
    df["McG_Cross_Down"] = (
        (df["Close"] < df["McG_Exit"]) &
        (df["Close"].shift(1) >= df["McG_Exit"].shift(1))
    )
    buy_ago = bars_ago_for_signal(df["PVI_Cross_Up"], CONFIG["LOOKBACK_SIGNAL"])
    pvi_sell_ago = bars_ago_for_signal(df["PVI_Cross_Down"], CONFIG["LOOKBACK_SIGNAL"])
    mcg_sell_ago = bars_ago_for_signal(df["McG_Cross_Down"], CONFIG["LOOKBACK_SIGNAL"])
    above = df["Close"] > df["McG_Regime"]
    recent_crosses = int(
        above.tail(CONFIG["LATERAL_LOOKBACK"])
        .astype(int).diff().abs().fillna(0).sum()
    )
    c = df.iloc[-1]
    close_now = safe_float(c["Close"])
    mcg_regime_now = safe_float(c["McG_Regime"])
    mcg_exit_now = safe_float(c["McG_Exit"])
    pvi_now = safe_float(c["PVI"])
    pvi_sig_now = safe_float(c["PVI_Signal"])
    pvi_prev = safe_float(df["PVI"].iloc[-2])
    pvi_sig_prev = safe_float(df["PVI_Signal"].iloc[-2])
    atr_now = safe_float(c["ATR"])
    chop_now = safe_float(c["CHOP"])
    rvol_now = safe_float(c["RVOL"])
    close_prev = safe_float(df["Close"].iloc[-2])
    bullish_high_volume = (
        valid_number(rvol_now, close_now, close_prev) and
        rvol_now >= CONFIG["RVOL_HIGH"] and close_now > close_prev
    )
    bearish_high_volume = (
        valid_number(rvol_now, close_now, close_prev) and
        rvol_now >= CONFIG["RVOL_HIGH"] and close_now < close_prev
    )
    raw_has_buy = buy_ago is not None
    raw_has_pvi_sell = pvi_sell_ago is not None
    raw_has_mcg_sell = mcg_sell_ago is not None
    if CONFIG.get("REQUIRE_CURRENT_SIGNAL_STATE", True):
        has_buy = (
            raw_has_buy and
            valid_number(pvi_now, pvi_sig_now) and pvi_now > pvi_sig_now
        )
        has_pvi_sell = (
            raw_has_pvi_sell and
            valid_number(pvi_now, pvi_sig_now) and pvi_now < pvi_sig_now
        )
        has_mcg_sell = (
            raw_has_mcg_sell and
            valid_number(close_now, mcg_exit_now) and close_now < mcg_exit_now
        )
    else:
        has_buy = raw_has_buy
        has_pvi_sell = raw_has_pvi_sell
        has_mcg_sell = raw_has_mcg_sell
    vol_confirms = bool(valid_number(rvol_now) and rvol_now >= CONFIG["RVOL_HIGH"])
    raw_buy_blocked_by_vol = False
    mode = CONFIG.get("VOL_FILTER_MODE", "score")
    if mode == "hard":
        if vol_q["volume_quality"] and not vol_confirms and has_buy:
            has_buy = False
            raw_buy_blocked_by_vol = True
    elif mode == "warn":
        raw_buy_blocked_by_vol = bool(vol_q["volume_quality"] and not vol_confirms)
    pvi_cross_current = (
        valid_number(pvi_now, pvi_sig_now, pvi_prev, pvi_sig_prev) and
        pvi_now > pvi_sig_now and pvi_prev <= pvi_sig_prev
    )
    if valid_number(close_now, mcg_regime_now) and close_now > mcg_regime_now:
        base_regime = "ALCISTA"
    else:
        base_regime = "BAJISTA"
    if recent_crosses > 2 or (valid_number(chop_now) and chop_now >= 61.8):
        regime = "LATERAL"
    else:
        regime = base_regime
    events = []
    if has_buy:
        events.append(("COMPRA", buy_ago))
    if has_pvi_sell:
        label = "VENTA 50% PVI"
        if bearish_high_volume:
            label = "VENTA PVI FUERTE"
        events.append((label, pvi_sell_ago))
    if has_mcg_sell:
        label = "VENTA 50% McGINLEY"
        if bearish_high_volume:
            label = "ROTURA McGINLEY CON VOLUMEN"
        events.append((label, mcg_sell_ago))
    if has_buy and (has_pvi_sell or has_mcg_sell):
        main_signal = "MIXTA"
    elif has_buy:
        main_signal = "COMPRA"
    elif has_pvi_sell and has_mcg_sell:
        main_signal = "VENTA 100%"
    elif has_pvi_sell:
        main_signal = "VENTA 50% PVI"
    elif has_mcg_sell:
        main_signal = "VENTA 50% McGINLEY"
    else:
        main_signal = "—"
    bars_min = min([e[1] for e in events], default=None)
    events_text = " · ".join([f"{n} ({ago_txt(a)})" for n, a in events])
    freshness = signal_freshness(bars_min)
    dist_to_mcg_exit = np.nan
    dist_to_mcg_exit_atr = np.nan
    if valid_number(close_now, mcg_exit_now) and mcg_exit_now != 0:
        dist_to_mcg_exit = close_now / mcg_exit_now - 1
    if valid_number(close_now, mcg_exit_now, atr_now) and atr_now > 0:
        dist_to_mcg_exit_atr = (close_now - mcg_exit_now) / atr_now
    if valid_number(pvi_now, pvi_sig_now):
        pvi_status = "POSITIVO" if pvi_now > pvi_sig_now else "NEGATIVO"
        pvi_gap = pvi_now / pvi_sig_now - 1 if pvi_sig_now != 0 else np.nan
    else:
        pvi_status = "N/A"
        pvi_gap = np.nan
    price_below_mcg_exit = (
        bool(close_now < mcg_exit_now)
        if valid_number(close_now, mcg_exit_now) else False
    )
    entry_score, entry_label, entry_notes = calculate_entry_quality(
        has_buy=has_buy, close_now=close_now, mcg_regime_now=mcg_regime_now,
        mcg_exit_now=mcg_exit_now, recent_crosses=recent_crosses,
        pvi_gap=pvi_gap, rvol_now=rvol_now, bullish_high_volume=bullish_high_volume,
        volume_quality=vol_q["volume_quality"], dist_to_mcg_exit=dist_to_mcg_exit,
        dist_to_mcg_exit_atr=dist_to_mcg_exit_atr,
    )
    exit_score, exit_label, exit_notes = calculate_exit_pressure(
        pvi_status=pvi_status, close_now=close_now, mcg_exit_now=mcg_exit_now,
        mcg_regime_now=mcg_regime_now, regime=regime, recent_crosses=recent_crosses,
        bearish_high_volume=bearish_high_volume, pvi_gap=pvi_gap
    )
    technical_state = technical_state_label(regime, pvi_status, price_below_mcg_exit)
    stop_status = stop_status_label(dist_to_mcg_exit, dist_to_mcg_exit_atr)
    return {
        "ticker": ticker, "name": name, "bucket": bucket,
        "currency": currency, "sector": sector,
        "technical_only": bool(row["technical_only"]),
        "has_signal": bool(events), "main_signal": main_signal,
        "events_text": events_text, "signal_freshness": freshness,
        "buy_ago": buy_ago, "pvi_sell_ago": pvi_sell_ago,
        "mcg_sell_ago": mcg_sell_ago, "bars_min": bars_min,
        "regime": regime, "technical_state": technical_state,
        "recent_crosses": recent_crosses,
        "close": close_now, "mcg_regime": mcg_regime_now,
        "mcg_exit": mcg_exit_now, "dist_to_mcg_exit": dist_to_mcg_exit,
        "dist_to_mcg_exit_atr": dist_to_mcg_exit_atr, "stop_status": stop_status,
        "atr": atr_now, "chop": chop_now,
        "pvi": pvi_now, "pvi_signal": pvi_sig_now, "pvi_status": pvi_status,
        "pvi_gap": pvi_gap, "pvi_prev": pvi_prev, "pvi_signal_prev": pvi_sig_prev,
        "pvi_cross_current": bool(pvi_cross_current),
        "pvi_signal_type": CONFIG.get("PVI_SIGNAL_TYPE", "EMA"),
        "rvol": rvol_now, "rvol_high": bool(vol_confirms),
        "bullish_high_volume": bool(bullish_high_volume),
        "bearish_high_volume": bool(bearish_high_volume),
        "volume_quality": vol_q["volume_quality"],
        "volume_nonzero_pct": vol_q["volume_nonzero_pct"],
        "volume_zero_days": vol_q["volume_zero_days"],
        "volume_warning": vol_q["volume_warning"],
        "vol_confirms": bool(vol_confirms),
        "raw_buy_blocked_by_vol": bool(raw_buy_blocked_by_vol),
        "entry_quality_score": entry_score, "entry_quality_label": entry_label,
        "entry_quality_notes": entry_notes,
        "exit_pressure_score": exit_score, "exit_pressure_label": exit_label,
        "exit_pressure_notes": exit_notes,
        "price_below_mcg_exit": price_below_mcg_exit,
        "last_date": str(df.index[-1].date()), "error": ""
    }


# ============================================================
# FUNDAMENTALES
# ============================================================

def safe_div(n, d):
    try:
        n = safe_float(n)
        d = safe_float(d)
        if not np.isfinite(n) or not np.isfinite(d) or d == 0:
            return np.nan
        return n / d
    except Exception:
        return np.nan


def scale_score(x, bad, good):
    x = safe_float(x)
    if not np.isfinite(x):
        return None
    if good == bad:
        return None
    s = (x - bad) / (good - bad) * 100
    return float(np.clip(s, 0, 100))


def avg_available(values, weights=None):
    clean = []
    clean_w = []
    if weights is None:
        weights = [1] * len(values)
    for v, w in zip(values, weights):
        if v is None:
            continue
        v = safe_float(v)
        if np.isfinite(v):
            clean.append(v)
            clean_w.append(w)
    if not clean:
        return None
    return float(np.average(clean, weights=clean_w))


def quality_label(score):
    if score is None or pd.isna(score):
        return "N/A"
    score = float(score)
    if score >= 85:
        return "Excelente"
    if score >= 75:
        return "Muy buena"
    if score >= 65:
        return "Buena"
    if score >= 50:
        return "Media"
    return "Debil"


def valuation_label(score):
    if score is None or pd.isna(score):
        return "N/A"
    score = float(score)
    if score >= 80:
        return "Muy barata"
    if score >= 65:
        return "Barata"
    if score >= 45:
        return "Razonable"
    if score >= 25:
        return "Cara"
    return "Muy cara"


def fundamental_profile(q_score, v_score):
    if q_score is None or v_score is None:
        return "—"
    if pd.isna(q_score) or pd.isna(v_score):
        return "—"
    q = float(q_score)
    v = float(v_score)
    if q >= 70 and v >= 70:
        return "Calidad con descuento"
    if q >= 70 and v >= 45:
        return "Calidad razonable"
    if q >= 70 and v < 45:
        return "Calidad cara"
    if 50 <= q < 70 and v >= 65:
        return "Value especulativo"
    if q < 50 and v >= 65:
        return "Value trap"
    if q < 50 and v < 45:
        return "Debil y cara"
    return "Equilibrado"


def confidence_grade(score):
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def fundamental_trend(metrics):
    signals = []
    if valid_number(metrics.get("revenue_growth")):
        signals.append(1 if metrics["revenue_growth"] > 0 else -1)
    if valid_number(metrics.get("op_income_growth")):
        signals.append(1 if metrics["op_income_growth"] > 0 else -1)
    if valid_number(metrics.get("net_income_growth")):
        signals.append(1 if metrics["net_income_growth"] > 0 else -1)
    if not signals:
        return "—"
    avg = sum(signals) / len(signals)
    if avg > 0.3:
        return "Mejorando"
    if avg < -0.3:
        return "Deteriorando"
    return "Estable"


def valuation_style(sector, industry, model):
    sector_l = str(sector or "").lower()
    industry_l = str(industry or "").lower()
    if model in ("Bank", "Insurance", "Financial"):
        return "financial"
    growth_words = [
        "technology", "communication", "software", "semiconductor",
        "internet", "interactive media", "biotechnology"
    ]
    defensive_words = [
        "consumer defensive", "healthcare", "pharmaceutical",
        "medical", "staples"
    ]
    value_cyclical_words = [
        "energy", "utilities", "basic materials", "oil",
        "gas", "metals", "mining"
    ]
    if any(w in sector_l or w in industry_l for w in growth_words):
        return "growth"
    if any(w in sector_l or w in industry_l for w in value_cyclical_words):
        return "value_cyclical"
    if any(w in sector_l or w in industry_l for w in defensive_words):
        return "defensive"
    return "standard"


def valuation_score_corporate(metrics):
    style = valuation_style(
        metrics.get("sector"),
        metrics.get("industry"),
        metrics.get("fundamental_model")
    )
    if style == "growth":
        pe_bad, pe_good = 55, 18
        pb_bad, pb_good = 14, 2.5
        ev_bad, ev_good = 35, 14
        fcf_good = 0.045
    elif style == "value_cyclical":
        pe_bad, pe_good = 25, 8
        pb_bad, pb_good = 4, 0.8
        ev_bad, ev_good = 14, 4
        fcf_good = 0.08
    elif style == "defensive":
        pe_bad, pe_good = 35, 12
        pb_bad, pb_good = 7, 1.2
        ev_bad, ev_good = 22, 8
        fcf_good = 0.06
    else:
        pe_bad, pe_good = 40, 12
        pb_bad, pb_good = 8, 1.5
        ev_bad, ev_good = 25, 10
        fcf_good = 0.06
    return avg_available([
        scale_score(metrics.get("pe"), pe_bad, pe_good),
        scale_score(metrics.get("pb"), pb_bad, pb_good),
        scale_score(metrics.get("ev_ebitda"), ev_bad, ev_good),
        scale_score(metrics.get("fcf_yield"), 0.00, fcf_good),
        scale_score(metrics.get("upside"), -0.10, 0.20),
    ])


def get_statement(ticker_obj, names):
    for name in names:
        try:
            df = getattr(ticker_obj, name)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def get_series(df, aliases):
    if df is None or df.empty:
        return None
    for alias in aliases:
        if alias in df.index:
            try:
                obj = df.loc[alias]
                if isinstance(obj, pd.DataFrame):
                    obj = obj.iloc[0]
                return pd.to_numeric(obj, errors="coerce")
            except Exception:
                pass
    aliases_low = [a.lower() for a in aliases]
    for idx in df.index:
        idx_low = str(idx).lower()
        if any(a in idx_low for a in aliases_low):
            try:
                obj = df.loc[idx]
                if isinstance(obj, pd.DataFrame):
                    obj = obj.iloc[0]
                return pd.to_numeric(obj, errors="coerce")
            except Exception:
                pass
    return None


def get_val(df, aliases, pos=0, lookahead=4):
    s = get_series(df, aliases)
    if s is None or len(s) == 0:
        return np.nan
    try:
        if pos < len(s) and pd.notna(s.iloc[pos]):
            return safe_float(s.iloc[pos])
    except Exception:
        pass
    try:
        window = s.iloc[pos:pos + lookahead].dropna()
        if len(window) > 0:
            return safe_float(window.iloc[0])
    except Exception:
        pass
    return np.nan


def get_ttm(q_df, a_df, aliases, multiplier=4):
    s = get_series(q_df, aliases)
    if s is not None:
        vals = s.dropna()
        if len(vals) >= 4:
            return safe_float(vals.iloc[:4].sum())
        if len(vals) >= 1 and multiplier:
            return safe_float(vals.iloc[0] * multiplier)
    return get_val(a_df, aliases, 0)


def statement_latest_date(*dfs):
    dates = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for c in df.columns:
            try:
                dates.append(pd.to_datetime(c).to_pydatetime().replace(tzinfo=None))
            except Exception:
                pass
    if not dates:
        return None
    return max(dates)


def period_label_from_date(dt, annual=False):
    if dt is None:
        return "N/A"
    if annual:
        return f"FY{str(dt.year)[-2:]}"
    q = ((dt.month - 1) // 3) + 1
    return f"{q}Q{str(dt.year)[-2:]}"


def reporting_context(q_df, a_df):
    latest = statement_latest_date(q_df)
    if q_df is not None and not q_df.empty and len(q_df.columns) >= 2:
        try:
            d1 = pd.to_datetime(q_df.columns[0])
            d2 = pd.to_datetime(q_df.columns[1])
            diff = abs((d1 - d2).days)
            if diff > 300:
                return 1, "Annual-like", period_label_from_date(d1, annual=True), d1
            if diff > 150:
                return 2, "Semiannual", period_label_from_date(d1), d1
            return 4, "Quarterly", period_label_from_date(d1), d1
        except Exception:
            pass
    if latest is not None:
        return 4, "Quarterly/partial", period_label_from_date(latest), latest
    annual_latest = statement_latest_date(a_df)
    if annual_latest is not None:
        return 1, "Annual", period_label_from_date(annual_latest, annual=True), annual_latest
    return 4, "Unknown", "N/A", None


def route_fundamental_model(ticker, info):
    sector = str(info.get("sector") or "")
    industry = str(info.get("industry") or "")
    bank_tickers = {"JPM", "BAC", "SAN.MC", "BBVA.MC", "BNP.PA", "SAN.PA"}
    insurance_tickers = {"ALV.DE", "MUV2.DE", "MAP.MC"}
    if ticker.endswith("-USD") or ticker.endswith("=F"):
        return "Technical-only"
    if ticker in insurance_tickers or "insurance" in industry.lower():
        return "Insurance"
    if ticker in bank_tickers or "bank" in industry.lower() or "banks" in industry.lower():
        return "Bank"
    if "financial services" in sector.lower():
        return "Financial"
    return "Corporate"


def calculate_piotroski_v2(f_df, b_df, cf_df, prev_pos=1, multiplier=1):
    if f_df is None or f_df.empty or b_df is None or b_df.empty:
        return np.nan
    points = 0
    possible = 0

    def add(cond):
        nonlocal points, possible
        if cond is None:
            return
        possible += 1
        if bool(cond):
            points += 1

    ni0 = get_val(f_df, ["Net Income", "Net Income Common Stockholders"], 0)
    ni1 = get_val(f_df, ["Net Income", "Net Income Common Stockholders"], prev_pos)
    cfo0 = get_val(cf_df, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
    ta0 = get_val(b_df, ["Total Assets"], 0)
    ta1 = get_val(b_df, ["Total Assets"], prev_pos)
    ltd0 = get_val(b_df, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Total Debt"], 0)
    ltd1 = get_val(b_df, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation", "Total Debt"], prev_pos)
    ca0 = get_val(b_df, ["Current Assets", "Total Current Assets"], 0)
    ca1 = get_val(b_df, ["Current Assets", "Total Current Assets"], prev_pos)
    cl0 = get_val(b_df, ["Current Liabilities", "Total Current Liabilities"], 0)
    cl1 = get_val(b_df, ["Current Liabilities", "Total Current Liabilities"], prev_pos)
    gp0 = get_val(f_df, ["Gross Profit"], 0)
    gp1 = get_val(f_df, ["Gross Profit"], prev_pos)
    rev0 = get_val(f_df, ["Total Revenue", "Operating Revenue"], 0)
    rev1 = get_val(f_df, ["Total Revenue", "Operating Revenue"], prev_pos)
    shares0 = get_val(b_df, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 0)
    shares1 = get_val(b_df, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], prev_pos)
    if valid_number(ni0):
        add(ni0 > 0)
    if valid_number(cfo0):
        add(cfo0 > 0)
    if valid_number(ni0, ni1, ta0, ta1) and ta0 != 0 and ta1 != 0:
        add((ni0 * multiplier / ta0) > (ni1 * multiplier / ta1))
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
        add((rev0 * multiplier / ta0) > (rev1 * multiplier / ta1))
    if possible == 0:
        return np.nan
    return 9 * points / possible


def calculate_altman_z_v2(b_df, f_df, info, market_cap, multiplier):
    try:
        ta = get_val(b_df, ["Total Assets"], 0)
        if not valid_number(ta) or ta <= 0:
            return np.nan
        ca = get_val(b_df, ["Current Assets", "Total Current Assets"], 0)
        cl = get_val(b_df, ["Current Liabilities", "Total Current Liabilities"], 0)
        re = get_val(b_df, ["Retained Earnings"], 0)
        ebit = get_val(f_df, ["Operating Income", "EBIT"], 0)
        revenue = get_val(f_df, ["Total Revenue", "Operating Revenue"], 0)
        total_liab = get_val(
            b_df,
            ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"], 0
        )
        if not valid_number(ca): ca = 0
        if not valid_number(cl): cl = 0
        if not valid_number(re): re = 0
        ebit = ebit * multiplier if valid_number(ebit) else np.nan
        revenue = revenue * multiplier if valid_number(revenue) else np.nan
        if not valid_number(total_liab) or total_liab == 0:
            return np.nan
        wc = ca - cl
        return (
            1.2 * safe_div(wc, ta) + 1.4 * safe_div(re, ta) +
            3.3 * safe_div(ebit, ta) + 0.6 * safe_div(market_cap, total_liab) +
            1.0 * safe_div(revenue, ta)
        )
    except Exception:
        return np.nan


def calculate_beneish_m_score(f_df, b_df, cf_df, prev_pos=1):
    try:
        rev0 = get_val(f_df, ["Total Revenue", "Operating Revenue"], 0)
        rev1 = get_val(f_df, ["Total Revenue", "Operating Revenue"], prev_pos)
        rec0 = get_val(b_df, ["Accounts Receivable", "Receivables", "Net Receivables"], 0)
        rec1 = get_val(b_df, ["Accounts Receivable", "Receivables", "Net Receivables"], prev_pos)
        gp0 = get_val(f_df, ["Gross Profit"], 0)
        gp1 = get_val(f_df, ["Gross Profit"], prev_pos)
        ca0 = get_val(b_df, ["Current Assets", "Total Current Assets"], 0)
        ca1 = get_val(b_df, ["Current Assets", "Total Current Assets"], prev_pos)
        ppe0 = get_val(b_df, ["Net PPE", "Property Plant Equipment", "Property Plant And Equipment Net"], 0)
        ppe1 = get_val(b_df, ["Net PPE", "Property Plant Equipment", "Property Plant And Equipment Net"], prev_pos)
        ta0 = get_val(b_df, ["Total Assets"], 0)
        ta1 = get_val(b_df, ["Total Assets"], prev_pos)
        dep0 = abs(get_val(cf_df, ["Depreciation", "Depreciation And Amortization"], 0))
        dep1 = abs(get_val(cf_df, ["Depreciation", "Depreciation And Amortization"], prev_pos))
        sga0 = get_val(f_df, ["Selling General And Administration", "Selling General Administrative"], 0)
        sga1 = get_val(f_df, ["Selling General And Administration", "Selling General Administrative"], prev_pos)
        debt0 = get_val(b_df, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
        debt1 = get_val(b_df, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"], prev_pos)
        ni0 = get_val(f_df, ["Net Income", "Net Income Common Stockholders"], 0)
        cfo0 = get_val(cf_df, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
        comps = []
        dsri = safe_div(safe_div(rec0, rev0), safe_div(rec1, rev1))
        if valid_number(dsri): comps.append("dsri")
        gm0 = safe_div(gp0, rev0)
        gm1 = safe_div(gp1, rev1)
        gmi = safe_div(gm1, gm0)
        if valid_number(gmi): comps.append("gmi")
        aqi = safe_div(
            1 - safe_div(ca0 + ppe0, ta0),
            1 - safe_div(ca1 + ppe1, ta1)
        )
        if valid_number(aqi): comps.append("aqi")
        sgi = safe_div(rev0, rev1)
        if valid_number(sgi): comps.append("sgi")
        depi = safe_div(
            safe_div(dep1, dep1 + ppe1),
            safe_div(dep0, dep0 + ppe0)
        )
        if valid_number(depi): comps.append("depi")
        sgai = safe_div(safe_div(sga0, rev0), safe_div(sga1, rev1))
        if valid_number(sgai): comps.append("sgai")
        lvgi = safe_div(safe_div(debt0, ta0), safe_div(debt1, ta1))
        if valid_number(lvgi): comps.append("lvgi")
        tata = safe_div(ni0 - cfo0, ta0)
        if valid_number(tata): comps.append("tata")
        if len(comps) < 5:
            return np.nan
        dsri  = dsri  if valid_number(dsri)  else 1
        gmi   = gmi   if valid_number(gmi)   else 1
        aqi   = aqi   if valid_number(aqi)   else 1
        sgi   = sgi   if valid_number(sgi)   else 1
        depi  = depi  if valid_number(depi)  else 1
        sgai  = sgai  if valid_number(sgai)  else 1
        lvgi  = lvgi  if valid_number(lvgi)  else 1
        tata  = tata  if valid_number(tata)  else 0
        m = (
            -4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi +
            0.892 * sgi + 0.115 * depi - 0.172 * sgai +
            4.679 * tata - 0.327 * lvgi
        )
        return safe_float(m)
    except Exception:
        return np.nan


def growth_rate(df, aliases, prev_pos=4):
    v0 = get_val(df, aliases, 0)
    v1 = get_val(df, aliases, prev_pos)
    if not valid_number(v0, v1) or v1 == 0:
        return np.nan
    return v0 / abs(v1) - 1


def consistency_score(f_df):
    s = get_series(f_df, ["Net Income", "Net Income Common Stockholders"])
    if s is None:
        return None
    vals = s.dropna().iloc[:4]
    if len(vals) == 0:
        return None
    return float((vals > 0).sum() / len(vals) * 100)


def compute_confidence(model, latest_date, q_df, a_df, metrics):
    expected_by_model = {
        "Corporate": [
            "revenue_ttm", "op_income_ttm", "net_income_ttm", "cfo_ttm",
            "total_assets", "equity", "total_debt", "roic", "op_margin",
            "cash_quality", "piotroski"
        ],
        "Bank": [
            "revenue_ttm", "net_income_ttm", "total_assets", "equity",
            "roe", "roa", "equity_assets", "pb"
        ],
        "Insurance": [
            "revenue_ttm", "net_income_ttm", "total_assets", "equity",
            "roe", "equity_assets", "pb"
        ],
        "Financial": [
            "revenue_ttm", "net_income_ttm", "total_assets", "equity",
            "roe", "op_margin", "pb", "piotroski"
        ],
    }
    expected = expected_by_model.get(model, expected_by_model["Corporate"])
    available = 0
    for k in expected:
        if k in metrics and valid_number(metrics[k]):
            available += 1
    coverage = available / len(expected) if expected else 0
    freshness_points = 0
    if latest_date is not None:
        age = (datetime.now().replace(tzinfo=None) - pd.to_datetime(latest_date).replace(tzinfo=None)).days
        if age <= 150:
            freshness_points = 15
        elif age <= 270:
            freshness_points = 10
        elif age <= 420:
            freshness_points = 5
    history_points = 0
    try:
        q_cols = len(q_df.columns) if q_df is not None and not q_df.empty else 0
        a_cols = len(a_df.columns) if a_df is not None and not a_df.empty else 0
        if q_cols >= 4 or a_cols >= 2:
            history_points = 15
        elif q_cols >= 2 or a_cols >= 1:
            history_points = 8
    except Exception:
        pass
    confidence = coverage * 70 + freshness_points + history_points
    confidence = float(np.clip(confidence, 0, 100))
    return confidence, confidence_grade(confidence), available, len(expected)


def score_quality_axis(sub_quality, sub_cash, sub_solvency, sub_growth, sub_risk):
    return avg_available(
        [sub_quality, sub_cash, sub_solvency, sub_growth, sub_risk],
        weights=[30, 25, 20, 15, 10]
    )


def score_corporate(metrics):
    sub_quality = avg_available([
        scale_score(metrics.get("roic"), 0.05, 0.20),
        scale_score(metrics.get("op_margin"), 0.05, 0.25),
        scale_score(metrics.get("gross_margin"), 0.20, 0.60),
        scale_score(metrics.get("piotroski"), 3, 8),
    ])
    sub_cash = avg_available([
        scale_score(metrics.get("cash_quality"), 0.60, 1.50),
        scale_score(metrics.get("fcf_margin"), -0.03, 0.12),
        100 if valid_number(metrics.get("cfo_ttm")) and metrics.get("cfo_ttm") > 0 else 20,
        100 if valid_number(metrics.get("fcf_ttm")) and metrics.get("fcf_ttm") > 0 else 20,
    ])
    sub_solvency = avg_available([
        scale_score(metrics.get("altman_z"), 1.8, 3.0),
        scale_score(metrics.get("net_debt_ebitda"), 4.0, 1.0),
        scale_score(metrics.get("interest_coverage"), 1.5, 8.0),
        scale_score(metrics.get("current_ratio"), 0.8, 2.0),
    ])
    sub_growth = avg_available([
        scale_score(metrics.get("revenue_growth"), -0.10, 0.15),
        scale_score(metrics.get("op_income_growth"), -0.10, 0.20),
        metrics.get("consistency_score"),
    ])
    sub_valuation = valuation_score_corporate(metrics)
    beneish = metrics.get("beneish_m")
    if valid_number(beneish):
        if beneish > -1.78:
            beneish_score = 20
        elif beneish > -2.22:
            beneish_score = 60
        else:
            beneish_score = 100
    else:
        beneish_score = None
    sub_risk = avg_available([
        beneish_score,
        scale_score(metrics.get("shares_growth"), 0.05, 0.00),
        scale_score(metrics.get("net_debt_ebitda"), 4.0, 1.0),
        scale_score(metrics.get("altman_z"), 1.8, 3.0),
        100 if valid_number(metrics.get("fcf_ttm")) and metrics.get("fcf_ttm") > 0 else 40,
    ])
    return sub_quality, sub_cash, sub_solvency, sub_growth, sub_valuation, sub_risk


def score_financial(metrics, model):
    roe = metrics.get("roe")
    roa = metrics.get("roa")
    pb = metrics.get("pb")
    roe_pb = np.nan
    if valid_number(roe, pb) and pb > 0:
        roe_pb = roe / pb
    sub_quality = avg_available([
        scale_score(roe, 0.08, 0.18),
        scale_score(roa, 0.005, 0.015),
        scale_score(metrics.get("piotroski"), 3, 8),
        scale_score(metrics.get("op_margin"), 0.15, 0.45),
    ])
    sub_solvency = avg_available([
        scale_score(metrics.get("equity_assets"), 0.05, 0.12),
        scale_score(metrics.get("debt_equity"), 3.0, 0.5),
    ])
    sub_cash = avg_available([
        scale_score(metrics.get("cash_quality"), 0.50, 1.30),
        metrics.get("consistency_score"),
    ])
    sub_growth = avg_available([
        scale_score(metrics.get("revenue_growth"), -0.08, 0.12),
        scale_score(metrics.get("net_income_growth"), -0.10, 0.15),
        metrics.get("consistency_score"),
    ])
    sub_valuation = avg_available([
        scale_score(pb, 3.0, 0.8),
        scale_score(metrics.get("pe"), 25, 8),
        scale_score(roe_pb, 0.04, 0.12),
        scale_score(metrics.get("upside"), -0.10, 0.20),
    ])
    beneish = metrics.get("beneish_m")
    if valid_number(beneish):
        if beneish > -1.78:
            beneish_score = 30
        elif beneish > -2.22:
            beneish_score = 65
        else:
            beneish_score = 100
    else:
        beneish_score = None
    sub_risk = avg_available([
        beneish_score,
        scale_score(metrics.get("shares_growth"), 0.05, 0.00),
        scale_score(metrics.get("equity_assets"), 0.05, 0.12),
    ])
    return sub_quality, sub_cash, sub_solvency, sub_growth, sub_valuation, sub_risk


def get_fundamental_raw(ticker):
    try:
        asset = yf.Ticker(ticker)
        try:
            info = asset.info or {}
        except Exception:
            info = {}
        model = route_fundamental_model(ticker, info)
        if model == "Technical-only":
            return None
        bs_a = get_statement(asset, ["balance_sheet", "balancesheet"])
        is_a = get_statement(asset, ["income_stmt", "financials"])
        cf_a = get_statement(asset, ["cashflow"])
        bs_q = get_statement(asset, ["quarterly_balance_sheet", "quarterly_balancesheet"])
        is_q = get_statement(asset, ["quarterly_income_stmt", "quarterly_financials"])
        cf_q = get_statement(asset, ["quarterly_cashflow"])
        if (bs_a.empty and bs_q.empty) or (is_a.empty and is_q.empty):
            return None
        multiplier, reporting_frequency, period_label, latest_date = reporting_context(is_q, is_a)
        if not is_a.empty and len(is_a.columns) >= 2:
            p_f, p_b, p_cf = is_a, bs_a, cf_a
            prev_pos = 1
            p_mult = 1
        else:
            p_f, p_b, p_cf = is_q, bs_q, cf_q
            prev_pos = 4 if len(is_q.columns) >= 5 else 1
            p_mult = multiplier
        sector = info.get("sector") or "—"
        industry = info.get("industry") or "—"
        name = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency") or "—"
        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
        market_cap = safe_float(info.get("marketCap"))
        target = safe_float(info.get("targetMeanPrice"))
        upside = np.nan
        if valid_number(price, target) and price > 0 and target > 0:
            upside = target / price - 1
        revenue_ttm = get_ttm(is_q, is_a, ["Total Revenue", "Operating Revenue"], multiplier)
        gross_profit_ttm = get_ttm(is_q, is_a, ["Gross Profit"], multiplier)
        op_income_ttm = get_ttm(is_q, is_a, ["Operating Income", "EBIT"], multiplier)
        net_income_ttm = get_ttm(is_q, is_a, ["Net Income", "Net Income Common Stockholders"], multiplier)
        cfo_ttm = get_ttm(cf_q, cf_a, ["Operating Cash Flow", "Total Cash From Operating Activities"], multiplier)
        capex_ttm = get_ttm(cf_q, cf_a, ["Capital Expenditure", "Capital Expenditures"], multiplier)
        fcf_ttm = np.nan
        if valid_number(cfo_ttm, capex_ttm):
            fcf_ttm = cfo_ttm + capex_ttm
        tax_ttm = get_ttm(is_q, is_a, ["Tax Provision"], multiplier)
        pretax_ttm = get_ttm(is_q, is_a, ["Pretax Income", "Income Before Tax"], multiplier)
        interest_expense_ttm = get_ttm(is_q, is_a, ["Interest Expense", "Interest Expense Non Operating"], multiplier)
        ebitda_ttm = get_ttm(is_q, is_a, ["EBITDA", "Normalized EBITDA"], multiplier)
        if pd.isna(ebitda_ttm):
            ebitda_ttm = safe_float(info.get("ebitda"))
        bs_ref = bs_q if not bs_q.empty else bs_a
        total_assets = get_val(bs_ref, ["Total Assets"], 0)
        current_assets = get_val(bs_ref, ["Current Assets", "Total Current Assets"], 0)
        current_liabilities = get_val(bs_ref, ["Current Liabilities", "Total Current Liabilities"], 0)
        total_liabilities = get_val(bs_ref, ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"], 0)
        equity = get_val(bs_ref, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], 0)
        if pd.isna(equity) and valid_number(total_assets, total_liabilities):
            equity = total_assets - total_liabilities
        total_debt = get_val(bs_ref, ["Total Debt"], 0)
        if pd.isna(total_debt):
            lt_debt = get_val(bs_ref, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
            st_debt = get_val(bs_ref, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], 0)
            total_debt = np.nansum([lt_debt, st_debt])
        cash = get_val(bs_ref, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"], 0)
        if pd.isna(cash):
            cash = 0
        shares0 = get_val(bs_ref, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 0)
        shares1 = get_val(bs_ref, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], prev_pos)
        shares_growth = (
            safe_div(shares0, shares1) - 1
            if valid_number(shares0, shares1) and shares1 != 0 else np.nan
        )
        tax_rate = 0.21
        if valid_number(tax_ttm, pretax_ttm) and pretax_ttm > 0:
            tax_rate = np.clip(tax_ttm / pretax_ttm, 0, 0.35)
        nopat = op_income_ttm * (1 - tax_rate) if valid_number(op_income_ttm) else np.nan
        invested_capital = np.nan
        if valid_number(equity, total_debt):
            invested_capital = equity + total_debt - cash
        roic = safe_div(nopat, invested_capital) if valid_number(nopat, invested_capital) and invested_capital > 0 else np.nan
        roe = safe_div(net_income_ttm, equity)
        roa = safe_div(net_income_ttm, total_assets)
        equity_assets = safe_div(equity, total_assets)
        debt_equity = safe_div(total_debt, equity)
        gross_margin = safe_div(gross_profit_ttm, revenue_ttm)
        op_margin = safe_div(op_income_ttm, revenue_ttm)
        fcf_margin = safe_div(fcf_ttm, revenue_ttm)
        fcf_yield = safe_div(fcf_ttm, market_cap)
        cash_quality = safe_div(cfo_ttm, net_income_ttm)
        current_ratio = safe_div(current_assets, current_liabilities)
        interest_coverage = np.nan
        if valid_number(op_income_ttm, interest_expense_ttm):
            if abs(interest_expense_ttm) < 1e-9:
                interest_coverage = 10
            else:
                interest_coverage = op_income_ttm / abs(interest_expense_ttm)
        net_debt_ebitda = np.nan
        if valid_number(total_debt, cash, ebitda_ttm) and ebitda_ttm > 0:
            net_debt_ebitda = (total_debt - cash) / ebitda_ttm
        pb = safe_float(info.get("priceToBook"))
        pe = safe_float(info.get("trailingPE"))
        if pd.isna(pe):
            pe = safe_float(info.get("forwardPE"))
        ps = safe_float(info.get("priceToSalesTrailing12Months"))
        if pd.isna(ps) and valid_number(market_cap, revenue_ttm) and revenue_ttm > 0:
            ps = market_cap / revenue_ttm
        ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
        if pd.isna(ev_ebitda):
            enterprise_value = safe_float(info.get("enterpriseValue"))
            if valid_number(enterprise_value, ebitda_ttm) and ebitda_ttm > 0:
                ev_ebitda = enterprise_value / ebitda_ttm
        revenue_growth = growth_rate(is_q if not is_q.empty else is_a, ["Total Revenue", "Operating Revenue"], prev_pos)
        op_income_growth = growth_rate(is_q if not is_q.empty else is_a, ["Operating Income", "EBIT"], prev_pos)
        net_income_growth = growth_rate(is_q if not is_q.empty else is_a, ["Net Income", "Net Income Common Stockholders"], prev_pos)
        piotroski = calculate_piotroski_v2(p_f, p_b, p_cf, prev_pos=prev_pos, multiplier=p_mult)
        altman_z = np.nan
        if model == "Corporate":
            altman_z = calculate_altman_z_v2(bs_ref, is_q if not is_q.empty else is_a, info, market_cap, multiplier)
        beneish_m = calculate_beneish_m_score(p_f, p_b, p_cf, prev_pos=prev_pos)
        consistency = consistency_score(is_q if not is_q.empty else is_a)
        raw_metrics = {
            "ticker": ticker, "sector": sector, "industry": industry,
            "fundamental_model": model,
            "revenue_ttm": revenue_ttm, "op_income_ttm": op_income_ttm,
            "net_income_ttm": net_income_ttm, "cfo_ttm": cfo_ttm, "fcf_ttm": fcf_ttm,
            "total_assets": total_assets, "equity": equity, "total_debt": total_debt,
            "roic": roic, "roe": roe, "roa": roa, "equity_assets": equity_assets,
            "debt_equity": debt_equity, "gross_margin": gross_margin,
            "op_margin": op_margin, "fcf_margin": fcf_margin, "fcf_yield": fcf_yield,
            "cash_quality": cash_quality, "current_ratio": current_ratio,
            "interest_coverage": interest_coverage, "net_debt_ebitda": net_debt_ebitda,
            "pb": pb, "pe": pe, "ps": ps, "ev_ebitda": ev_ebitda, "upside": upside,
            "revenue_growth": revenue_growth, "op_income_growth": op_income_growth,
            "net_income_growth": net_income_growth, "shares_growth": shares_growth,
            "piotroski": piotroski, "altman_z": altman_z, "beneish_m": beneish_m,
            "consistency_score": consistency,
        }
        conf_score, conf_grade, available_fields, expected_fields = compute_confidence(
            model, latest_date, is_q, is_a, raw_metrics
        )
        if conf_grade == "D":
            return None
        if model == "Corporate":
            sub_quality, sub_cash, sub_solvency, sub_growth, sub_valuation, sub_risk = score_corporate(raw_metrics)
            model_note = "Modelo corporativo"
            not_applicable = ""
        else:
            sub_quality, sub_cash, sub_solvency, sub_growth, sub_valuation, sub_risk = score_financial(raw_metrics, model)
            model_note = f"Modelo {model}"
            not_applicable = "Altman-Z industrial, Current Ratio industrial, Debt/EBITDA industrial."
        q_score = score_quality_axis(sub_quality, sub_cash, sub_solvency, sub_growth, sub_risk)
        v_score = sub_valuation
        if q_score is None and v_score is None:
            return None
        if q_score is not None:
            q_score = float(np.clip(q_score, 0, 100))
        if v_score is not None:
            v_score = float(np.clip(v_score, 0, 100))
        q_lbl = quality_label(q_score)
        v_lbl = valuation_label(v_score)
        profile = fundamental_profile(q_score, v_score)
        ftrend = fundamental_trend(raw_metrics)
        red_flags = []
        if valid_number(beneish_m) and beneish_m > -1.78:
            red_flags.append("Beneish sospechoso")
        if model == "Corporate" and valid_number(altman_z) and altman_z < 1.8:
            red_flags.append("Altman peligro")
        if valid_number(net_debt_ebitda) and net_debt_ebitda > 4:
            red_flags.append("Deuda alta")
        if valid_number(fcf_ttm) and fcf_ttm < 0:
            red_flags.append("FCF negativo")
        if valid_number(shares_growth) and shares_growth > 0.05:
            red_flags.append("Dilucion alta")
        if conf_grade == "C":
            red_flags.append("Cobertura limitada")
        return {
            "ticker": ticker, "name": name, "sector": sector,
            "industry": industry, "currency": currency,
            "fundamental_model": model, "period_label": period_label,
            "latest_report_date": str(pd.to_datetime(latest_date).date()) if latest_date is not None else None,
            "reporting_frequency": reporting_frequency, "reporting_multiplier": multiplier,
            "price": price, "market_cap": market_cap, "upside": upside,
            "revenue_ttm": revenue_ttm, "gross_profit_ttm": gross_profit_ttm,
            "op_income_ttm": op_income_ttm, "net_income_ttm": net_income_ttm,
            "cfo_ttm": cfo_ttm, "capex_ttm": capex_ttm, "fcf_ttm": fcf_ttm,
            "total_assets": total_assets, "total_liabilities": total_liabilities,
            "equity": equity, "total_debt": total_debt, "cash": cash,
            "roic": roic, "roe": roe, "roa": roa,
            "equity_assets": equity_assets, "debt_equity": debt_equity,
            "gross_margin": gross_margin, "op_margin": op_margin,
            "fcf_margin": fcf_margin, "fcf_yield": fcf_yield,
            "cash_quality": cash_quality, "current_ratio": current_ratio,
            "interest_coverage": interest_coverage, "net_debt_ebitda": net_debt_ebitda,
            "pb": pb, "pe": pe, "ps": ps, "ev_ebitda": ev_ebitda,
            "revenue_growth": revenue_growth, "op_income_growth": op_income_growth,
            "net_income_growth": net_income_growth, "shares_growth": shares_growth,
            "piotroski": piotroski, "altman_z": altman_z,
            "beneish_m": beneish_m, "consistency_score": consistency,
            "has_fundamentals": True,
            "quality_score": q_score, "quality_label": q_lbl,
            "valuation_score": v_score, "valuation_label": v_lbl,
            "fundamental_profile": profile, "fundamental_trend": ftrend,
            "confidence_score": conf_score, "confidence_grade": conf_grade,
            "available_fields": available_fields, "expected_fields": expected_fields,
            "score_quality": sub_quality, "score_cash": sub_cash,
            "score_solvency": sub_solvency, "score_growth": sub_growth,
            "score_valuation": sub_valuation, "score_risk": sub_risk,
            "valuation_style": valuation_style(sector, industry, model),
            "red_flags": ", ".join(red_flags) if red_flags else "",
            "model_note": model_note, "not_applicable": not_applicable,
        }
    except Exception:
        return None


def compute_opportunity(row):
    tech = safe_float(row.get("entry_quality_score"))
    qual = safe_float(row.get("quality_score"))
    val = safe_float(row.get("valuation_score"))
    dist_atr = safe_float(row.get("dist_to_mcg_exit_atr"))
    rvol = safe_float(row.get("rvol"))
    if not np.isfinite(tech):
        return None, "—"
    score = 0.0
    w = 0.0
    if np.isfinite(tech):
        score += tech * 0.4
        w += 0.4
    if np.isfinite(qual):
        score += qual * 0.3
        w += 0.3
    if np.isfinite(val):
        score += val * 0.2
        w += 0.2
    if np.isfinite(dist_atr):
        if dist_atr <= 1:
            s_stop = 100
        elif dist_atr <= 2:
            s_stop = 80
        elif dist_atr <= 3:
            s_stop = 60
        else:
            s_stop = 40
        score += s_stop * 0.1
        w += 0.1
    if w > 0:
        score /= w
    if np.isfinite(rvol) and rvol < CONFIG["RVOL_HIGH"]:
        score -= 5
    score = float(np.clip(score, 0, 100))
    cond_100 = (
        np.isfinite(tech) and tech >= 90 and
        np.isfinite(qual) and qual >= 80 and
        np.isfinite(val) and val >= 70 and
        np.isfinite(dist_atr) and dist_atr <= 1.0 and
        np.isfinite(rvol) and rvol >= CONFIG["RVOL_HIGH"]
    )
    if cond_100:
        score = 100.0
        label = "100/100 Senal + volumen + stop corto + calidad + barata"
    else:
        if score >= 85:
            label = "Oportunidad A"
        elif score >= 70:
            label = "Oportunidad B"
        elif score >= 55:
            label = "Oportunidad C (trade, no core)"
        else:
            label = "Oportunidad D (especulativa)"
    return score, label


def add_fundamentals(assets_df, universe_df):
    fund_rows = []
    candidates = universe_df[~universe_df["technical_only"]]["ticker"].tolist()
    print(f"Fundamental: {len(candidates)} activos elegibles...")
    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = {executor.submit(get_fundamental_raw, t): t for t in candidates}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                raw = fut.result()
                if raw:
                    fund_rows.append(raw)
                    print(
                        f"   OK {ticker} · "
                        f"Cal {raw.get('quality_score', 0):.0f} ({raw.get('quality_label','—')}) · "
                        f"Val {raw.get('valuation_score', 0):.0f} ({raw.get('valuation_label','—')}) · "
                        f"Conf {raw.get('confidence_grade','—')} · "
                        f"{raw.get('fundamental_trend','—')}"
                    )
            except Exception:
                pass
    if not fund_rows:
        assets_df["has_fundamentals"] = False
        return assets_df
    fund_df = pd.DataFrame(fund_rows)
    fund_df["quality_percentile"] = np.nan
    valid = fund_df["quality_score"].notna()
    if valid.sum() > 0:
        global_pct = fund_df.loc[valid, "quality_score"].rank(pct=True)
        fund_df.loc[valid, "quality_percentile"] = global_pct
        try:
            group_cols = ["fundamental_model", "sector"]
            group_sizes = fund_df.groupby(group_cols)["ticker"].transform("count")
            pct_sector = fund_df.groupby(group_cols)["quality_score"].rank(pct=True)
            mask = valid & (group_sizes >= 4)
            fund_df.loc[mask, "quality_percentile"] = pct_sector.loc[mask]
        except Exception:
            pass
        try:
            group_sizes_model = fund_df.groupby("fundamental_model")["ticker"].transform("count")
            pct_model = fund_df.groupby("fundamental_model")["quality_score"].rank(pct=True)
            mask = valid & (fund_df["quality_percentile"].isna()) & (group_sizes_model >= 4)
            fund_df.loc[mask, "quality_percentile"] = pct_model.loc[mask]
        except Exception:
            pass

    def q_from_pct(p):
        if pd.isna(p):
            return "N/A"
        if p >= 0.75:
            return "Q1"
        if p >= 0.50:
            return "Q2"
        if p >= 0.25:
            return "Q3"
        return "Q4"

    fund_df["quality_q"] = fund_df["quality_percentile"].apply(q_from_pct)
    merged = assets_df.merge(fund_df, on="ticker", how="left", suffixes=("", "_fund"))
    merged["has_fundamentals"] = merged["has_fundamentals"].fillna(False)
    merged["opportunity_score"], merged["opportunity_label"] = zip(
        *merged.apply(compute_opportunity, axis=1)
    )
    return merged


# ============================================================
# MACRO + FX
# ============================================================

def yf_close_series(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
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
            rows.append({"id": name, "value": None, "roc5": None, "roc20": None, "impact": 0.0, "diag": "DATA_DELAY"})
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
                impact, diag = 4.0, "Giro comprador"
            elif roc5 > 0.15:
                impact, diag = -2.5, "Presion volatilidad"
            elif v_now < 21:
                impact, diag = 1.0, "Volatilidad contenida"
        elif name == "MOVE":
            if roc20 < -0.25:
                impact, diag = 2.5, "Calma bonos"
            elif roc5 > 0.15:
                impact, diag = -3.5, "Tension tipos"
        elif name == "JUNK_CDS":
            if roc5 > 0.08:
                impact, diag = -4.5, "Riesgo credito"
            elif v_now < 3.5:
                impact, diag = 1.5, "Credito sano"
        elif name == "TED_LIQ":
            if v_now > 0.45:
                impact, diag = -4.0, "Iliquidez"
            elif v_now > 0.01:
                impact, diag = 1.0, "Liquidez OK"
        elif name == "BRENT":
            p80 = s.tail(252).quantile(0.80) if len(s) >= 100 else s.quantile(0.80)
            if v_now > p80 and roc5 > 0.03:
                impact, diag = -3.5, "Brent caro/acelerando"
            elif roc5 > 0.06:
                impact, diag = -3.0, "Shock coste"
            elif roc20 < -0.10:
                impact, diag = 1.0, "Energia aflojando"
            else:
                diag = "Percentil dinamico OK"
        elif name == "CRYPTO_F&G":
            if v_now < 25:
                impact, diag = 2.5, "Miedo extremo"
            elif v_now > 75:
                impact, diag = -2.5, "Euforia peligrosa"
            else:
                diag = "Neutral"
        score += impact
        rows.append({"id": name, "value": v_now, "roc5": roc5, "roc20": roc20, "impact": impact, "diag": diag})
    if score >= 4:
        label = "FAVORABLE"
    elif score >= 0:
        label = "NEUTRAL"
    else:
        label = "CAUTELOSO"
    return {
        "score": score, "label": label,
        "warning": bool(score < 0),
        "warning_text": "Macro cautelosa" if score < 0 else "",
        "rows": rows
    }


def currency_context():
    s = yf_close_series("EURUSD=X", "1y")
    if s is None or len(s) < 60:
        return {
            "label": "DATA_DELAY", "score": 0.0, "eurusd": None,
            "roc5": None, "roc20": None, "sma50": None,
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
        label = "EUR FUERTE"
        diag = "Riesgo de erosion para inversor europeo con activos USD"
    elif score < 0:
        label = "EUR PRESIONA"
        diag = "El EUR/USD puede restar rentabilidad a activos USD"
    elif score > 0:
        label = "USD FAVORABLE"
        diag = "El cambio acompana exposicion USD"
    else:
        label = "NEUTRAL"
        diag = "Sin presion clara por divisa"
    return {
        "label": label, "score": score, "eurusd": eurusd,
        "roc5": roc5, "roc20": roc20, "sma50": sma50, "diag": diag
    }


# ============================================================
# DEEP DIVE
# ============================================================

def make_ticker_detail_json(ticker, tech_result, fund_result=None):
    """
    Genera el JSON para el modulo Deep Dive.
    Siempre devuelve un dict (nunca None).
    Si no hay datos suficientes, devuelve esqueleto con chart_error.
    """
    skeleton = {
        "ticker":         ticker,
        "name":           tech_result.get("name", ticker),
        "bucket":         tech_result.get("bucket", "—"),
        "currency":       tech_result.get("currency", "—"),
        "sector":         tech_result.get("sector", "—"),
        "last_date":      tech_result.get("last_date", "—"),
        "has_chart_data": False,
        "chart_error":    "",
        "candles":        [],
        "mcg_regime":     [],
        "mcg_exit":       [],
        "pvi":            [],
        "pvi_signal":     [],
        "atr":            [],
        "rvol":           [],
        "chop":           [],
        "dist_mcg_exit":  [],
        "markers":        [],
        "technical":  {k: clean_json_value(v) for k, v in tech_result.items()},
        "fundamental": ({k: clean_json_value(v) for k, v in fund_result.items()}
                        if fund_result else None),
    }

    try:
        df = yf_download_prices(ticker, CONFIG["PRICE_PERIOD"])
        try:
            if CONFIG.get("DROP_TODAY_CANDLE", True):
                today_utc = datetime.now(timezone.utc).date()
                if len(df) > 1 and pd.to_datetime(df.index[-1]).date() == today_utc:
                    df = df.iloc[:-1]
        except Exception:
            pass
        min_bars = CONFIG["PVI_MA"] + 10
        if df.empty:
            skeleton["chart_error"] = "yfinance no devolvio datos para este ticker."
            return skeleton
        if len(df) < min_bars:
            skeleton["chart_error"] = (
                f"Historico insuficiente: {len(df)} velas "
                f"(minimo {min_bars})."
            )
            return skeleton
    except Exception as e:
        skeleton["chart_error"] = f"Error descargando precios: {e}"
        return skeleton

    try:
        df = calculate_pvi(df, CONFIG["PVI_MA"])
        df["McG_Regime"] = calculate_mcginley(df["Close"], CONFIG["MCG_REGIME_N"])
        df["McG_Exit"]   = calculate_mcginley(df["Close"], CONFIG["MCG_EXIT_N"])
        df["ATR"]        = calculate_atr(df, CONFIG["ATR_PERIOD"])
        df["CHOP"]       = calculate_chop(df, CONFIG["CHOP_PERIOD"])
        df["RVOL"]       = calculate_rvol(df, CONFIG["RVOL_PERIOD"])
        df["PVI_Cross_Up"] = (
            (df["PVI"] > df["PVI_Signal"]) &
            (df["PVI"].shift(1) <= df["PVI_Signal"].shift(1))
        )
        df["PVI_Cross_Down"] = (
            (df["PVI"] < df["PVI_Signal"]) &
            (df["PVI"].shift(1) >= df["PVI_Signal"].shift(1))
        )
        df["McG_Cross_Down"] = (
            (df["Close"] < df["McG_Exit"]) &
            (df["Close"].shift(1) >= df["McG_Exit"].shift(1))
        )
    except Exception as e:
        skeleton["chart_error"] = f"Error calculando indicadores: {e}"
        return skeleton

    def to_time(idx):
        return str(pd.to_datetime(idx).date())

    def series_to_list(s, decimals=4):
        out = []
        for t, v in s.items():
            fv = safe_float(v)
            if np.isfinite(fv):
                out.append({"time": to_time(t), "value": round(fv, decimals)})
        return out

    def bool_to_markers(s, series_type, color, shape, position):
        out = []
        for t, v in s.items():
            if bool(v):
                out.append({
                    "time": to_time(t), "type": series_type,
                    "color": color, "shape": shape, "position": position,
                })
        return out

    try:
        candles = []
        for t, row in df.iterrows():
            o = safe_float(row["Open"])
            h = safe_float(row["High"])
            l = safe_float(row["Low"])
            c = safe_float(row["Close"])
            vol = safe_float(row["Volume"])
            if not all(np.isfinite(x) for x in [o, h, l, c]):
                continue
            candles.append({
                "time":   to_time(t),
                "open":   round(o, 4),
                "high":   round(h, 4),
                "low":    round(l, 4),
                "close":  round(c, 4),
                "volume": int(vol) if np.isfinite(vol) else 0,
            })
    except Exception as e:
        skeleton["chart_error"] = f"Error serializando velas: {e}"
        return skeleton

    markers = (
        bool_to_markers(df["PVI_Cross_Up"],   "buy",      "#22c55e", "arrowUp",   "belowBar") +
        bool_to_markers(df["PVI_Cross_Down"], "sell_pvi", "#ef4444", "arrowDown", "aboveBar") +
        bool_to_markers(df["McG_Cross_Down"], "sell_mcg", "#f97316", "arrowDown", "aboveBar")
    )
    markers.sort(key=lambda x: x["time"])

    dist_series = (
        (df["Close"] - df["McG_Exit"]) /
        df["McG_Exit"].replace(0, np.nan) * 100
    ).round(2)

    return {
        "ticker":         ticker,
        "name":           tech_result.get("name", ticker),
        "bucket":         tech_result.get("bucket", "—"),
        "currency":       tech_result.get("currency", "—"),
        "sector":         tech_result.get("sector", "—"),
        "last_date":      tech_result.get("last_date", "—"),
        "has_chart_data": True,
        "chart_error":    "",
        "candles":        candles,
        "mcg_regime":     series_to_list(df["McG_Regime"], 2),
        "mcg_exit":       series_to_list(df["McG_Exit"],   2),
        "pvi":            series_to_list(df["PVI"],        2),
        "pvi_signal":     series_to_list(df["PVI_Signal"], 2),
        "atr":            series_to_list(df["ATR"],        4),
        "rvol":           series_to_list(df["RVOL"],       3),
        "chop":           series_to_list(df["CHOP"],       1),
        "dist_mcg_exit":  series_to_list(dist_series,      2),
        "markers":        markers,
        "technical":  {k: clean_json_value(v) for k, v in tech_result.items()},
        "fundamental": ({k: clean_json_value(v) for k, v in fund_result.items()}
                        if fund_result else None),
    }


# ============================================================
# HTML
# ============================================================

INDEX_HTML = (
    "<!doctype html>\n"
    "<html lang=\"es\">\n"
    "<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>LCrack Sovereign</title>\n"
    "<script src=\"https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js\"></script>\n"
    "<style>\n"
    ":root{"
    "--bg:#020617;--panel:#0f172a;--border:rgba(148,163,184,.22);"
    "--text:#e5e7eb;--muted:#94a3b8;--green:#86efac;--red:#fca5a5;"
    "--yellow:#fde68a;--blue:#93c5fd;--purple:#d8b4fe;--orange:#fdba74;"
    "}\n"
    "*{box-sizing:border-box}\n"
    "body{margin:0;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;"
    "background:radial-gradient(circle at top left,rgba(30,64,175,.42),transparent 28%),"
    "radial-gradient(circle at bottom right,rgba(7,89,133,.24),transparent 28%),var(--bg);color:var(--text)}\n"
    ".container{max-width:1650px;margin:auto;padding:28px}\n"
    "h1{font-size:38px;margin:0;letter-spacing:-.03em}\n"
    ".subtitle{color:var(--muted);margin-top:8px;margin-bottom:24px}\n"
    ".grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}\n"
    ".grid2{display:grid;grid-template-columns:1.1fr .9fr;gap:14px;margin-bottom:18px}\n"
    ".grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}\n"
    ".card{background:rgba(15,23,42,.92);border:1px solid var(--border);border-radius:18px;"
    "padding:18px;box-shadow:0 10px 35px rgba(0,0,0,.25)}\n"
    ".metric-title{color:var(--muted);font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:.05em}\n"
    ".metric-big{font-size:28px;font-weight:950;margin-top:8px}\n"
    ".metric-val{font-size:22px;font-weight:800}\n"
    ".small{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.35}\n"
    ".tabs{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}\n"
    ".tab{cursor:pointer;border:1px solid var(--border);background:rgba(15,23,42,.72);"
    "color:var(--text);border-radius:999px;padding:10px 14px;font-weight:800}\n"
    ".tab.active{background:rgba(59,130,246,.25);color:var(--blue);border-color:rgba(59,130,246,.5)}\n"
    ".section{display:none}\n"
    ".section.active{display:block}\n"
    "table{width:100%;border-collapse:collapse;overflow:hidden;border-radius:14px;border:1px solid var(--border)}\n"
    "th{background:rgba(30,41,59,.96);color:#cbd5e1;text-align:left;padding:11px;font-size:12px;"
    "text-transform:uppercase;letter-spacing:.04em;position:sticky;top:0}\n"
    "td{border-top:1px solid rgba(148,163,184,.13);padding:11px;vertical-align:top}\n"
    "tr:hover{background:rgba(30,41,59,.5)}\n"
    ".badge{display:inline-block;padding:5px 9px;border-radius:999px;font-size:12px;"
    "font-weight:900;border:1px solid var(--border);white-space:nowrap;margin:1px}\n"
    ".buy{color:var(--green);background:rgba(34,197,94,.15);border-color:rgba(34,197,94,.35)}\n"
    ".sell{color:var(--red);background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.35)}\n"
    ".partial{color:var(--yellow);background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.35)}\n"
    ".neutral{color:#cbd5e1;background:rgba(148,163,184,.12);border-color:rgba(148,163,184,.28)}\n"
    ".mixed{color:var(--purple);background:rgba(168,85,247,.15);border-color:rgba(168,85,247,.35)}\n"
    ".qbadge{color:var(--blue);background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.35)}\n"
    ".vbadge{color:var(--orange);background:rgba(251,146,60,.15);border-color:rgba(251,146,60,.35)}\n"
    ".warnbadge{color:var(--yellow);background:rgba(245,158,11,.14);border-color:rgba(245,158,11,.35)}\n"
    ".ticker{font-weight:950;font-size:16px}\n"
    ".name{color:var(--muted);font-size:12px;margin-top:3px}\n"
    ".controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:center}\n"
    "input,select,button{border:1px solid var(--border);background:rgba(15,23,42,.95);"
    "color:var(--text);border-radius:10px;padding:10px}\n"
    "button{cursor:pointer;font-weight:800}\n"
    "button.primary{background:rgba(37,99,235,.42);border-color:rgba(59,130,246,.55);color:white}\n"
    "button.danger{background:rgba(220,38,38,.22);border-color:rgba(248,113,113,.4);color:var(--red)}\n"
    ".form-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr 2fr auto;gap:8px;margin-bottom:14px}\n"
    ".warning{color:var(--yellow)}\n"
    ".score-bar-wrap{width:100%;height:6px;background:rgba(148,163,184,.2);border-radius:999px;overflow:hidden;margin-top:5px}\n"
    ".score-bar{height:6px;border-radius:999px}\n"
    ".footer{color:var(--muted);margin-top:28px;font-size:12px}\n"
    "#dd-header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:20px}\n"
    "#dd-ticker-input{font-size:18px;font-weight:900;padding:12px 18px;border-radius:12px;width:180px;text-transform:uppercase}\n"
    "#dd-ticker-list{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}\n"
    "#dd-ticker-list button{padding:6px 12px;font-size:12px;font-weight:700;border-radius:999px}\n"
    "#dd-loading{color:var(--muted);font-size:14px;padding:20px 0}\n"
    "#dd-body{display:none}\n"
    "#dd-chart-price{width:100%;height:400px;border-radius:12px;overflow:hidden}\n"
    "#dd-chart-pvi{width:100%;height:180px;border-radius:12px;overflow:hidden;margin-top:8px}\n"
    "#dd-chart-rvol{width:100%;height:140px;border-radius:12px;overflow:hidden;margin-top:8px}\n"
    "#dd-chart-dist{width:100%;height:140px;border-radius:12px;overflow:hidden;margin-top:8px}\n"
    ".dd-legend{display:flex;flex-wrap:wrap;gap:10px;padding:8px 0 12px 0;font-size:12px}\n"
    ".dd-legend span{display:flex;align-items:center;gap:5px}\n"
    ".dd-legend .dot{width:10px;height:10px;border-radius:50%}\n"
    ".kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}\n"
    ".kv-item{background:rgba(30,41,59,.6);border-radius:10px;padding:10px 12px}\n"
    ".kv-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}\n"
    ".kv-value{font-size:16px;font-weight:800;margin-top:4px}\n"
    ".dd-section-title{font-size:13px;font-weight:900;text-transform:uppercase;letter-spacing:.06em;"
    "color:var(--muted);margin:20px 0 10px 0;border-bottom:1px solid var(--border);padding-bottom:6px}\n"
    ".signal-timeline{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}\n"
    ".chart-placeholder{display:flex;align-items:center;justify-content:center;min-height:80px;"
    "color:var(--muted);font-size:13px;background:rgba(30,41,59,.4);border-radius:10px;"
    "padding:18px;border:1px dashed var(--border)}\n"
    "@media(max-width:900px){"
    ".grid,.grid2,.grid4,.form-grid{grid-template-columns:1fr}"
    "#dd-chart-price{height:260px}"
    "}\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "<div class=\"container\">\n"
    "<h1>LCrack Sovereign</h1>\n"
    "<div class=\"subtitle\" id=\"subtitle\">Cargando datos...</div>\n"
    "<div class=\"grid\" id=\"summaryCards\"></div>\n"
    "<div class=\"tabs\">\n"
    "<button class=\"tab active\" onclick=\"showTab('global',event)\">Panel global</button>\n"
    "<button class=\"tab\" onclick=\"showTab('signals',event)\">Senales</button>\n"
    "<button class=\"tab\" onclick=\"showTab('universe',event)\">Universo</button>\n"
    "<button class=\"tab\" onclick=\"showTab('deepdive',event)\">Deep Dive</button>\n"
    "<button class=\"tab\" onclick=\"showTab('portfolio',event)\">Mi cartera</button>\n"
    "<button class=\"tab\" onclick=\"showTab('rules',event)\">Reglas</button>\n"
    "</div>\n"
    "<section id=\"global\" class=\"section active\">\n"
    "<div class=\"grid2\">\n"
    "<div class=\"card\"><h2>Macro global</h2><div id=\"macroTable\"></div></div>\n"
    "<div class=\"card\"><h2>EUR/USD</h2><div id=\"fxBox\"></div></div>\n"
    "</div></section>\n"
    "<section id=\"signals\" class=\"section\">\n"
    "<div class=\"card\"><h2>Senales recientes</h2>\n"
    "<div class=\"controls\">\n"
    "<input id=\"signalSearch\" placeholder=\"Buscar ticker o nombre...\" oninput=\"renderSignals()\">\n"
    "<select id=\"signalType\" onchange=\"renderSignals()\">"
    "<option value=\"\">Todas las senales</option>"
    "<option value=\"COMPRA\">Compras</option>"
    "<option value=\"VENTA\">Ventas</option>"
    "<option value=\"MIXTA\">Mixtas</option>"
    "</select>\n"
    "<select id=\"signalRegime\" onchange=\"renderSignals()\">"
    "<option value=\"\">Todos los regimenes</option>"
    "<option value=\"ALCISTA\">Alcista</option>"
    "<option value=\"LATERAL\">Lateral</option>"
    "<option value=\"BAJISTA\">Bajista</option>"
    "</select>\n"
    "</div><div id=\"signalsTable\"></div></div></section>\n"
    "<section id=\"universe\" class=\"section\">\n"
    "<div class=\"card\"><h2>Universo completo</h2>\n"
    "<div class=\"controls\">\n"
    "<input id=\"universeSearch\" placeholder=\"Buscar ticker, nombre, sector...\" oninput=\"renderUniverse()\">\n"
    "<select id=\"universeRegime\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Todos los regimenes</option>"
    "<option value=\"ALCISTA\">Alcista</option>"
    "<option value=\"LATERAL\">Lateral</option>"
    "<option value=\"BAJISTA\">Bajista</option>"
    "</select>\n"
    "<select id=\"fundConfidenceFilter\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Confianza: todas</option>"
    "<option value=\"A\">Confianza A</option>"
    "<option value=\"B\">Confianza B+</option>"
    "<option value=\"C\">Confianza C+</option>"
    "</select>\n"
    "<select id=\"fundQualityFilter\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Calidad: todas</option>"
    "<option value=\"85\">Excelente 85+</option>"
    "<option value=\"75\">Muy buena 75+</option>"
    "<option value=\"65\">Buena 65+</option>"
    "<option value=\"50\">Media 50+</option>"
    "</select>\n"
    "<select id=\"fundValuationFilter\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Precio/Fund.: todos</option>"
    "<option value=\"80\">Muy barata 80+</option>"
    "<option value=\"65\">Barata 65+</option>"
    "<option value=\"45\">Razonable 45+</option>"
    "</select>\n"
    "<select id=\"entryQualityFilter\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Entrada: todas</option>"
    "<option value=\"85\">Entrada A</option>"
    "<option value=\"70\">Entrada B+</option>"
    "<option value=\"55\">Entrada C+</option>"
    "</select>\n"
    "<select id=\"exitPressureFilter\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Presion salida: todas</option>"
    "<option value=\"25\">Vigilar+</option>"
    "<option value=\"50\">Reducir+</option>"
    "<option value=\"75\">Salida fuerte</option>"
    "</select>\n"
    "<select id=\"fundQFilter\" onchange=\"renderUniverse()\">"
    "<option value=\"\">Q calidad: todos</option>"
    "<option value=\"Q1\">Q1 TOP</option>"
    "<option value=\"Q2\">Q2</option>"
    "<option value=\"Q3\">Q3</option>"
    "<option value=\"Q4\">Q4</option>"
    "</select>\n"
    "<label class=\"small\"><input id=\"onlyFundamentals\" type=\"checkbox\" onchange=\"renderUniverse()\"> Solo con fundamentales</label>\n"
    "</div><div id=\"universeTable\"></div></div></section>\n"
    "<section id=\"deepdive\" class=\"section\">\n"
    "<div class=\"card\"><h2>Deep Dive - Analisis profundo</h2>\n"
    "<div id=\"dd-header\">\n"
    "<input id=\"dd-ticker-input\" placeholder=\"AAPL\" onkeydown=\"if(event.key==='Enter')loadDeepDive()\">\n"
    "<button class=\"primary\" onclick=\"loadDeepDive()\">Analizar</button>\n"
    "<span class=\"small\">Selecciona un ticker o escribelo directamente</span>\n"
    "</div>\n"
    "<div id=\"dd-ticker-list\"></div>\n"
    "<div id=\"dd-loading\"></div>\n"
    "<div id=\"dd-body\">\n"
    "<div id=\"dd-asset-header\" style=\"margin-bottom:18px\"></div>\n"
    "<div class=\"dd-section-title\">Estado tecnico y senal</div>\n"
    "<div class=\"grid4\" id=\"dd-signal-cards\"></div>\n"
    "<div class=\"signal-timeline\" id=\"dd-signal-timeline\"></div>\n"
    "<div class=\"dd-section-title\">Precio + McGinley Regimen (N=20) + McGinley Salida (N=45)</div>\n"
    "<div class=\"dd-legend\">"
    "<span><span class=\"dot\" style=\"background:#e2e8f0\"></span>Precio</span>"
    "<span><span class=\"dot\" style=\"background:#3b82f6\"></span>McG Reg</span>"
    "<span><span class=\"dot\" style=\"background:#f97316\"></span>McG Salida</span>"
    "<span><span class=\"dot\" style=\"background:#22c55e\"></span>Compra PVI</span>"
    "<span><span class=\"dot\" style=\"background:#ef4444\"></span>Venta PVI</span>"
    "<span><span class=\"dot\" style=\"background:#f97316\"></span>Rotura McG</span>"
    "</div>\n"
    "<div id=\"dd-chart-price\"></div>\n"
    "<div class=\"dd-section-title\">PVI vs Senal PVI (EMA 120)</div>\n"
    "<div class=\"dd-legend\">"
    "<span><span class=\"dot\" style=\"background:#60a5fa\"></span>PVI</span>"
    "<span><span class=\"dot\" style=\"background:#f59e0b\"></span>Senal EMA 120</span>"
    "</div>\n"
    "<div id=\"dd-chart-pvi\"></div>\n"
    "<div class=\"dd-section-title\">Volumen relativo (RVOL) - Umbral 1.5x</div>\n"
    "<div id=\"dd-chart-rvol\"></div>\n"
    "<div class=\"dd-section-title\">Distancia al McGinley de salida (%)</div>\n"
    "<div id=\"dd-chart-dist\"></div>\n"
    "<div class=\"dd-section-title\">Indicadores snapshot</div>\n"
    "<div class=\"kv-grid\" id=\"dd-tech-kv\"></div>\n"
    "<div id=\"dd-fund-section\"></div>\n"
    "</div></div></section>\n"
    "<section id=\"portfolio\" class=\"section\">\n"
    "<div class=\"card\"><h2>Mi cartera privada</h2>\n"
    "<p class=\"small\">Las posiciones se guardan en localStorage de este navegador.</p>\n"
    "<div class=\"form-grid\">\n"
    "<input id=\"pfTicker\" placeholder=\"Ticker\">\n"
    "<input id=\"pfQty\" type=\"number\" step=\"0.0001\" placeholder=\"Cantidad\">\n"
    "<input id=\"pfPrice\" type=\"number\" step=\"0.0001\" placeholder=\"Precio compra\">\n"
    "<input id=\"pfDate\" type=\"date\">\n"
    "<input id=\"pfNote\" placeholder=\"Nota\">\n"
    "<button class=\"primary\" onclick=\"addPosition()\">Anadir</button>\n"
    "</div>\n"
    "<div class=\"controls\">\n"
    "<button onclick=\"exportPortfolio()\">Exportar JSON</button>\n"
    "<input type=\"file\" id=\"importFile\" accept=\".json\" onchange=\"importPortfolio(event)\">\n"
    "<button class=\"danger\" onclick=\"clearPortfolio()\">Vaciar cartera</button>\n"
    "</div>\n"
    "<div id=\"portfolioSummary\"></div>\n"
    "<div id=\"portfolioTable\"></div>\n"
    "</div></section>\n"
    "<section id=\"rules\" class=\"section\">\n"
    "<div class=\"card\"><h2>Reglas del sistema</h2>\n"
    "<h3>1. Senales tecnicas</h3>\n"
    "<p><b>Compra:</b> PVI cruza EMA(120) de abajo hacia arriba y se mantiene por encima.</p>\n"
    "<p><b>Venta 50% PVI:</b> PVI cruza EMA(120) de arriba hacia abajo.</p>\n"
    "<p><b>Venta 50% McGinley:</b> precio cruza McGinley salida (N=45) de arriba hacia abajo.</p>\n"
    "<p><b>Venta 100%:</b> PVI negativo + precio bajo McGinley salida.</p>\n"
    "<h3>2. Calidad de entrada</h3>\n"
    "<p>Score &gt;= 80 ideal. Score 70-80 aceptable. Score &lt; 70 evitar.</p>\n"
    "<h3>3. Presion de salida</h3>\n"
    "<p>&lt;25 Baja | 25-50 Vigilar | 50-75 Reducir | &gt;=75 Salida fuerte.</p>\n"
    "<h3>4. Stop y tamano</h3>\n"
    "<p>&lt;1 ATR normal | 1-2 prudente | 2-3 pequeno | &gt;3 esperar pullback.</p>\n"
    "<h3>5. Oportunidad global</h3>\n"
    "<p>100/100 senal+volumen+stop cercano+calidad+barata. A(85+) B(70+) C(55+) D(&lt;55).</p>\n"
    "<h3>6. Fundamentales</h3>\n"
    "<p>Calidad &gt;=75 + Val &gt;=45: core largo plazo. Calidad &lt;50 + cara: solo trade tactico.</p>\n"
    "<p class=\"warning\"><b>Aviso:</b> herramienta mecanica. No es asesoramiento financiero.</p>\n"
    "</div></section>\n"
    "<div class=\"footer\">LCrack Sovereign - Datos via yfinance / FRED / Fear &amp; Greed.</div>\n"
    "</div>\n"
    "<script>\n"
    "var allAssets=[];var signals=[];var summary={};var portfolio=JSON.parse(localStorage.getItem('sovereign_portfolio')||'[]');var _ddCharts=[];\n"
    "function fmtNum(x,d){d=(d===undefined)?2:d;if(x===null||x===undefined||isNaN(Number(x)))return'—';return Number(x).toLocaleString('es-ES',{maximumFractionDigits:d,minimumFractionDigits:d});}\n"
    "function fmtPct(x,d){d=(d===undefined)?1:d;if(x===null||x===undefined||isNaN(Number(x)))return'—';return(Number(x)*100).toLocaleString('es-ES',{maximumFractionDigits:d,minimumFractionDigits:d})+'%';}\n"
    "function clsFor(t){t=String(t||'');if(t.indexOf('COMPRA')>=0||t.indexOf('ALCISTA')>=0||t.indexOf('MANTENER')>=0||t.indexOf('Baja')>=0)return'buy';if(t.indexOf('VENTA 100')>=0||t.indexOf('BAJISTA')>=0||t.indexOf('Salida fuerte')>=0)return'sell';if(t.indexOf('VENTA')>=0||t.indexOf('LATERAL')>=0||t.indexOf('REDUCIR')>=0||t.indexOf('Reducir')>=0||t.indexOf('Vigilar')>=0)return'partial';if(t.indexOf('MIXTA')>=0)return'mixed';return'neutral';}\n"
    "function badge(t,c){c=c||clsFor(t);return'<span class=\"badge '+c+'\">'+(t||'—')+'</span>';}\n"
    "function scoreBar(v,c){c=c||'#3b82f6';if(v===null||v===undefined||isNaN(Number(v)))return'';var p=Math.max(0,Math.min(100,Number(v)));return'<div class=\"score-bar-wrap\"><div class=\"score-bar\" style=\"width:'+p+'%;background:'+c+'\"></div></div>';}\n"
    "function qualityColor(s){s=Number(s);if(s>=85)return'#86efac';if(s>=75)return'#4ade80';if(s>=65)return'#a3e635';if(s>=50)return'#fde68a';return'#fca5a5';}\n"
    "function valuationColor(s){s=Number(s);if(s>=80)return'#86efac';if(s>=65)return'#4ade80';if(s>=45)return'#fde68a';if(s>=25)return'#fb923c';return'#fca5a5';}\n"
    "function showTab(id,ev){document.querySelectorAll('.section').forEach(function(s){s.classList.remove('active');});document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});document.getElementById(id).classList.add('active');if(ev&&ev.target)ev.target.classList.add('active');}\n"
    "async function loadData(){allAssets=await fetch('data/all_assets.json').then(function(r){return r.json();});signals=await fetch('data/signals.json').then(function(r){return r.json();});summary=await fetch('data/summary.json').then(function(r){return r.json();});renderSummary();renderGlobal();renderSignals();renderUniverse();renderPortfolio();buildDeepDiveTickers();}\n"
    "function renderSummary(){document.getElementById('subtitle').innerText='Ultima actualizacion: '+(summary.generated_at||'—')+' - Activos analizados: '+(summary.total_assets||0);var cards=[['Macro',(summary.macro&&summary.macro.label)||'—',(summary.macro&&summary.macro.warning_text)||('Score: '+fmtNum(summary.macro&&summary.macro.score,1))],['EUR/USD',(summary.fx&&summary.fx.label)||'—',(summary.fx&&summary.fx.diag)||'—'],['Senales',String(summary.total_signals||0),'Compras: '+(summary.buy_signals||0)+' - Ventas: '+(summary.sell_signals||0)],['Parametros',((summary.config&&summary.config.LOOKBACK_SIGNAL)||5)+' velas','PVI '+((summary.config&&summary.config.PVI_MA)||120)+' - RVOL '+((summary.config&&summary.config.RVOL_HIGH)||1.5)+'x']];document.getElementById('summaryCards').innerHTML=cards.map(function(c){return'<div class=\"card\"><div class=\"metric-title\">'+c[0]+'</div><div class=\"metric-big\">'+c[1]+'</div><div class=\"small\">'+c[2]+'</div></div>';}).join('');}\n"
    "function renderGlobal(){var rows=((summary.macro&&summary.macro.rows)||[]).map(function(r){return'<tr><td>'+r.id+'</td><td>'+fmtNum(r.value,2)+'</td><td>'+fmtPct(r.roc5,1)+'</td><td>'+fmtPct(r.roc20,1)+'</td><td>'+fmtNum(r.impact,1)+'</td><td>'+(r.diag||'—')+'</td></tr>';}).join('');document.getElementById('macroTable').innerHTML='<table><thead><tr><th>Indicador</th><th>Valor</th><th>ROC 5D</th><th>ROC 20D</th><th>Impacto</th><th>Diagnostico</th></tr></thead><tbody>'+rows+'</tbody></table>';var fx=summary.fx||{};document.getElementById('fxBox').innerHTML='<p>'+badge(fx.label||'—')+'</p><p>'+(fx.diag||'—')+'</p><table><tbody><tr><td>EUR/USD</td><td>'+fmtNum(fx.eurusd,4)+'</td></tr><tr><td>ROC 5D</td><td>'+fmtPct(fx.roc5,2)+'</td></tr><tr><td>ROC 20D</td><td>'+fmtPct(fx.roc20,2)+'</td></tr><tr><td>SMA 50</td><td>'+fmtNum(fx.sma50,4)+'</td></tr><tr><td>Score</td><td>'+fmtNum(fx.score,1)+'</td></tr></tbody></table>';}\n"
    "function fundBlock(a){if(!a.has_fundamentals)return'<div class=\"small\">Sin fundamentales</div>';return'<div>'+badge('Calidad '+fmtNum(a.quality_score,0)+' - '+(a.quality_label||'—'),'qbadge')+badge('Precio/Fund. '+fmtNum(a.valuation_score,0)+' - '+(a.valuation_label||'—'),'vbadge')+badge('Conf '+(a.confidence_grade||'—'),'neutral')+badge(a.quality_q||'—','qbadge')+'</div>'+scoreBar(a.quality_score,qualityColor(a.quality_score))+scoreBar(a.valuation_score,valuationColor(a.valuation_score))+'<div class=\"small\">'+(a.fundamental_profile||'—')+' - '+(a.fundamental_trend||'—')+'</div><div class=\"small\">ROIC '+fmtPct(a.roic,1)+' - ROE '+fmtPct(a.roe,1)+' - FCF Yield '+fmtPct(a.fcf_yield,1)+' - Altman '+fmtNum(a.altman_z,2)+' - Piotroski '+fmtNum(a.piotroski,1)+'</div><div class=\"small\">P/E '+fmtNum(a.pe,1)+' - P/B '+fmtNum(a.pb,2)+' - EV/EBITDA '+fmtNum(a.ev_ebitda,1)+' - ND/EBITDA '+fmtNum(a.net_debt_ebitda,2)+'</div>'+(a.red_flags?'<div class=\"small warning\">'+a.red_flags+'</div>':'');}\n"
    "function techBlock(a){var mw=(summary.macro&&summary.macro.warning&&String(a.main_signal||'').indexOf('COMPRA')>=0)?badge('Macro cautelosa','warnbadge'):'';var vb=a.raw_buy_blocked_by_vol?'<div class=\"small warning\">Senal bloqueada por volumen</div>':'';return'<div>'+badge(a.technical_state||'—')+' '+badge(a.signal_freshness||'—','neutral')+' '+mw+'</div><div class=\"small\"><b>Entrada:</b> '+(a.entry_quality_label||'—')+(a.entry_quality_score!=null?' - '+fmtNum(a.entry_quality_score,0)+'/100':'')+'</div><div class=\"small\"><b>Oportunidad:</b> '+(a.opportunity_label||'—')+(a.opportunity_score!=null?' - '+fmtNum(a.opportunity_score,0)+'/100':'')+'</div><div class=\"small\">'+(a.entry_quality_notes||'')+'</div><div class=\"small\">Salida: <b>'+(a.exit_pressure_label||'—')+'</b> - '+fmtNum(a.exit_pressure_score,0)+'/100</div><div class=\"small\">'+(a.exit_pressure_notes||'')+'</div><div class=\"small\">PVI '+(a.pvi_status||'—')+' - Gap '+fmtPct(a.pvi_gap,2)+' - RVOL '+fmtNum(a.rvol,2)+'x'+(a.bullish_high_volume?' - Vol alto alcista':'')+(a.bearish_high_volume?' - Vol alto bajista':'')+'</div><div class=\"small\">Dist McG '+fmtPct(a.dist_to_mcg_exit,1)+' - ATR '+fmtNum(a.dist_to_mcg_exit_atr,2)+'x - CHOP '+fmtNum(a.chop,1)+'<br>Stop: '+(a.stop_status||'—')+'</div>'+(a.volume_quality===false?'<div class=\"small warning\">'+(a.volume_warning||'Volumen dudoso')+'</div>':'')+vb;}\n"
    "function signalRow(a){return'<tr><td><div class=\"ticker\">'+a.ticker+'</div><div class=\"name\">'+(a.name||'—')+'</div></td><td>'+badge(a.main_signal)+'<div class=\"small\">'+(a.events_text||'')+'</div></td><td>'+badge(a.regime)+'<div class=\"small\">Cruces: '+(a.recent_crosses!=null?a.recent_crosses:'—')+'</div></td><td>'+techBlock(a)+'</td><td>'+fundBlock(a)+'</td><td>'+(a.bucket||'—')+'</td></tr>';}\n"
    "function renderSignals(){var q=(document.getElementById('signalSearch').value||'').toUpperCase();var tp=document.getElementById('signalType').value||'';var rg=document.getElementById('signalRegime').value||'';var data=signals.slice();if(q)data=data.filter(function(a){return String(a.ticker).toUpperCase().indexOf(q)>=0||String(a.name).toUpperCase().indexOf(q)>=0;});if(tp)data=data.filter(function(a){return String(a.main_signal||'').indexOf(tp)>=0;});if(rg)data=data.filter(function(a){return String(a.regime||'').indexOf(rg)>=0;});if(!data.length){document.getElementById('signalsTable').innerHTML='<p class=\"small\">No hay senales.</p>';return;}document.getElementById('signalsTable').innerHTML='<table><thead><tr><th>Activo</th><th>Senal</th><th>Regimen</th><th>Tecnico</th><th>Fundamental</th><th>Universo</th></tr></thead><tbody>'+data.map(signalRow).join('')+'</tbody></table>';}\n"
    "function confVal(c){if(c==='A')return 3;if(c==='B')return 2;if(c==='C')return 1;return 0;}\n"
    "function renderUniverse(){var q=(document.getElementById('universeSearch').value||'').toUpperCase();var rg=document.getElementById('universeRegime').value||'';var cf=document.getElementById('fundConfidenceFilter').value||'';var qm=document.getElementById('fundQualityFilter').value;var vm=document.getElementById('fundValuationFilter').value;var em=document.getElementById('entryQualityFilter').value;var xm=document.getElementById('exitPressureFilter').value;var qf=document.getElementById('fundQFilter').value||'';var of=document.getElementById('onlyFundamentals').checked||false;var data=allAssets.slice();if(q)data=data.filter(function(a){return String(a.ticker).toUpperCase().indexOf(q)>=0||String(a.name).toUpperCase().indexOf(q)>=0||String(a.sector||'').toUpperCase().indexOf(q)>=0||String(a.industry||'').toUpperCase().indexOf(q)>=0;});if(rg)data=data.filter(function(a){return String(a.regime||'').indexOf(rg)>=0;});if(of)data=data.filter(function(a){return a.has_fundamentals===true;});if(cf)data=data.filter(function(a){return confVal(a.confidence_grade)>=confVal(cf);});if(qm)data=data.filter(function(a){return Number(a.quality_score)>=Number(qm);});if(vm)data=data.filter(function(a){return Number(a.valuation_score)>=Number(vm);});if(em)data=data.filter(function(a){return Number(a.entry_quality_score)>=Number(em);});if(xm)data=data.filter(function(a){return Number(a.exit_pressure_score)>=Number(xm);});if(qf)data=data.filter(function(a){return String(a.quality_q||'')===qf;});data=data.sort(function(a,b){var fa=(a.quality_score==null)?-1:Number(a.quality_score);var fb=(b.quality_score==null)?-1:Number(b.quality_score);if(fb!==fa)return fb-fa;return String(a.ticker).localeCompare(String(b.ticker));});var rows=data.map(function(a){return'<tr><td><div class=\"ticker\">'+a.ticker+'</div><div class=\"name\">'+(a.name||'—')+'</div></td><td>'+badge(a.main_signal||'—')+'</td><td>'+badge(a.regime||'—')+'<div class=\"small\">'+(a.technical_state||'—')+'</div></td><td>'+fmtNum(a.close,2)+'</td><td>'+techBlock(a)+'</td><td>'+fundBlock(a)+'</td><td>'+(a.bucket||'—')+'</td></tr>';}).join('');document.getElementById('universeTable').innerHTML='<table><thead><tr><th>Activo</th><th>Senal</th><th>Regimen</th><th>Precio</th><th>Tecnico</th><th>Fundamental</th><th>Universo</th></tr></thead><tbody>'+rows+'</tbody></table>';}\n"
    "function buildDeepDiveTickers(){var ws=allAssets.filter(function(a){return a.has_signal;}).map(function(a){return a.ticker;});var rs=allAssets.filter(function(a){return!a.has_signal;}).map(function(a){return a.ticker;});var all=ws.concat(rs);var el=document.getElementById('dd-ticker-list');var html=all.slice(0,40).map(function(t){var a=allAssets.find(function(x){return x.ticker===t;});var c=(a&&a.has_signal)?'primary':'';return'<button class=\"'+c+'\" onclick=\"setAndLoad(\\''+t+'\\')\">' +t+'</button>';}).join('');if(all.length>40)html+='<span class=\"small\"> ...y '+(all.length-40)+' mas</span>';el.innerHTML=html;}\n"
    "function setAndLoad(t){document.getElementById('dd-ticker-input').value=t;loadDeepDive();}\n"
    "async function loadDeepDive(){var ticker=document.getElementById('dd-ticker-input').value.trim().toUpperCase();if(!ticker)return;var loading=document.getElementById('dd-loading');var body=document.getElementById('dd-body');loading.innerText='Cargando datos de '+ticker+'...';body.style.display='none';_ddCharts.forEach(function(c){try{c.remove();}catch(e){}});_ddCharts=[];var data=null;try{var resp=await fetch('data/tickers/'+ticker+'.json');if(resp.ok)data=await resp.json();}catch(e){}if(!data){var asset=allAssets.find(function(a){return String(a.ticker).toUpperCase()===ticker;});if(!asset){loading.innerHTML='<div style=\"padding:16px;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:10px\"><b>'+ticker+'</b> no esta en el universo analizado.</div>';return;}data={ticker:asset.ticker,name:asset.name||ticker,bucket:asset.bucket||'—',currency:asset.currency||'—',sector:asset.sector||'—',last_date:asset.last_date||'—',has_chart_data:false,chart_error:'JSON de grafico no disponible (datos insuficientes).',candles:[],mcg_regime:[],mcg_exit:[],pvi:[],pvi_signal:[],atr:[],rvol:[],chop:[],dist_mcg_exit:[],markers:[],technical:asset,fundamental:asset.has_fundamentals?asset:null};}loading.innerText='';body.style.display='block';renderDeepHeader(data);renderDeepSignalCards(data);renderDeepCharts(data);renderDeepTechKV(data);renderDeepFundamentals(data);}\n"
    "function renderDeepHeader(d){var t=d.technical||{};document.getElementById('dd-asset-header').innerHTML='<div style=\"display:flex;align-items:center;gap:18px;flex-wrap:wrap\"><div><div style=\"font-size:32px;font-weight:950\">'+d.ticker+'</div><div style=\"color:var(--muted);font-size:14px\">'+(d.name||'')+' - '+(d.bucket||'')+' - '+(d.sector||'')+' - '+(d.currency||'')+'</div><div class=\"small\">Ultimo dato: '+(d.last_date||'—')+'</div></div><div>'+badge(t.main_signal||'—')+' '+badge(t.regime||'—')+' '+badge(t.signal_freshness||'—','neutral')+' '+badge(t.technical_state||'—')+'</div><div><div style=\"font-size:28px;font-weight:950;color:var(--blue)\">'+fmtNum(t.close,2)+' '+(d.currency||'')+'</div><div class=\"small\">McG Reg '+fmtNum(t.mcg_regime,2)+' - McG Salida '+fmtNum(t.mcg_exit,2)+'</div></div></div>';}\n"
    "function renderDeepSignalCards(d){var t=d.technical||{};var mw=(summary.macro&&summary.macro.warning&&String(t.main_signal||'').indexOf('COMPRA')>=0)?badge('Macro cautelosa','warnbadge'):'';var agoSub=[t.buy_ago!=null?'Compra: hace '+t.buy_ago+' velas':null,t.pvi_sell_ago!=null?'Venta PVI: hace '+t.pvi_sell_ago+' velas':null,t.mcg_sell_ago!=null?'Venta McG: hace '+t.mcg_sell_ago+' velas':null].filter(Boolean).join(' - ')||'—';var ec=t.entry_quality_score!=null?qualityColor(t.entry_quality_score):'#94a3b8';var oc=t.opportunity_score!=null?qualityColor(t.opportunity_score):'#94a3b8';var xc='#22c55e';if(t.exit_pressure_score>=75)xc='#ef4444';else if(t.exit_pressure_score>=50)xc='#f97316';else if(t.exit_pressure_score>=25)xc='#fde68a';var sc='#94a3b8';if(String(t.main_signal||'').indexOf('COMPRA')>=0)sc='#22c55e';if(String(t.main_signal||'').indexOf('VENTA')>=0)sc='#ef4444';var cards=[{title:'Senal principal',value:t.main_signal||'—',sub:t.events_text||'Sin eventos',color:sc},{title:'Frescura - Velas',value:t.signal_freshness||'—',sub:agoSub,color:'#60a5fa'},{title:'Calidad de entrada',value:t.entry_quality_score!=null?fmtNum(t.entry_quality_score,0)+'/100':'—',sub:t.entry_quality_label||'—',bar:t.entry_quality_score,barColor:ec},{title:'Oportunidad global',value:t.opportunity_score!=null?fmtNum(t.opportunity_score,0)+'/100':'—',sub:t.opportunity_label||'—',bar:t.opportunity_score,barColor:oc},{title:'Presion de salida',value:t.exit_pressure_score!=null?fmtNum(t.exit_pressure_score,0)+'/100':'—',sub:t.exit_pressure_label||'—',bar:t.exit_pressure_score,barColor:xc},{title:'Estado stop',value:t.stop_status||'—',sub:'Dist '+fmtPct(t.dist_to_mcg_exit,1)+' - '+fmtNum(t.dist_to_mcg_exit_atr,2)+' ATR',color:'#a78bfa'},{title:'PVI',value:t.pvi_status||'—',sub:'Gap '+fmtPct(t.pvi_gap,2)+' - PVI '+fmtNum(t.pvi,2)+' - Senal '+fmtNum(t.pvi_signal,2),color:t.pvi_status==='POSITIVO'?'#22c55e':'#ef4444'},{title:'RVOL - CHOP - ATR',value:fmtNum(t.rvol,2)+'x',sub:'CHOP '+fmtNum(t.chop,1)+' - ATR '+fmtNum(t.atr,2)+' '+(d.currency||''),color:Number(t.rvol)>=1.5?'#22c55e':'#94a3b8'}];document.getElementById('dd-signal-cards').innerHTML=cards.map(function(c){return'<div class=\"card\" style=\"border-left:3px solid '+(c.color||'#334155')+'\"><div class=\"kv-label\">'+c.title+'</div><div class=\"metric-val\" style=\"color:'+(c.color||'var(--text)')+'\">'+c.value+'</div>'+(c.bar!=null?scoreBar(c.bar,c.barColor||'#3b82f6'):'')+' <div class=\"small\" style=\"margin-top:6px\">'+c.sub+'</div></div>';}).join('')+mw;document.getElementById('dd-signal-timeline').innerHTML=[t.entry_quality_notes?badge('Entrada: '+t.entry_quality_notes,'neutral'):'',t.exit_pressure_notes?badge('Salida: '+t.exit_pressure_notes,'neutral'):'',t.volume_warning?badge(t.volume_warning,'warnbadge'):'',(summary.macro&&summary.macro.warning&&String(t.main_signal||'').indexOf('COMPRA')>=0)?badge('Macro cautelosa','warnbadge'):''].filter(Boolean).join('');}\n"
    "var DD_CHART_OPT=function(h){return{width:0,height:h||400,layout:{background:{color:'#0f172a'},textColor:'#94a3b8'},grid:{vertLines:{color:'rgba(148,163,184,.08)'},horzLines:{color:'rgba(148,163,184,.08)'}},crosshair:{mode:1},rightPriceScale:{borderColor:'rgba(148,163,184,.2)'},timeScale:{borderColor:'rgba(148,163,184,.2)',timeVisible:true,secondsVisible:false},handleScroll:true,handleScale:true};};\n"
    "function newChart(id,h){var el=document.getElementById(id);el.innerHTML='';var chart=LightweightCharts.createChart(el,DD_CHART_OPT(h));chart.applyOptions({width:el.clientWidth});new ResizeObserver(function(){chart.applyOptions({width:el.clientWidth});}).observe(el);_ddCharts.push(chart);return chart;}\n"
    "function renderDeepCharts(d){if(!d.has_chart_data){var msg='<div class=\"chart-placeholder\">'+(d.chart_error||'Sin datos de grafico')+'</div>';['dd-chart-price','dd-chart-pvi','dd-chart-rvol','dd-chart-dist'].forEach(function(id){var el=document.getElementById(id);if(el)el.innerHTML=msg;});return;}renderDeepChartPrice(d);renderDeepChartPVI(d);renderDeepChartRVOL(d);renderDeepChartDist(d);}\n"
    "function renderDeepChartPrice(d){var chart=newChart('dd-chart-price',400);var cs=chart.addCandlestickSeries({upColor:'#22c55e',downColor:'#ef4444',borderUpColor:'#22c55e',borderDownColor:'#ef4444',wickUpColor:'#22c55e',wickDownColor:'#ef4444'});var vc=(d.candles||[]).filter(function(c){return isFinite(c.open)&&isFinite(c.high)&&isFinite(c.low)&&isFinite(c.close);});cs.setData(vc);var vs=chart.addHistogramSeries({color:'rgba(148,163,184,.18)',priceFormat:{type:'volume'},priceScaleId:'vol'});chart.priceScale('vol').applyOptions({scaleMargins:{top:0.85,bottom:0}});vs.setData((d.candles||[]).map(function(c){return{time:c.time,value:c.volume,color:c.close>=c.open?'rgba(34,197,94,.25)':'rgba(239,68,68,.25)'};}));var mr=chart.addLineSeries({color:'#3b82f6',lineWidth:2,title:'McG Reg'});mr.setData(d.mcg_regime||[]);var me=chart.addLineSeries({color:'#f97316',lineWidth:2,lineStyle:1,title:'McG Exit'});me.setData(d.mcg_exit||[]);var mm={buy:'#22c55e',sell_pvi:'#ef4444',sell_mcg:'#f97316'};var mk=(d.markers||[]).map(function(m){return{time:m.time,position:m.position,color:mm[m.type]||'#94a3b8',shape:m.shape,text:m.type==='buy'?'B':m.type==='sell_pvi'?'VP':'VM',size:1};});cs.setMarkers(mk);chart.timeScale().fitContent();}\n"
    "function renderDeepChartPVI(d){var chart=newChart('dd-chart-pvi',180);var ps=chart.addLineSeries({color:'#60a5fa',lineWidth:2,title:'PVI'});ps.setData(d.pvi||[]);var ss=chart.addLineSeries({color:'#f59e0b',lineWidth:2,lineStyle:2,title:'Senal'});ss.setData(d.pvi_signal||[]);var bm=(d.markers||[]).filter(function(m){return m.type==='buy';}).map(function(m){return{time:m.time,position:'belowBar',color:'#22c55e',shape:'arrowUp',size:1};});var sm=(d.markers||[]).filter(function(m){return m.type==='sell_pvi';}).map(function(m){return{time:m.time,position:'aboveBar',color:'#ef4444',shape:'arrowDown',size:1};});var am=bm.concat(sm).sort(function(a,b){return a.time>b.time?1:-1;});ps.setMarkers(am);chart.timeScale().fitContent();}\n"
    "function renderDeepChartRVOL(d){var chart=newChart('dd-chart-rvol',140);var rv=chart.addHistogramSeries({title:'RVOL'});rv.setData((d.rvol||[]).map(function(v){return{time:v.time,value:v.value,color:v.value>=1.5?'rgba(34,197,94,.7)':v.value>=1.0?'rgba(148,163,184,.5)':'rgba(148,163,184,.25)'};}));var tl=chart.addLineSeries({color:'#f97316',lineWidth:1,lineStyle:2,title:'1.5x'});tl.setData((d.rvol||[]).map(function(v){return{time:v.time,value:1.5};}));var ol=chart.addLineSeries({color:'rgba(148,163,184,.4)',lineWidth:1,lineStyle:3});ol.setData((d.rvol||[]).map(function(v){return{time:v.time,value:1.0};}));chart.timeScale().fitContent();}\n"
    "function renderDeepChartDist(d){var chart=newChart('dd-chart-dist',140);var ds=chart.addHistogramSeries({title:'Dist %'});ds.setData((d.dist_mcg_exit||[]).map(function(v){return{time:v.time,value:v.value,color:v.value>=0?'rgba(34,197,94,.55)':'rgba(239,68,68,.55)'};}));var zl=chart.addLineSeries({color:'rgba(148,163,184,.4)',lineWidth:1,lineStyle:2});zl.setData((d.dist_mcg_exit||[]).map(function(v){return{time:v.time,value:0};}));chart.timeScale().fitContent();}\n"
    "function renderDeepTechKV(d){var t=d.technical||{};var items=[['Precio',fmtNum(t.close,2)+' '+(d.currency||'')],['McG Regimen',fmtNum(t.mcg_regime,2)],['McG Salida',fmtNum(t.mcg_exit,2)],['ATR',fmtNum(t.atr,4)],['CHOP',fmtNum(t.chop,1)],['RVOL',fmtNum(t.rvol,2)+'x'],['PVI',fmtNum(t.pvi,2)],['PVI Senal',fmtNum(t.pvi_signal,2)],['PVI Gap',fmtPct(t.pvi_gap,2)],['PVI Estado',t.pvi_status||'—'],['Dist McG %',fmtPct(t.dist_to_mcg_exit,2)],['Dist McG ATR',fmtNum(t.dist_to_mcg_exit_atr,2)+'x'],['Regimen',t.regime||'—'],['Cruces laterales',String(t.recent_crosses!=null?t.recent_crosses:'—')],['Vol calidad',t.volume_quality?'OK':'DUDOSO'],['Vol 0 dias',String(t.volume_zero_days!=null?t.volume_zero_days:'—')],['RVOL alto',t.rvol_high?'Si':'—'],['Vol alcista',t.bullish_high_volume?'Si':'—'],['Vol bajista',t.bearish_high_volume?'Si':'—'],['Senal',t.main_signal||'—'],['Compra hace',t.buy_ago!=null?t.buy_ago+' velas':'—'],['Venta PVI hace',t.pvi_sell_ago!=null?t.pvi_sell_ago+' velas':'—'],['Venta McG hace',t.mcg_sell_ago!=null?t.mcg_sell_ago+' velas':'—'],['Stop',t.stop_status||'—'],['Ultima vela',d.last_date||'—']];document.getElementById('dd-tech-kv').innerHTML=items.map(function(i){return'<div class=\"kv-item\"><div class=\"kv-label\">'+i[0]+'</div><div class=\"kv-value\">'+i[1]+'</div></div>';}).join('');}\n"
    "function renderDeepFundamentals(d){var el=document.getElementById('dd-fund-section');if(!d.fundamental){el.innerHTML='<div class=\"dd-section-title\">Fundamentales</div><div class=\"small\" style=\"padding:12px 0\">Sin datos fundamentales (tecnico-only, cripto, commodity o confianza insuficiente).</div>';return;}var f=d.fundamental;function kv(l,v){return'<div class=\"kv-item\"><div class=\"kv-label\">'+l+'</div><div class=\"kv-value\">'+v+'</div></div>';}function altZ(v){if(!isFinite(v))return'';if(v<1.8)return' PELIGRO';if(v<3.0)return' GRIS';return' SANO';}function benM(v){if(!isFinite(v))return'';if(v>-1.78)return' SOSPECHOSO';if(v>-2.22)return' REVISAR';return' OK';}el.innerHTML='<div class=\"dd-section-title\">Fundamentales - '+(f.fundamental_model||'—')+' - '+(f.period_label||'N/A')+' - '+(f.reporting_frequency||'')+' - Conf '+(f.confidence_grade||'—')+'</div>'+'<div class=\"grid4\" style=\"margin-bottom:14px\">'+'<div class=\"card\"><div class=\"kv-label\">Calidad del negocio</div><div class=\"metric-val\" style=\"color:'+qualityColor(f.quality_score)+'\">'+fmtNum(f.quality_score,0)+'/100</div>'+scoreBar(f.quality_score,qualityColor(f.quality_score))+'<div class=\"small\">'+(f.quality_label||'—')+'</div></div>'+'<div class=\"card\"><div class=\"kv-label\">Precio/Fundamentales</div><div class=\"metric-val\" style=\"color:'+valuationColor(f.valuation_score)+'\">'+fmtNum(f.valuation_score,0)+'/100</div>'+scoreBar(f.valuation_score,valuationColor(f.valuation_score))+'<div class=\"small\">'+(f.valuation_label||'—')+'</div></div>'+'<div class=\"card\"><div class=\"kv-label\">Perfil fundamental</div><div class=\"metric-val\">'+(f.fundamental_profile||'—')+'</div><div class=\"small\">'+(f.fundamental_trend||'—')+'</div></div>'+'<div class=\"card\"><div class=\"kv-label\">Confianza datos</div><div class=\"metric-val\">'+(f.confidence_grade||'—')+' - '+fmtNum(f.confidence_score,0)+'/100</div><div class=\"small\">'+(f.available_fields||'?')+'/'+(f.expected_fields||'?')+' campos - '+(f.latest_report_date||'—')+'</div></div></div>'+'<div class=\"dd-section-title\">Sub-scores de calidad</div><div class=\"grid4\" style=\"margin-bottom:14px\">'+[['Calidad operativa',f.score_quality],['Generacion de caja',f.score_cash],['Solvencia',f.score_solvency],['Crecimiento',f.score_growth],['Riesgo',f.score_risk],['Valoracion',f.score_valuation]].map(function(x){return'<div class=\"kv-item\"><div class=\"kv-label\">'+x[0]+'</div><div class=\"kv-value\">'+fmtNum(x[1],0)+'</div>'+scoreBar(x[1],qualityColor(x[1]))+'</div>';}).join('')+'</div>'+'<div class=\"dd-section-title\">P&amp;L TTM</div><div class=\"kv-grid\" style=\"margin-bottom:14px\">'+[['Revenue TTM',fmtNum((f.revenue_ttm||0)/1e9,2)+'B'],['Gross Profit',fmtNum((f.gross_profit_ttm||0)/1e9,2)+'B'],['Op. Income',fmtNum((f.op_income_ttm||0)/1e9,2)+'B'],['Net Income',fmtNum((f.net_income_ttm||0)/1e9,2)+'B'],['CFO',fmtNum((f.cfo_ttm||0)/1e9,2)+'B'],['CapEx',fmtNum((f.capex_ttm||0)/1e9,2)+'B'],['FCF',fmtNum((f.fcf_ttm||0)/1e9,2)+'B'],['Rev Growth',fmtPct(f.revenue_growth,1)],['Op Inc Growth',fmtPct(f.op_income_growth,1)],['Net Inc Growth',fmtPct(f.net_income_growth,1)],['Shares Growth',fmtPct(f.shares_growth,2)],['Market Cap',fmtNum((f.market_cap||0)/1e9,2)+'B']].map(function(x){return kv(x[0],x[1]);}).join('')+'</div>'+'<div class=\"dd-section-title\">Balance</div><div class=\"kv-grid\" style=\"margin-bottom:14px\">'+[['Total Assets',fmtNum((f.total_assets||0)/1e9,2)+'B'],['Total Liab.',fmtNum((f.total_liabilities||0)/1e9,2)+'B'],['Equity',fmtNum((f.equity||0)/1e9,2)+'B'],['Total Debt',fmtNum((f.total_debt||0)/1e9,2)+'B'],['Cash',fmtNum((f.cash||0)/1e9,2)+'B'],['ND/EBITDA',fmtNum(f.net_debt_ebitda,2)+'x'],['Current Ratio',fmtNum(f.current_ratio,2)+'x'],['Int. Coverage',fmtNum(f.interest_coverage,1)+'x'],['Debt/Equity',fmtNum(f.debt_equity,2)+'x'],['Equity/Assets',fmtPct(f.equity_assets,1)]].map(function(x){return kv(x[0],x[1]);}).join('')+'</div>'+'<div class=\"dd-section-title\">Rentabilidad y margenes</div><div class=\"kv-grid\" style=\"margin-bottom:14px\">'+[['ROIC',fmtPct(f.roic,1)],['ROE',fmtPct(f.roe,1)],['ROA',fmtPct(f.roa,1)],['Gross Margin',fmtPct(f.gross_margin,1)],['Op Margin',fmtPct(f.op_margin,1)],['FCF Margin',fmtPct(f.fcf_margin,1)],['FCF Yield',fmtPct(f.fcf_yield,1)],['Cash Qual.',fmtNum(f.cash_quality,2)],['Consistencia',fmtNum(f.consistency_score,0)+'%']].map(function(x){return kv(x[0],x[1]);}).join('')+'</div>'+'<div class=\"dd-section-title\">Multiplos de valoracion</div><div class=\"kv-grid\" style=\"margin-bottom:14px\">'+[['P/E',fmtNum(f.pe,1)+'x'],['P/B',fmtNum(f.pb,2)+'x'],['P/S',fmtNum(f.ps,2)+'x'],['EV/EBITDA',fmtNum(f.ev_ebitda,1)+'x'],['Upside TP',fmtPct(f.upside,1)],['Val. Style',f.valuation_style||'—']].map(function(x){return kv(x[0],x[1]);}).join('')+'</div>'+'<div class=\"dd-section-title\">Modelos de riesgo</div><div class=\"kv-grid\" style=\"margin-bottom:14px\">'+kv('Piotroski F',fmtNum(f.piotroski,1)+'/9')+kv('Altman Z',fmtNum(f.altman_z,2)+altZ(f.altman_z))+kv('Beneish M',fmtNum(f.beneish_m,3)+benM(f.beneish_m))+'</div>'+(f.red_flags?'<div style=\"padding:12px 16px;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.35);border-radius:10px;margin-bottom:14px\"><span class=\"warning\">Red flags: '+f.red_flags+'</span></div>':'')+'<div class=\"small\">'+(f.model_note||'')+' - '+(f.not_applicable||'')+'</div>';}\n"
    "function findAsset(t){return allAssets.find(function(a){return String(a.ticker).toUpperCase()===String(t).toUpperCase();});}\n"
    "function savePortfolio(){localStorage.setItem('sovereign_portfolio',JSON.stringify(portfolio));}\n"
    "function addPosition(){var t=document.getElementById('pfTicker').value.trim().toUpperCase();var q=parseFloat(document.getElementById('pfQty').value);var p=parseFloat(document.getElementById('pfPrice').value);var dt=document.getElementById('pfDate').value;var n=document.getElementById('pfNote').value;if(!t||!q||!p){alert('Ticker, cantidad y precio son obligatorios.');return;}var a=findAsset(t);portfolio.push({ticker:t,name:a?a.name:t,quantity:q,buy_price:p,buy_date:dt,note:n});savePortfolio();document.getElementById('pfTicker').value='';document.getElementById('pfQty').value='';document.getElementById('pfPrice').value='';document.getElementById('pfNote').value='';renderPortfolio();}\n"
    "function deletePosition(i){portfolio.splice(i,1);savePortfolio();renderPortfolio();}\n"
    "function clearPortfolio(){if(!confirm('Vaciar cartera?'))return;portfolio=[];savePortfolio();renderPortfolio();}\n"
    "function systemAction(a){if(!a)return['SIN DATOS','Sin datos.'];var p=Number(a.exit_pressure_score||0);var s=a.main_signal||'';if(s.indexOf('VENTA 100')>=0||p>=75)return['VENDER TODO','Presion fuerte.'];if(p>=50)return['REDUCIR','Presion elevada.'];if(p>=25)return['VIGILAR','Deterioro parcial.'];if(String(a.technical_state||'').indexOf('Alcista confirmado')>=0)return['MANTENER','Estado favorable.'];return['MANTENER/VIGILAR','Sin senal operativa.'];}\n"
    "function renderPortfolio(){if(!portfolio.length){document.getElementById('portfolioSummary').innerHTML='';document.getElementById('portfolioTable').innerHTML='<p class=\"small\">Sin posiciones guardadas.</p>';return;}var tv=0,tc=0;var rows=portfolio.map(function(p,i){var a=findAsset(p.ticker);var pr=a?Number(a.close):null;var q=Number(p.quantity);var b=Number(p.buy_price);var v=pr?q*pr:null;var c=q*b;var pnl=v!==null?v-c:null;var pp=(v!==null&&c)?v/c-1:null;if(v!==null)tv+=v;tc+=c;var sa=systemAction(a);return'<tr><td><div class=\"ticker\">'+p.ticker+'</div><div class=\"name\">'+(p.name||(a?a.name:''))+'</div>'+(a&&a.has_fundamentals?'<div class=\"small\">Cal '+fmtNum(a.quality_score,0)+' - Val '+fmtNum(a.valuation_score,0)+' - '+(a.fundamental_trend||'—')+'</div>':'')+(a&&a.opportunity_score!=null?'<div class=\"small\">Oport. '+fmtNum(a.opportunity_score,0)+' - '+(a.opportunity_label||'—')+'</div>':'')+'</td><td>'+fmtNum(q,4)+'</td><td>'+fmtNum(b,2)+'</td><td>'+(pr?fmtNum(pr,2):'—')+'</td><td>'+(pp!==null?fmtPct(pp,2):'—')+'<div class=\"small\">'+(pnl!==null?fmtNum(pnl,2):'—')+'</div></td><td>'+(a?badge(a.main_signal||'—'):'—')+'<div class=\"small\">'+(a?a.events_text||'':'')+'</div></td><td>'+(a?badge(a.exit_pressure_label||'—'):'—')+'<div class=\"small\">'+(a?a.exit_pressure_notes||'':'')+'</div></td><td>'+badge(sa[0])+'<div class=\"small\">'+sa[1]+'</div></td><td>'+(p.buy_date||'—')+'<div class=\"small\">'+(p.note||'')+'</div></td><td><button class=\"danger\" onclick=\"deletePosition('+i+')\">Borrar</button></td></tr>';}).join('');var tp=tv-tc;var tpp=tc?tv/tc-1:null;document.getElementById('portfolioSummary').innerHTML='<div class=\"grid\"><div class=\"card\"><div class=\"metric-title\">Valor cartera</div><div class=\"metric-big\">'+fmtNum(tv,2)+'</div></div><div class=\"card\"><div class=\"metric-title\">Coste</div><div class=\"metric-big\">'+fmtNum(tc,2)+'</div></div><div class=\"card\"><div class=\"metric-title\">PnL</div><div class=\"metric-big\">'+fmtNum(tp,2)+'</div><div class=\"small\">'+fmtPct(tpp,2)+'</div></div><div class=\"card\"><div class=\"metric-title\">Posiciones</div><div class=\"metric-big\">'+portfolio.length+'</div></div></div>';document.getElementById('portfolioTable').innerHTML='<table><thead><tr><th>Activo</th><th>Cantidad</th><th>Compra</th><th>Actual</th><th>PnL</th><th>Senal</th><th>Presion salida</th><th>Accion sistema</th><th>Fecha/Nota</th><th></th></tr></thead><tbody>'+rows+'</tbody></table>';}\n"
    "function exportPortfolio(){var b=new Blob([JSON.stringify(portfolio,null,2)],{type:'application/json'});var u=URL.createObjectURL(b);var a=document.createElement('a');a.href=u;a.download='sovereign_portfolio.json';a.click();}\n"
    "function importPortfolio(ev){var f=ev.target.files[0];if(!f)return;var r=new FileReader();r.onload=function(e){try{portfolio=JSON.parse(e.target.result);savePortfolio();renderPortfolio();}catch(ex){alert('JSON invalido.');}};r.readAsText(f);}\n"
    "loadData();\n"
    "</script>\n"
    "</body>\n"
    "</html>\n"
)


# ============================================================
# MAIN
# ============================================================

def main():
    print("LCrack Sovereign")
    print(f"   Modo filtro volumen: {CONFIG.get('VOL_FILTER_MODE')} - RVOL>={CONFIG['RVOL_HIGH']}x")

    site_dir = Path("site")
    data_dir = site_dir / "data"
    site_dir.mkdir(exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    universe_df = build_universe_df()
    print(f"Universo: {len(universe_df)} activos")

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
                    print(
                        f"Senal: {ticker} -> {res.get('main_signal')} - "
                        f"{res.get('entry_quality_label','—')} - "
                        f"{res.get('exit_pressure_label','—')}"
                    )
            except Exception as e:
                print(f"Error tecnico {ticker}: {e}")

    assets_df = pd.DataFrame(tech_rows)
    assets_df = add_fundamentals(assets_df, universe_df)

    if not assets_df.empty:
        assets_df["sort_signal"] = assets_df["bars_min"].fillna(999)
        assets_df = assets_df.sort_values(["sort_signal", "ticker"], ascending=[True, True])

    # Deep Dive JSONs
    print("Generando Deep Dive por ticker...")
    tickers_dir = data_dir / "tickers"
    tickers_dir.mkdir(parents=True, exist_ok=True)

    fund_lookup = {}
    if not assets_df.empty and "has_fundamentals" in assets_df.columns:
        for _, row in assets_df[assets_df["has_fundamentals"] == True].iterrows():
            fund_lookup[row["ticker"]] = row.to_dict()

    def save_ticker_detail(row):
        t = row["ticker"]
        try:
            tech = row.to_dict()
            fund = fund_lookup.get(t)
            detail = make_ticker_detail_json(t, tech, fund)
            (tickers_dir / f"{t}.json").write_text(
                json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8"
            )
            if not detail["has_chart_data"]:
                print(f"   Sin grafico {t}: {detail['chart_error']}")
            return t
        except Exception as e:
            print(f"   Error Deep Dive {t}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures_dd = {
            executor.submit(save_ticker_detail, row): row["ticker"]
            for _, row in assets_df.iterrows()
        }
        saved = sum(1 for fut in as_completed(futures_dd) if fut.result())

    print(f"   {saved} JSONs de ticker guardados")

    signals_df = (
        assets_df[assets_df["has_signal"] == True].copy()
        if not assets_df.empty else pd.DataFrame()
    )

    macro = macro_context()
    fx = currency_context()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buy_signals = 0
    sell_signals = 0
    if not signals_df.empty:
        buy_signals  = int(signals_df["main_signal"].astype(str).str.contains("COMPRA").sum())
        sell_signals = int(signals_df["main_signal"].astype(str).str.contains("VENTA").sum())

    summary_data = {
        "generated_at":  generated_at,
        "total_assets":  int(len(assets_df)),
        "total_signals": int(len(signals_df)),
        "buy_signals":   buy_signals,
        "sell_signals":  sell_signals,
        "macro": macro,
        "fx":    fx,
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
        json.dumps(summary_data, ensure_ascii=False, indent=2, default=clean_json_value),
        encoding="utf-8"
    )
    (site_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    print("Site generado en /site")
    print(f"Ultima actualizacion: {generated_at}")


if __name__ == "__main__":
    main()
