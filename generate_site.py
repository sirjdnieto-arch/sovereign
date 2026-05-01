Te dejo el archivo completo **fusionado** con todas las propuestas aplicadas.

- Base → Mejora B  
- Añadido → Filtro de volumen tipo A como **modo configurable**  
- Añadido → `stop_status` (distancia al McGinley en % y ATR, estilo A)  
- Mantengo → Scores `entry_quality_score` / `exit_pressure_score`, `technical_state`, `CHOP`, valoración contextual por estilo, `fundamental_trend`, macro warning visual, etc.

Por defecto dejo `VOL_FILTER_MODE = "score"` (no bloquea, solo puntúa).  
Si quieres el comportamiento duro de A, cambia a `"hard"`.

```python
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

    # Volumen y contexto
    "RVOL_PERIOD": 30,
    "RVOL_HIGH": 1.5,                 # umbral de RVOL "alto"
    "VOLUME_MIN_NONZERO_PCT": 0.60,   # calidad mínima de volumen

    # Modo de filtro de volumen en compras:
    #   "score" → no bloquea, solo ajusta entry_quality_score (estilo B)
    #   "hard"  → bloquea compras si RVOL<umbral y vol es fiable (estilo A, backtest)
    #   "warn"  → no bloquea, pero marca raw_buy_blocked_by_vol=True
    "VOL_FILTER_MODE": "score",

    "ATR_PERIOD": 14,
    "CHOP_PERIOD": 14,

    "PVI_GAP_MIN_STRONG": 0.001,  # 0.10%

    # Solo para clasificar, no para bloquear (entrada técnica)
    "ENTRY_QUALITY_A": 85,
    "ENTRY_QUALITY_B": 70,
    "ENTRY_QUALITY_C": 55,

    # Presión de salida
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
# TÉCNICO: INDICADORES Y SCORES
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
    """
    PVI:
    - Valor inicial 1000.
    - Solo suma variación cuando Vol actual > Vol anterior.
    - Señal: EMA del PVI.
    """
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
    """
    Control de calidad del volumen.
    No bloquea por sí solo, pero marca si el PVI puede ser poco fiable.
    """
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
        "volume_nonzero_pct": nonzero_pct,  # 0–1
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
        return "🔴 Hoy"
    if bars_ago <= 2:
        return "🟠 Reciente"
    if bars_ago <= 5:
        return "🟡 Esta semana"
    return "⚪ Antigua"


def entry_quality_label_from_score(score, volume_quality):
    """
    Etiqueta de calidad de entrada estilo A, derivada del score 0–100 de B.
    """
    if score is None or pd.isna(score):
        return "—"
    if not volume_quality:
        return "⚪ VOL NO FIABLE"

    s = float(score)
    if s >= CONFIG["ENTRY_QUALITY_A"]:
        return "🟢 ALTA · Compra muy limpia"
    if s >= CONFIG["ENTRY_QUALITY_B"]:
        return "🟢 BUENA · Compra buena"
    if s >= CONFIG["ENTRY_QUALITY_C"]:
        return "🟡 MEDIA · Aceptable"
    return "🟠 BAJA · Señal débil"


def exit_pressure_label(score):
    if score is None or pd.isna(score):
        return "—"
    score = float(score)
    if score >= CONFIG["EXIT_PRESSURE_FUERTE"]:
        return "🔴 Salida fuerte"
    if score >= CONFIG["EXIT_PRESSURE_REDUCIR"]:
        return "🟠 Reducir"
    if score >= CONFIG["EXIT_PRESSURE_VIGILAR"]:
        return "🟡 Vigilar"
    return "🟢 Baja"


def technical_state_label(regime, pvi_status, price_below_mcg_exit):
    if price_below_mcg_exit and pvi_status == "NEGATIVO":
        return "🔴 Deteriorado"
    if price_below_mcg_exit:
        return "🟠 Bajo McGinley"
    if "ALCISTA" in str(regime) and pvi_status == "POSITIVO":
        return "🟢 Alcista confirmado"
    if "LATERAL" in str(regime):
        return "🟡 Lateral"
    if "BAJISTA" in str(regime):
        return "🔴 Bajista"
    return "⚪ Neutral"


def stop_status_label(dist_ratio, dist_atr):
    """
    Estado del stop según distancia al McGinley de salida:
    - dist_ratio: ratio (close/exit - 1)
    - dist_atr: múltiplos de ATR
    """
    if not valid_number(dist_ratio):
        return "—"

    dist_pct = float(dist_ratio) * 100.0

    if dist_pct < 2:
        pct_status = "🔴 MUY CERCA"
    elif dist_pct < 5:
        pct_status = "🟢 AJUSTADO"
    elif dist_pct < 10:
        pct_status = "🟡 HOLGADO"
    else:
        pct_status = "🟠 LEJANO"

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
    has_buy,
    close_now,
    mcg_regime_now,
    mcg_exit_now,
    recent_crosses,
    pvi_gap,
    rvol_now,
    bullish_high_volume,
    volume_quality
):
    if not has_buy:
        return None, "—", ""

    score = 40
    notes = ["PVI compra reciente"]

    if valid_number(close_now, mcg_regime_now) and close_now > mcg_regime_now:
        score += 20
        notes.append("precio sobre McGinley régimen")
    else:
        notes.append("contra tendencia / bajo régimen")

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
        notes.append(f"RVOL alto ≥{CONFIG['RVOL_HIGH']}x alcista")
    elif valid_number(rvol_now) and rvol_now >= 1.0:
        score += 8
        notes.append("volumen normal/positivo")
    else:
        notes.append("sin confirmación de volumen")

    if valid_number(pvi_gap) and pvi_gap >= CONFIG["PVI_GAP_MIN_STRONG"]:
        score += 10
        notes.append("gap PVI no micro")
    else:
        notes.append("microcruce PVI / gap débil")

    if not volume_quality:
        score -= 10
        notes.append("⚠️ volumen poco fiable")

    score = float(np.clip(score, 0, 100))
    label = entry_quality_label_from_score(score, volume_quality)
    return score, label, " · ".join(notes)


def calculate_exit_pressure(
    pvi_status,
    close_now,
    mcg_exit_now,
    mcg_regime_now,
    regime,
    recent_crosses,
    bearish_high_volume,
    pvi_gap
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
        notes.append("precio bajo McGinley régimen")

    if "BAJISTA" in str(regime):
        score += 10
        notes.append("régimen bajista")
    elif "LATERAL" in str(regime):
        score += 8
        notes.append("régimen lateral")

    if recent_crosses > 2:
        score += 5
        notes.append("cruces recientes/lateralidad")

    if bearish_high_volume:
        score += 15
        notes.append(f"RVOL bajista ≥{CONFIG['RVOL_HIGH']}x")

    if valid_number(pvi_gap) and pvi_gap <= -CONFIG["PVI_GAP_MIN_STRONG"]:
        score += 10
        notes.append("gap PVI negativo claro")

    score = float(np.clip(score, 0, 100))
    label = exit_pressure_label(score)

    if not notes:
        notes.append("sin presión técnica relevante")

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
            "ticker": ticker,
            "name": name,
            "bucket": bucket,
            "currency": currency,
            "sector": sector,
            "technical_only": bool(row["technical_only"]),
            "has_signal": False,
            "error": "Sin datos suficientes"
        }

    vol_q = volume_quality_check(df, CONFIG["VOLUME_MIN_NONZERO_PCT"])

    df = calculate_pvi(df, CONFIG["PVI_MA"])
    df["McG_Regime"] = calculate_mcginley(df["Close"], CONFIG["MCG_REGIME_N"])
    df["McG_Exit"] = calculate_mcginley(df["Close"], CONFIG["MCG_EXIT_N"])
    df["ATR"] = calculate_atr(df, CONFIG["ATR_PERIOD"])
    df["CHOP"] = calculate_chop(df, CONFIG["CHOP_PERIOD"])
    df["RVOL"] = calculate_rvol(df, CONFIG["RVOL_PERIOD"])

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

    pvi_prev = safe_float(df["PVI"].iloc[-2])
    pvi_sig_prev = safe_float(df["PVI_Signal"].iloc[-2])

    atr_now = safe_float(c["ATR"])
    chop_now = safe_float(c["CHOP"])
    rvol_now = safe_float(c["RVOL"])

    close_prev = safe_float(df["Close"].iloc[-2])

    bullish_high_volume = (
        valid_number(rvol_now, close_now, close_prev)
        and rvol_now >= CONFIG["RVOL_HIGH"]
        and close_now > close_prev
    )

    bearish_high_volume = (
        valid_number(rvol_now, close_now, close_prev)
        and rvol_now >= CONFIG["RVOL_HIGH"]
        and close_now < close_prev
    )

    raw_has_buy = buy_ago is not None
    raw_has_pvi_sell = pvi_sell_ago is not None
    raw_has_mcg_sell = mcg_sell_ago is not None

    if CONFIG.get("REQUIRE_CURRENT_SIGNAL_STATE", True):
        has_buy = (
            raw_has_buy
            and valid_number(pvi_now, pvi_sig_now)
            and pvi_now > pvi_sig_now
        )

        has_pvi_sell = (
            raw_has_pvi_sell
            and valid_number(pvi_now, pvi_sig_now)
            and pvi_now < pvi_sig_now
        )

        has_mcg_sell = (
            raw_has_mcg_sell
            and valid_number(close_now, mcg_exit_now)
            and close_now < mcg_exit_now
        )
    else:
        has_buy = raw_has_buy
        has_pvi_sell = raw_has_pvi_sell
        has_mcg_sell = raw_has_mcg_sell

    # ── Filtro de volumen estilo A/B según modo ───────────────
    vol_confirms = bool(valid_number(rvol_now) and rvol_now >= CONFIG["RVOL_HIGH"])
    raw_buy_blocked_by_vol = False
    mode = CONFIG.get("VOL_FILTER_MODE", "score")

    if mode == "hard":
        if vol_q["volume_quality"] and not vol_confirms and has_buy:
            has_buy = False
            raw_buy_blocked_by_vol = True
    elif mode == "warn":
        raw_buy_blocked_by_vol = bool(vol_q["volume_quality"] and not vol_confirms)
    # mode == "score" → no bloquea, solo afecta a entry_quality_score

    pvi_cross_current = (
        valid_number(pvi_now, pvi_sig_now, pvi_prev, pvi_sig_prev)
        and pvi_now > pvi_sig_now
        and pvi_prev <= pvi_sig_prev
    )

    # Régimen base
    if valid_number(close_now, mcg_regime_now) and close_now > mcg_regime_now:
        base_regime = "🟢 ALCISTA"
    else:
        base_regime = "🔴 BAJISTA"

    # Lateralidad enriquecida con CHOP
    if recent_crosses > 2 or (valid_number(chop_now) and chop_now >= 61.8):
        regime = "🟡 LATERAL"
    else:
        regime = base_regime

    events = []

    if has_buy:
        events.append(("🟢 COMPRA", buy_ago))

    if has_pvi_sell:
        label = "🟠 VENTA 50% PVI"
        if bearish_high_volume:
            label = "🔴 VENTA PVI FUERTE"
        events.append((label, pvi_sell_ago))

    if has_mcg_sell:
        label = "🟠 VENTA 50% McGINLEY"
        if bearish_high_volume:
            label = "🔴 ROTURA McGINLEY CON VOLUMEN"
        events.append((label, mcg_sell_ago))

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
        if valid_number(close_now, mcg_exit_now)
        else False
    )

    entry_score, entry_label, entry_notes = calculate_entry_quality(
        has_buy=has_buy,
        close_now=close_now,
        mcg_regime_now=mcg_regime_now,
        mcg_exit_now=mcg_exit_now,
        recent_crosses=recent_crosses,
        pvi_gap=pvi_gap,
        rvol_now=rvol_now,
        bullish_high_volume=bullish_high_volume,
        volume_quality=vol_q["volume_quality"]
    )

    exit_score, exit_label, exit_notes = calculate_exit_pressure(
        pvi_status=pvi_status,
        close_now=close_now,
        mcg_exit_now=mcg_exit_now,
        mcg_regime_now=mcg_regime_now,
        regime=regime,
        recent_crosses=recent_crosses,
        bearish_high_volume=bearish_high_volume,
        pvi_gap=pvi_gap
    )

    technical_state = technical_state_label(regime, pvi_status, price_below_mcg_exit)
    stop_status = stop_status_label(dist_to_mcg_exit, dist_to_mcg_exit_atr)

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

        "signal_freshness": freshness,

        "buy_ago": buy_ago,
        "pvi_sell_ago": pvi_sell_ago,
        "mcg_sell_ago": mcg_sell_ago,
        "bars_min": bars_min,

        "regime": regime,
        "technical_state": technical_state,
        "recent_crosses": recent_crosses,

        "close": close_now,
        "mcg_regime": mcg_regime_now,
        "mcg_exit": mcg_exit_now,
        "dist_to_mcg_exit": dist_to_mcg_exit,
        "dist_to_mcg_exit_atr": dist_to_mcg_exit_atr,
        "stop_status": stop_status,

        "atr": atr_now,
        "chop": chop_now,

        "pvi": pvi_now,
        "pvi_signal": pvi_sig_now,
        "pvi_status": pvi_status,
        "pvi_gap": pvi_gap,
        "pvi_prev": pvi_prev,
        "pvi_signal_prev": pvi_sig_prev,
        "pvi_cross_current": bool(pvi_cross_current),
        "pvi_signal_type": CONFIG.get("PVI_SIGNAL_TYPE", "EMA"),

        "rvol": rvol_now,
        "rvol_high": bool(vol_confirms),
        "bullish_high_volume": bool(bullish_high_volume),
        "bearish_high_volume": bool(bearish_high_volume),

        "volume_quality": vol_q["volume_quality"],
        "volume_nonzero_pct": vol_q["volume_nonzero_pct"],
        "volume_zero_days": vol_q["volume_zero_days"],
        "volume_warning": vol_q["volume_warning"],

        "vol_confirms": bool(vol_confirms),
        "raw_buy_blocked_by_vol": bool(raw_buy_blocked_by_vol),

        "entry_quality_score": entry_score,
        "entry_quality_label": entry_label,
        "entry_quality_notes": entry_notes,

        "exit_pressure_score": exit_score,
        "exit_pressure_label": exit_label,
        "exit_pressure_notes": exit_notes,

        "price_below_mcg_exit": price_below_mcg_exit,

        "last_date": str(df.index[-1].date()),
        "error": ""
    }


# ============================================================
# FUNDAMENTALES — CALIDAD / VALORACIÓN / CONFIANZA
# (base Mejora B, sin tocar apenas)
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
    return "Débil"


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
        return "🟢 Calidad con descuento"
    if q >= 70 and v >= 45:
        return "✅ Calidad razonable"
    if q >= 70 and v < 45:
        return "💎 Calidad cara"
    if 50 <= q < 70 and v >= 65:
        return "🟡 Value especulativo"
    if q < 50 and v >= 65:
        return "🪤 Value trap"
    if q < 50 and v < 45:
        return "🔴 Débil y cara"
    return "⚖️ Equilibrado"


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
        return "📈 Mejorando"
    if avg < -0.3:
        return "📉 Deteriorando"
    return "➡️ Estable"


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

    else:  # standard
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

    bank_tickers = {
        "JPM", "BAC", "SAN.MC", "BBVA.MC", "BNP.PA", "SAN.PA"
    }

    insurance_tickers = {
        "ALV.DE", "MUV2.DE", "MAP.MC"
    }

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
            ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"],
            0
        )

        if not valid_number(ca):
            ca = 0

        if not valid_number(cl):
            cl = 0

        if not valid_number(re):
            re = 0

        ebit = ebit * multiplier if valid_number(ebit) else np.nan
        revenue = revenue * multiplier if valid_number(revenue) else np.nan

        if not valid_number(total_liab) or total_liab == 0:
            return np.nan

        wc = ca - cl

        return (
            1.2 * safe_div(wc, ta)
            + 1.4 * safe_div(re, ta)
            + 3.3 * safe_div(ebit, ta)
            + 0.6 * safe_div(market_cap, total_liab)
            + 1.0 * safe_div(revenue, ta)
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
        if valid_number(dsri):
            comps.append("dsri")

        gm0 = safe_div(gp0, rev0)
        gm1 = safe_div(gp1, rev1)
        gmi = safe_div(gm1, gm0)
        if valid_number(gmi):
            comps.append("gmi")

        aqi = safe_div(
            1 - safe_div(ca0 + ppe0, ta0),
            1 - safe_div(ca1 + ppe1, ta1)
        )
        if valid_number(aqi):
            comps.append("aqi")

        sgi = safe_div(rev0, rev1)
        if valid_number(sgi):
            comps.append("sgi")

        depi = safe_div(
            safe_div(dep1, dep1 + ppe1),
            safe_div(dep0, dep0 + ppe0)
        )
        if valid_number(depi):
            comps.append("depi")

        sgai = safe_div(safe_div(sga0, rev0), safe_div(sga1, rev1))
        if valid_number(sgai):
            comps.append("sgai")

        lvgi = safe_div(safe_div(debt0, ta0), safe_div(debt1, ta1))
        if valid_number(lvgi):
            comps.append("lvgi")

        tata = safe_div(ni0 - cfo0, ta0)
        if valid_number(tata):
            comps.append("tata")

        if len(comps) < 5:
            return np.nan

        dsri = dsri if valid_number(dsri) else 1
        gmi = gmi if valid_number(gmi) else 1
        aqi = aqi if valid_number(aqi) else 1
        sgi = sgi if valid_number(sgi) else 1
        depi = depi if valid_number(depi) else 1
        sgai = sgai if valid_number(sgai) else 1
        lvgi = lvgi if valid_number(lvgi) else 1
        tata = tata if valid_number(tata) else 0

        m = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
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

        total_liabilities = get_val(
            bs_ref,
            ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"],
            0
        )

        equity = get_val(
            bs_ref,
            ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
            0
        )

        if pd.isna(equity) and valid_number(total_assets, total_liabilities):
            equity = total_assets - total_liabilities

        total_debt = get_val(bs_ref, ["Total Debt"], 0)

        if pd.isna(total_debt):
            lt_debt = get_val(bs_ref, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
            st_debt = get_val(bs_ref, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], 0)
            total_debt = np.nansum([lt_debt, st_debt])

        cash = get_val(
            bs_ref,
            ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"],
            0
        )

        if pd.isna(cash):
            cash = 0

        shares0 = get_val(bs_ref, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 0)
        shares1 = get_val(bs_ref, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], prev_pos)

        shares_growth = (
            safe_div(shares0, shares1) - 1
            if valid_number(shares0, shares1) and shares1 != 0
            else np.nan
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
            altman_z = calculate_altman_z_v2(
                bs_ref,
                is_q if not is_q.empty else is_a,
                info,
                market_cap,
                multiplier
            )

        beneish_m = calculate_beneish_m_score(p_f, p_b, p_cf, prev_pos=prev_pos)
        consistency = consistency_score(is_q if not is_q.empty else is_a)

        raw_metrics = {
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "fundamental_model": model,

            "revenue_ttm": revenue_ttm,
            "op_income_ttm": op_income_ttm,
            "net_income_ttm": net_income_ttm,
            "cfo_ttm": cfo_ttm,
            "fcf_ttm": fcf_ttm,

            "total_assets": total_assets,
            "equity": equity,
            "total_debt": total_debt,

            "roic": roic,
            "roe": roe,
            "roa": roa,
            "equity_assets": equity_assets,
            "debt_equity": debt_equity,

            "gross_margin": gross_margin,
            "op_margin": op_margin,
            "fcf_margin": fcf_margin,
            "fcf_yield": fcf_yield,
            "cash_quality": cash_quality,
            "current_ratio": current_ratio,
            "interest_coverage": interest_coverage,
            "net_debt_ebitda": net_debt_ebitda,

            "pb": pb,
            "pe": pe,
            "ps": ps,
            "ev_ebitda": ev_ebitda,
            "upside": upside,

            "revenue_growth": revenue_growth,
            "op_income_growth": op_income_growth,
            "net_income_growth": net_income_growth,
            "shares_growth": shares_growth,

            "piotroski": piotroski,
            "altman_z": altman_z,
            "beneish_m": beneish_m,
            "consistency_score": consistency,
        }

        conf_score, conf_grade, available_fields, expected_fields = compute_confidence(
            model,
            latest_date,
            is_q,
            is_a,
            raw_metrics
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
            red_flags.append("Dilución alta")

        if conf_grade == "C":
            red_flags.append("Cobertura limitada")

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "currency": currency,

            "fundamental_model": model,
            "period_label": period_label,
            "latest_report_date": str(pd.to_datetime(latest_date).date()) if latest_date is not None else None,
            "reporting_frequency": reporting_frequency,
            "reporting_multiplier": multiplier,

            "price": price,
            "market_cap": market_cap,
            "upside": upside,

            "revenue_ttm": revenue_ttm,
            "gross_profit_ttm": gross_profit_ttm,
            "op_income_ttm": op_income_ttm,
            "net_income_ttm": net_income_ttm,
            "cfo_ttm": cfo_ttm,
            "capex_ttm": capex_ttm,
            "fcf_ttm": fcf_ttm,

            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "equity": equity,
            "total_debt": total_debt,
            "cash": cash,

            "roic": roic,
            "roe": roe,
            "roa": roa,
            "equity_assets": equity_assets,
            "debt_equity": debt_equity,

            "gross_margin": gross_margin,
            "op_margin": op_margin,
            "fcf_margin": fcf_margin,
            "fcf_yield": fcf_yield,
            "cash_quality": cash_quality,
            "current_ratio": current_ratio,
            "interest_coverage": interest_coverage,
            "net_debt_ebitda": net_debt_ebitda,

            "pb": pb,
            "pe": pe,
            "ps": ps,
            "ev_ebitda": ev_ebitda,

            "revenue_growth": revenue_growth,
            "op_income_growth": op_income_growth,
            "net_income_growth": net_income_growth,
            "shares_growth": shares_growth,

            "piotroski": piotroski,
            "altman_z": altman_z,
            "beneish_m": beneish_m,
            "consistency_score": consistency,

            "has_fundamentals": True,

            "quality_score": q_score,
            "quality_label": q_lbl,

            "valuation_score": v_score,
            "valuation_label": v_lbl,

            "fundamental_profile": profile,
            "fundamental_trend": ftrend,

            "confidence_score": conf_score,
            "confidence_grade": conf_grade,
            "available_fields": available_fields,
            "expected_fields": expected_fields,

            "score_quality": sub_quality,
            "score_cash": sub_cash,
            "score_solvency": sub_solvency,
            "score_growth": sub_growth,
            "score_valuation": sub_valuation,
            "score_risk": sub_risk,

            "valuation_style": valuation_style(sector, industry, model),

            "red_flags": ", ".join(red_flags) if red_flags else "",
            "model_note": model_note,
            "not_applicable": not_applicable,
        }

    except Exception:
        return None


def add_fundamentals(assets_df, universe_df):
    fund_rows = []

    candidates = universe_df[~universe_df["technical_only"]]["ticker"].tolist()

    print(f"🧬 LCrack Fundamental: {len(candidates)} activos elegibles...")

    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = {executor.submit(get_fundamental_raw, t): t for t in candidates}

        for fut in as_completed(futures):
            ticker = futures[fut]

            try:
                raw = fut.result()

                if raw:
                    fund_rows.append(raw)
                    print(
                        f"   ✅ {ticker} · "
                        f"Calidad {raw.get('quality_score', 0):.0f} ({raw.get('quality_label','—')}) · "
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

    return merged


# ============================================================
# MACRO + FX (base Mejora B)
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
        "warning": bool(score < 0),
        "warning_text": "⚠️ Macro cautelosa" if score < 0 else "",
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
# HTML / FRONTEND (base Mejora B + stop_status / bloqueo vol)
# ============================================================

INDEX_HTML = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LCrack Sovereign</title>

<style>
:root {
  --bg: #020617;
  --panel: #0f172a;
  --border: rgba(148,163,184,.22);
  --text: #e5e7eb;
  --muted: #94a3b8;
  --green: #86efac;
  --red: #fca5a5;
  --yellow: #fde68a;
  --blue: #93c5fd;
  --purple: #d8b4fe;
  --orange: #fdba74;
}

* { box-sizing: border-box; }

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
  max-width: 1650px;
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
  line-height: 1.35;
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

.section { display: none; }
.section.active { display: block; }

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

tr:hover { background: rgba(30,41,59,.5); }

.badge {
  display: inline-block;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  border: 1px solid var(--border);
  white-space: nowrap;
  margin: 1px;
}

.buy { color: var(--green); background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.35); }
.sell { color: var(--red); background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.35); }
.partial { color: var(--yellow); background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.35); }
.neutral { color: #cbd5e1; background: rgba(148,163,184,.12); border-color: rgba(148,163,184,.28); }
.mixed { color: var(--purple); background: rgba(168,85,247,.15); border-color: rgba(168,85,247,.35); }
.qbadge { color: var(--blue); background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.35); }
.vbadge { color: var(--orange); background: rgba(251,146,60,.15); border-color: rgba(251,146,60,.35); }
.warnbadge { color: var(--yellow); background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.35); }

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
  align-items: center;
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

.warning { color: var(--yellow); }

.score-bar-wrap {
  width: 100%;
  height: 6px;
  background: rgba(148,163,184,.2);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 5px;
}

.score-bar {
  height: 6px;
  border-radius: 999px;
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
  <h1>🛰️ LCrack Sovereign</h1>
  <div class="subtitle" id="subtitle">Cargando datos...</div>

  <div class="grid" id="summaryCards"></div>

  <div class="tabs">
    <button class="tab active" onclick="showTab('global', event)">🌍 Panel global</button>
    <button class="tab" onclick="showTab('signals', event)">🎯 Señales</button>
    <button class="tab" onclick="showTab('universe', event)">📡 Universo</button>
    <button class="tab" onclick="showTab('portfolio', event)">💼 Mi cartera</button>
    <button class="tab" onclick="showTab('rules', event)">📘 Reglas</button>
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

        <select id="fundConfidenceFilter" onchange="renderUniverse()">
          <option value="">Confianza: todas</option>
          <option value="A">Confianza A</option>
          <option value="B">Confianza B+</option>
          <option value="C">Confianza C+</option>
        </select>

        <select id="fundQualityFilter" onchange="renderUniverse()">
          <option value="">Calidad: todas</option>
          <option value="85">Excelente 85+</option>
          <option value="75">Muy buena 75+</option>
          <option value="65">Buena 65+</option>
          <option value="50">Media 50+</option>
        </select>

        <select id="fundValuationFilter" onchange="renderUniverse()">
          <option value="">Precio/Fund.: todos</option>
          <option value="80">Muy barata 80+</option>
          <option value="65">Barata 65+</option>
          <option value="45">Razonable 45+</option>
        </select>

        <select id="entryQualityFilter" onchange="renderUniverse()">
          <option value="">Entrada: todas</option>
          <option value="85">Entrada A</option>
          <option value="70">Entrada B+</option>
          <option value="55">Entrada C+</option>
        </select>

        <select id="exitPressureFilter" onchange="renderUniverse()">
          <option value="">Presión salida: todas</option>
          <option value="25">Vigilar+</option>
          <option value="50">Reducir+</option>
          <option value="75">Salida fuerte</option>
        </select>

        <select id="fundQFilter" onchange="renderUniverse()">
          <option value="">Q calidad: todos</option>
          <option value="Q1">Q1 ⭐ TOP</option>
          <option value="Q2">Q2 ✅</option>
          <option value="Q3">Q3 ⚖️</option>
          <option value="Q4">Q4 ⚠️</option>
        </select>

        <label class="small">
          <input id="onlyFundamentals" type="checkbox" onchange="renderUniverse()">
          Solo con fundamentales
        </label>
      </div>
      <div id="universeTable"></div>
    </div>
  </section>

  <section id="portfolio" class="section">
    <div class="card">
      <h2>💼 Mi cartera privada</h2>
      <p class="small">
        Tus posiciones se guardan solo en este navegador mediante localStorage.
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

      <h3>Técnico</h3>
      <p><b>Compra:</b> PVI cruza su EMA120 de abajo hacia arriba.</p>
      <p><b>Venta 50% PVI:</b> PVI cruza su EMA120 de arriba hacia abajo.</p>
      <p><b>Venta 50% McGinley:</b> precio cruza McGinley de salida hacia abajo.</p>
      <p><b>Venta 100%:</b> PVI negativo + precio bajo McGinley salida.</p>
      <p><b>RVOL alto:</b> volumen relativo ≥ 1.5x. En entradas suma calidad. En salidas agrava la presión si es bajista.</p>
      <p><b>Presión de salida:</b> estado persistente, aunque la señal reciente ya haya caducado.</p>
      <p><b>Calidad de entrada:</b> combina PVI, tendencia McGinley, lateralidad, volumen y gap PVI.</p>
      <p><b>Stop:</b> distancia al McGinley de salida en % y en ATR (MUY CERCA / AJUSTADO / HOLGADO / LEJANO).</p>

      <h3>Fundamentales</h3>
      <p><b>Calidad:</b> rentabilidad, caja, solvencia, crecimiento y riesgo contable.</p>
      <p><b>Precio/Fundamentales:</b> valoración separada, con baremos sectoriales aproximados.</p>
      <p><b>Confianza:</b> cobertura, frescura e historial de datos.</p>
      <p><b>Tendencia fundamental:</b> si ingresos, operativo y beneficio mejoran o se deterioran.</p>

      <p class="warning"><b>Aviso:</b> herramienta mecánica basada en reglas. No es asesoramiento financiero.</p>
    </div>
  </section>

  <div class="footer">
    LCrack Sovereign · Datos vía yfinance/FRED/Fear & Greed.
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
  if (text.includes("COMPRA") || text.includes("ALCISTA") || text.includes("MANTENER") || text.includes("Baja")) return "buy";
  if (text.includes("VENTA 100") || text.includes("BAJISTA") || text.includes("Salida fuerte") || text.includes("VENDER TODO")) return "sell";
  if (text.includes("VENTA") || text.includes("LATERAL") || text.includes("REDUCIR") || text.includes("Reducir") || text.includes("STOP") || text.includes("Vigilar")) return "partial";
  if (text.includes("MIXTA")) return "mixed";
  return "neutral";
}

function badge(text, cls="") {
  return `<span class="badge ${cls || clsFor(text)}">${text || "—"}</span>`;
}

function scoreBar(val, color="#3b82f6") {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return "";
  const pct = Math.max(0, Math.min(100, Number(val)));
  return `<div class="score-bar-wrap"><div class="score-bar" style="width:${pct}%;background:${color}"></div></div>`;
}

function qualityColor(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return "#64748b";
  if (score >= 85) return "#86efac";
  if (score >= 75) return "#4ade80";
  if (score >= 65) return "#a3e635";
  if (score >= 50) return "#fde68a";
  return "#fca5a5";
}

function valuationColor(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return "#64748b";
  if (score >= 80) return "#86efac";
  if (score >= 65) return "#4ade80";
  if (score >= 45) return "#fde68a";
  if (score >= 25) return "#fb923c";
  return "#fca5a5";
}

function showTab(id, ev) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  if (ev && ev.target) ev.target.classList.add("active");
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
    ["🌍 Macro", summary.macro?.label || "—", `${summary.macro?.warning_text || "Score: " + fmtNum(summary.macro?.score, 1)}`],
    ["💶 EUR/USD", summary.fx?.label || "—", summary.fx?.diag || "—"],
    ["🎯 Señales", String(summary.total_signals || 0), `Compras: ${summary.buy_signals || 0} · Ventas: ${summary.sell_signals || 0}`],
    ["⚙️ Parámetros", `${summary.config?.LOOKBACK_SIGNAL || 5} velas`, `PVI ${summary.config?.PVI_MA || 120} · RVOL ${summary.config?.RVOL_HIGH || 1.5}x`],
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

function fundBlock(a) {
  const hasFund = a.has_fundamentals === true;

  if (!hasFund) {
    return `<div class="small">Sin fundamentales / técnico-only / datos insuficientes</div>`;
  }

  return `
    <div>
      ${badge("Calidad " + fmtNum(a.quality_score,0) + " · " + (a.quality_label || "—"), "qbadge")}
      ${badge("Precio/Fund. " + fmtNum(a.valuation_score,0) + " · " + (a.valuation_label || "—"), "vbadge")}
      ${badge("Conf " + (a.confidence_grade || "—"), "neutral")}
      ${badge(a.quality_q || "—", "qbadge")}
    </div>

    ${scoreBar(a.quality_score, qualityColor(a.quality_score))}
    ${scoreBar(a.valuation_score, valuationColor(a.valuation_score))}

    <div class="small">${a.fundamental_profile || "—"} · ${a.fundamental_trend || "—"}</div>

    <div class="small">
      Calidad: ${fmtNum(a.score_quality,0)} op ·
      ${fmtNum(a.score_cash,0)} caja ·
      ${fmtNum(a.score_solvency,0)} solv ·
      ${fmtNum(a.score_growth,0)} crec ·
      ${fmtNum(a.score_risk,0)} riesgo
    </div>

    <div class="small">
      ROIC ${fmtPct(a.roic,1)} · ROE ${fmtPct(a.roe,1)} ·
      FCF Yield ${fmtPct(a.fcf_yield,1)} ·
      Altman ${fmtNum(a.altman_z,2)} ·
      Piotroski ${fmtNum(a.piotroski,1)}
    </div>

    <div class="small">
      P/E ${fmtNum(a.pe,1)} · P/B ${fmtNum(a.pb,2)} ·
      EV/EBITDA ${fmtNum(a.ev_ebitda,1)} ·
      ND/EBITDA ${fmtNum(a.net_debt_ebitda,2)}
    </div>

    <div class="small">
      ${a.fundamental_model || "—"} · ${a.period_label || "N/A"} · ${a.valuation_style || "—"}
    </div>

    ${a.red_flags ? `<div class="small warning">⚠️ ${a.red_flags}</div>` : ""}
  `;
}

function techBlock(a) {
  const macroWarn = summary.macro?.warning && String(a.main_signal || "").includes("COMPRA")
    ? badge("⚠️ Macro cautelosa", "warnbadge")
    : "";

  const volBlocked = a.raw_buy_blocked_by_vol
    ? `<div class="small warning">⚠️ Señal de compra bloqueada por volumen insuficiente</div>`
    : "";

  return `
    <div>
      ${badge(a.technical_state || "—")}
      ${badge(a.signal_freshness || "—", "neutral")}
      ${macroWarn}
    </div>

    <div class="small">
      Entrada: <b>${a.entry_quality_label || "—"}</b>
      ${a.entry_quality_score !== null && a.entry_quality_score !== undefined ? " · " + fmtNum(a.entry_quality_score,0) + "/100" : ""}
    </div>

    <div class="small">
      ${a.entry_quality_notes || ""}
    </div>

    <div class="small">
      Salida: <b>${a.exit_pressure_label || "—"}</b> · ${fmtNum(a.exit_pressure_score,0)}/100
    </div>

    <div class="small">
      ${a.exit_pressure_notes || ""}
    </div>

    <div class="small">
      PVI ${a.pvi_status || "—"} · Gap ${fmtPct(a.pvi_gap,2)} ·
      RVOL ${fmtNum(a.rvol,2)}x
      ${a.bullish_high_volume ? " · 🟢 Vol. alcista alto" : ""}
      ${a.bearish_high_volume ? " · 🔴 Vol. bajista alto" : ""}
    </div>

    <div class="small">
      Dist McG ${fmtPct(a.dist_to_mcg_exit,1)} ·
      ATR ${fmtNum(a.dist_to_mcg_exit_atr,2)}x ·
      CHOP ${fmtNum(a.chop,1)}
    </div>

    <div class="small">
      Stop: ${a.stop_status || "—"}
    </div>

    ${a.volume_quality === false ? `<div class="small warning">⚠️ ${a.volume_warning || "Volumen dudoso"}</div>` : ""}
    ${volBlocked}
  `;
}

function signalRow(a) {
  return `
    <tr>
      <td>
        <div class="ticker">${a.ticker}</div>
        <div class="name">${a.name || "—"}</div>
      </td>
      <td>${badge(a.main_signal)}<div class="small">${a.events_text || ""}</div></td>
      <td>${badge(a.regime)}<div class="small">Cruces: ${a.recent_crosses ?? "—"}</div></td>
      <td>${techBlock(a)}</td>
      <td>${fundBlock(a)}</td>
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
          <th>Activo</th><th>Señal</th><th>Régimen</th><th>Técnico avanzado</th><th>Fundamental</th><th>Universo</th>
        </tr>
      </thead>
      <tbody>${data.map(signalRow).join("")}</tbody>
    </table>
  `;
}

function confidenceValue(c) {
  if (c === "A") return 3;
  if (c === "B") return 2;
  if (c === "C") return 1;
  return 0;
}

function renderUniverse() {
  const q = (document.getElementById("universeSearch")?.value || "").toUpperCase();
  const regime = document.getElementById("universeRegime")?.value || "";
  const confFilter = document.getElementById("fundConfidenceFilter")?.value || "";
  const qualMin = document.getElementById("fundQualityFilter")?.value;
  const valMin = document.getElementById("fundValuationFilter")?.value;
  const entryMin = document.getElementById("entryQualityFilter")?.value;
  const exitMin = document.getElementById("exitPressureFilter")?.value;
  const qFilter = document.getElementById("fundQFilter")?.value || "";
  const onlyFund = document.getElementById("onlyFundamentals")?.checked || false;

  let data = allAssets.slice();

  if (q) {
    data = data.filter(a =>
      String(a.ticker).toUpperCase().includes(q) ||
      String(a.name).toUpperCase().includes(q) ||
      String(a.sector || "").toUpperCase().includes(q) ||
      String(a.industry || "").toUpperCase().includes(q)
    );
  }

  if (regime) data = data.filter(a => String(a.regime || "").includes(regime));
  if (onlyFund) data = data.filter(a => a.has_fundamentals === true);
  if (confFilter) data = data.filter(a => confidenceValue(a.confidence_grade) >= confidenceValue(confFilter));
  if (qualMin) data = data.filter(a => Number(a.quality_score) >= Number(qualMin));
  if (valMin) data = data.filter(a => Number(a.valuation_score) >= Number(valMin));
  if (entryMin) data = data.filter(a => Number(a.entry_quality_score) >= Number(entryMin));
  if (exitMin) data = data.filter(a => Number(a.exit_pressure_score) >= Number(exitMin));
  if (qFilter) data = data.filter(a => String(a.quality_q || "") === qFilter);

  data = data.sort((a,b) => {
    const fa = a.quality_score === null || a.quality_score === undefined ? -1 : Number(a.quality_score);
    const fb = b.quality_score === null || b.quality_score === undefined ? -1 : Number(b.quality_score);
    if (fb !== fa) return fb - fa;
    return String(a.ticker).localeCompare(String(b.ticker));
  });

  const rows = data.map(a => `
    <tr>
      <td>
        <div class="ticker">${a.ticker}</div>
        <div class="name">${a.name || "—"}</div>
      </td>
      <td>${badge(a.main_signal || "—")}</td>
      <td>${badge(a.regime || "—")}<div class="small">${a.technical_state || "—"}</div></td>
      <td>${fmtNum(a.close,2)}</td>
      <td>${techBlock(a)}</td>
      <td>${fundBlock(a)}</td>
      <td>${a.bucket || "—"}</td>
    </tr>
  `).join("");

  document.getElementById("universeTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Activo</th>
          <th>Señal</th>
          <th>Régimen</th>
          <th>Precio</th>
          <th>Técnico avanzado</th>
          <th>Fundamental</th>
          <th>Universo</th>
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

function systemAction(asset) {
  if (!asset) {
    return ["⚪ SIN DATOS", "No hay datos diarios para este activo."];
  }

  const pressure = Number(asset.exit_pressure_score || 0);
  const signal = asset.main_signal || "";
  const state = asset.technical_state || "";

  if (signal.includes("VENTA 100") || pressure >= 75) {
    return ["🔴 VENDER TODO / REVISAR", "Presión de salida fuerte."];
  }

  if (pressure >= 50) {
    return ["🟠 REDUCIR", "Presión de salida elevada."];
  }

  if (pressure >= 25) {
    return ["🟡 VIGILAR", "Hay deterioro técnico parcial."];
  }

  if (state.includes("Alcista confirmado")) {
    return ["🟢 MANTENER", "Estado técnico favorable."];
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

    const [action, reason] = systemAction(asset);

    return `
      <tr>
        <td>
          <div class="ticker">${p.ticker}</div>
          <div class="name">${p.name || (asset ? asset.name : "")}</div>
          ${asset && asset.has_fundamentals ? `<div class="small">Cal ${fmtNum(asset.quality_score,0)} · Val ${fmtNum(asset.valuation_score,0)} · ${asset.fundamental_trend || "—"}</div>` : ""}
        </td>
        <td>${fmtNum(qty,4)}</td>
        <td>${fmtNum(buy,2)}</td>
        <td>${price ? fmtNum(price,2) : "—"}</td>
        <td>${pnlPct !== null ? fmtPct(pnlPct,2) : "—"}<div class="small">${pnl !== null ? fmtNum(pnl,2) : "—"}</div></td>
        <td>${asset ? badge(asset.main_signal || "—") : "—"}<div class="small">${asset ? (asset.events_text || "") : ""}</div></td>
        <td>${asset ? badge(asset.exit_pressure_label || "—") : "—"}<div class="small">${asset ? asset.exit_pressure_notes || "" : ""}</div></td>
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
          <th>Señal</th><th>Presión salida</th><th>Acción sistema</th><th>Fecha/Nota</th><th></th>
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
    print("🚀 LCrack Sovereign")
    print(f"   Modo filtro volumen: {CONFIG.get('VOL_FILTER_MODE')} · RVOL≥{CONFIG['RVOL_HIGH']}x")

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
                    print(
                        f"✅ Señal: {ticker} -> {res.get('main_signal')} · "
                        f"{res.get('entry_quality_label','—')} · "
                        f"{res.get('exit_pressure_label','—')}"
                    )
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
```

Si quieres, en el siguiente mensaje puedo:

- Dejarte un **diff conceptual** respecto a tu Mejora B (para ver solo lo que he tocado), o  
- Ajustar el `VOL_FILTER_MODE` a `"hard"` y simplificar mensajes para operar 100% con el filtro ganador del backtest.
