# data_utilities.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from pykrx import stock
import os
import sys
import numpy as np
import logging

from technical_indicators import UniversalRiskRewardCalculator
from chart_plotting import (
    compute_prophet_forecast,
    compute_neuralprophet_forecast,
    compute_xgboost_forecast,
)

# Adjust path to import common and config from parent directory
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from common import fetch_data_in_parallel  # noqa: E402
import config as cfg  # noqa: E402

@st.cache_data(ttl=60 * 60)
def get_ticker_name_map():
    tickers = []
    try:
        tickers.extend(stock.get_market_ticker_list(market="KOSPI"))
        tickers.extend(stock.get_market_ticker_list(market="KOSDAQ"))
    except Exception:
        tickers = []

    if not tickers:
        try:
            df_kospi = fdr.StockListing('KOSPI')
            df_kosdaq = fdr.StockListing('KOSDAQ')
            combined = pd.concat([df_kospi, df_kosdaq])
            name_to_ticker = dict(zip(combined['Name'], combined['Code']))
            ticker_to_name = dict(zip(combined['Code'], combined['Name']))
            return name_to_ticker, ticker_to_name
        except Exception:
            return {}, {}

    name_to_ticker = {}
    ticker_to_name = {}
    for t in tickers:
        try:
            name = stock.get_market_ticker_name(t)
            if name:
                name_to_ticker[name] = t
                ticker_to_name[t] = name
        except Exception:
            continue
    return name_to_ticker, ticker_to_name

@st.cache_data(ttl=60 * 60)
def get_latest_fundamental(max_lookback=10):
    for i in range(max_lookback):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = stock.get_market_fundamental(date_str, market="ALL")
        except Exception as e:
            logging.exception("fundamental fetch error: %s", date_str)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            logging.info("fundamental loaded: %s rows=%s", date_str, len(df))
            return date_str, df
        logging.warning("fundamental empty: %s", date_str)
    return None, pd.DataFrame()

@st.cache_data(ttl=60 * 60)
def load_price_data(ticker, start_date, end_date):
    try:
        return fdr.DataReader(ticker, start=start_date, end=end_date)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60)
def load_fundamental_history(ticker, start_date, end_date):
    try:
        return stock.get_market_fundamental(start_date, end_date, ticker)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60)
def load_foreign_history(ticker, start_date, end_date):
    try:
        return stock.get_exhaustion_rates_of_foreign_investment(start_date, end_date, ticker)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 30)
def get_rr_analysis(ticker, entry_price):
    calc = UniversalRiskRewardCalculator()
    return calc.analyze(ticker, entry_price)


@st.cache_data(ttl=60 * 60)
def get_ai_forecasts(df, prophet_periods=30, neural_periods=5, xgb_periods=5):
    forecast_prophet = compute_prophet_forecast(df, periods=prophet_periods)
    forecast_np = compute_neuralprophet_forecast(df, periods=neural_periods)
    forecast_xgb = compute_xgboost_forecast(df, periods=xgb_periods)
    return {
        "prophet": forecast_prophet,
        "neural": forecast_np,
        "xgboost": forecast_xgb,
    }

def calculate_etf_data():
    etf_tickers = cfg.ETF_TICKERS
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        market_df = fdr.DataReader(cfg.ETF_MARKET_INDEX, start=start_date, end=end_date)
        market_index = market_df['Close'].ffill()
        raw_data = fetch_data_in_parallel(etf_tickers, start_date, end_date)
        if raw_data.empty: return None
    except Exception:
        return None

    w1, w2, w3 = cfg.MOMENTUM_WEIGHTS
    mom_1m = raw_data.pct_change(20).iloc[-1]
    mom_3m = raw_data.pct_change(60).iloc[-1]
    mom_6m = raw_data.pct_change(120).iloc[-1]
    weighted_score = (mom_1m.fillna(0) * w1) + (mom_3m.fillna(0) * w2) + (mom_6m.fillna(0) * w3)

    ma_series = market_index.rolling(window=60).mean()
    is_bull = market_index.iloc[-1] > ma_series.iloc[-1]
    is_neutral = not is_bull and (ma_series.iloc[-1] > ma_series.iloc[-6])
    status = "상승장" if is_bull else "중립장" if is_neutral else "하락장"
    reason = "적극투자" if is_bull else "분산투자" if is_neutral else "현금방어"

    scores = weighted_score.drop(cfg.ETF_DEFENSE_ASSET, errors='ignore').sort_values(ascending=False)
    selected = [n for n, s in scores.items() if s > 0][:2]

    if is_bull: targets = [(selected[0], 0.5), (selected[1], 0.5)] if len(selected) > 1 else [(selected[0], 1.0)] if selected else [(cfg.ETF_DEFENSE_ASSET, 1.0)]
    elif is_neutral: targets = [(selected[0], 0.25), (selected[1], 0.25), (cfg.ETF_DEFENSE_ASSET, 0.5)] if len(selected) > 1 else [(selected[0], 0.5), (cfg.ETF_DEFENSE_ASSET, 0.5)] if selected else [(cfg.ETF_DEFENSE_ASSET, 1.0)]
    else: targets = [(cfg.ETF_DEFENSE_ASSET, 1.0)]

    all_ranks = []
    for i, (n, s) in enumerate(scores.items(), 1):
        code = etf_tickers.get(n, "N/A")
        price = raw_data[n].iloc[-1] if n in raw_data.columns else 0
        all_ranks.append({'rank': i, 'name': n, 'code': code, 'score': s, 'price': price})

    return {"status": status, "reason": reason, "targets": targets, "rankings": all_ranks, "raw_data_last": raw_data.iloc[-1]}

def calculate_stock_data():
    try:
        df_kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSPI)
        df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSDAQ)
        tickers = {row['Name']: row['Code'] for _, row in pd.concat([df_kospi, df_kosdaq]).iterrows()}
        tickers[cfg.STOCK_DEFENSE_ASSET] = cfg.ETF_TICKERS.get(cfg.STOCK_DEFENSE_ASSET, '261240')
    except Exception:
        return None

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        market_df = fdr.DataReader(cfg.STOCK_MARKET_INDEX, start=start_date, end=end_date)
        raw_data = fetch_data_in_parallel(tickers, start_date, end_date)
        valid_cols = [c for c in raw_data.columns if raw_data[c].count() >= 120]
        raw_data = raw_data[valid_cols]
    except Exception:
        return None

    daily_rets = raw_data.pct_change()
    vol = daily_rets.rolling(60).std().iloc[-1]
    score = ((raw_data.pct_change(60).iloc[-1]/(vol+1e-6)).fillna(0)*0.5) + ((raw_data.pct_change(120).iloc[-1]/(vol+1e-6)).fillna(0)*0.5)

    market_ma = market_df['Close'].ffill().rolling(60).mean()
    is_bull = market_df['Close'].iloc[-1] > market_ma.iloc[-1]
    is_neutral = not is_bull and (market_ma.iloc[-1] > market_ma.iloc[-6])
    status = "상승장" if is_bull else "중립장" if is_neutral else "하락장"
    reason = "적극투자" if is_bull else "분산투자" if is_neutral else "현금방어"

    top_assets = score.drop(cfg.STOCK_DEFENSE_ASSET, errors='ignore').sort_values(ascending=False)
    selected = [n for n, s in top_assets.items() if s > 0][:cfg.STOCK_TOP_N]

    if is_bull: targets = [(s, 1.0/len(selected)) for s in selected] if selected else [(cfg.STOCK_DEFENSE_ASSET, 1.0)]
    elif is_neutral: targets = ([(s, 0.5/len(selected)) for s in selected] + [(cfg.STOCK_DEFENSE_ASSET, 0.5)]) if selected else [(cfg.STOCK_DEFENSE_ASSET, 1.0)]
    else: targets = [(cfg.STOCK_DEFENSE_ASSET, 1.0)]

    all_ranks = []
    for i, (n, s) in enumerate(top_assets.items(), 1):
        code = tickers.get(n, n)
        price = raw_data[n].iloc[-1] if n in raw_data.columns else 0
        all_ranks.append({'rank': i, 'name': n, 'code': code, 'score': s, 'price': price})

    return {"status": status, "reason": reason, "targets": targets, "rankings": all_ranks, "raw_data_last": raw_data.iloc[-1], "tickers_map": tickers}

def calculate_us_data():
    try:
        df_sp = fdr.StockListing('S&P500')
        sp500_tickers = set(df_sp['Symbol'].tolist())
        df_nasdaq = fdr.StockListing('NASDAQ')
        nasdaq100_tickers = set(df_nasdaq.head(100)['Symbol'].tolist())
        combined_tickers = sp500_tickers.union(nasdaq100_tickers)
        tickers = {t: t for t in combined_tickers}
        tickers[cfg.US_DEFENSE_ASSET] = cfg.US_DEFENSE_ASSET
    except Exception:
        return None

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        market_df = fdr.DataReader(cfg.US_MARKET_INDEX, start=start_date, end=end_date)
        raw_data = fetch_data_in_parallel(tickers, start_date, end_date)
    except Exception:
        return None

    w1, w2, w3 = cfg.MOMENTUM_WEIGHTS
    score = (raw_data.pct_change(20).iloc[-1].fillna(0)*w1) + (raw_data.pct_change(60).iloc[-1].fillna(0)*w2) + (raw_data.pct_change(120).iloc[-1].fillna(0)*w3)

    market_ma = market_df['Close'].ffill().rolling(60).mean()
    is_bull = market_df['Close'].iloc[-1] > market_ma.iloc[-1]
    is_neutral = not is_bull and (market_ma.iloc[-1] > market_ma.iloc[-6])
    status = "상승장" if is_bull else "중립장" if is_neutral else "하락장"
    reason = "적극투자" if is_bull else "분산투자" if is_neutral else "현금방어"

    selected = [n for n, s in score.drop(cfg.US_DEFENSE_ASSET, errors='ignore').sort_values(ascending=False).items() if s > 0][:cfg.US_TOP_N]

    if is_bull: targets = [(s, 1.0/len(selected)) for s in selected] if selected else [(cfg.US_DEFENSE_ASSET, 1.0)]
    elif is_neutral: targets = ([(s, 0.5/len(selected)) for s in selected] + [(cfg.US_DEFENSE_ASSET, 0.5)]) if selected else [(cfg.US_DEFENSE_ASSET, 1.0)]
    else: targets = [(cfg.US_DEFENSE_ASSET, 1.0)]

    all_ranks = []
    for i, (n, s) in enumerate(score.drop(cfg.US_DEFENSE_ASSET, errors='ignore').sort_values(ascending=False).items(), 1):
        price = raw_data[n].iloc[-1] if n in raw_data.columns else 0
        all_ranks.append({'rank': i, 'name': n, 'code': n, 'score': s, 'price': price})

    return {"status": status, "reason": reason, "targets": targets, "rankings": all_ranks, "raw_data_last": raw_data.iloc[-1]}

def normalize_holdings(raw_input, name_to_ticker):
    tokens = (
        raw_input.replace(",", " ")
    .replace("\n", " ")
    .replace("\t", " ")
        .split(" ")
    )
    tokens = [t.strip() for t in tokens if t.strip()]

    resolved = []
    unresolved = []

    for item in tokens:
        if item.isdigit() and len(item) == 6:
            resolved.append(item)
            continue

        if item in name_to_ticker:
            resolved.append(name_to_ticker[item])
            continue

        # 부분 일치 검색
        partial = [name for name in name_to_ticker.keys() if item in name]
        if partial:
            resolved.append(name_to_ticker[partial[0]])
        else:
            unresolved.append(item)

    return list(dict.fromkeys(resolved)), unresolved
