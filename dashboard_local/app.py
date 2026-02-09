import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import FinanceDataReader as fdr
from pykrx import stock
import os
import json
import sys
import time
import pickle
import logging
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import argrelextrema
from prophet import Prophet
from prophet.plot import plot_plotly
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="로컬 대시보드", layout="wide", page_icon="📊")

st.markdown("# 로컬 대시보드")
st.caption("보유종목 관리 → 펀더멘탈/차트 확인")

st.markdown(
    """
    <style>
        :root {
            --card-bg: #f7f6f3;
            --card-border: #e6e2da;
            --muted: #6b7280;
            --pill-bg: rgba(13, 148, 136, 0.12);
            --pill-text: #0f766e;
            --metric-bg: #f2f4f7;
            --metric-border: #e5e7eb;
            --divider: #e5e7eb;
            --grid: rgba(0,0,0,0.05);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --card-bg: #0f172a;
                --card-border: rgba(255,255,255,0.08);
                --muted: #9aa0a6;
                --pill-bg: rgba(14, 165, 233, 0.18);
                --pill-text: #7dd3fc;
                --metric-bg: #111827;
                --metric-border: rgba(255,255,255,0.08);
                --divider: rgba(255,255,255,0.08);
                --grid: rgba(255,255,255,0.08);
            }
        }
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
        h1 {margin-bottom: 0.2rem;}
        h2, h3 {margin-top: 0.6rem;}
        .section-title {font-size: 1.1rem; font-weight: 700; margin: 0.6rem 0 0.4rem 0;}
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 8px 20px rgba(30, 64, 175, 0.08);
        }
        .muted {color: var(--muted); font-size: 0.85rem;}
        .pill {display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; background: var(--pill-bg); color: var(--pill-text); font-size: 0.75rem;}
        [data-testid="stMetric"] {background: var(--metric-bg); border-radius: 12px; padding: 0.5rem 0.8rem; border: 1px solid var(--metric-border);}
        [data-testid="stMetric"] label {color: var(--muted);}        
        [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
        .divider {margin: 0.8rem 0; border-bottom: 1px solid var(--divider);} 
        @media (max-width: 640px) {
            .block-container {padding-left: 0.8rem !important; padding-right: 0.8rem !important;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MOMENTUM_DATA_FILE = os.path.join(DATA_DIR, "momentum_dashboard.pkl")
HOLDINGS_FILE = os.path.join(DATA_DIR, "holdings.json")

def load_holdings_from_disk():
    if os.path.exists(HOLDINGS_FILE):
        try:
            with open(HOLDINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []

def save_holdings_to_disk(rows):
    try:
        with open(HOLDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def save_momentum_data_to_disk(data):
    try:
        with open(MOMENTUM_DATA_FILE, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception:
        return False


def load_momentum_data_from_disk():
    if os.path.exists(MOMENTUM_DATA_FILE):
        try:
            with open(MOMENTUM_DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def initialize_session_state():
    if "holdings" not in st.session_state:
        st.session_state["holdings"] = load_holdings_from_disk()

    if "holdings_query" not in st.session_state:
        st.session_state["holdings_query"] = False

    if 'cached_data' not in st.session_state:
        loaded_data = load_momentum_data_from_disk()
        if loaded_data:
            st.session_state['cached_data'] = loaded_data
        else:
            st.session_state['cached_data'] = {'etf': None, 'stock': None, 'us': None, 'last_update': '-'}

    if 'ticker_for_rr' not in st.session_state:
        st.session_state['ticker_for_rr'] = ""

    if 'price_for_rr' not in st.session_state:
        st.session_state['price_for_rr'] = 0.0

    if 'use_candlestick' not in st.session_state:
        st.session_state['use_candlestick'] = True

    if 'show_rr_lines' not in st.session_state:
        st.session_state['show_rr_lines'] = True

initialize_session_state()

# [설정] 스레드 컨텍스트 경고 메시지 차단 (기능에는 영향 없음)
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)



CANDLE_UP_COLOR = '#26A69A' # Green for increasing candles
CANDLE_DOWN_COLOR = '#EF5350' # Red for decreasing candles

# 기존 프로젝트 모듈 경로 추가
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from common import fetch_data_in_parallel  # noqa: E402
import config as cfg  # noqa: E402


def init_font():
    font_filename = "NanumGothic.ttf"
    font_path = os.path.join(os.path.dirname(__file__), font_filename)

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        if platform.system() == 'Darwin':
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='Malgun Gothic')

    plt.rcParams['axes.unicode_minus'] = False


init_font()
plt.style.use('ggplot')




@st.cache_data(ttl=60 * 60)
def get_ticker_name_map():
    tickers = []
    try:
        tickers.extend(stock.get_market_ticker_list(market="KOSPI"))
        tickers.extend(stock.get_market_ticker_list(market="KOSDAQ"))
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
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            return date_str, df
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


def calculate_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ichimoku(df):
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    tenkan = (high9 + low9) / 2

    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    kijun = (high26 + low26) / 2

    span_a = ((tenkan + kijun) / 2).shift(26)

    high52 = df['High'].rolling(window=52).max()
    low52 = df['Low'].rolling(window=52).min()
    span_b = ((high52 + low52) / 2).shift(26)

    chikou = df['Close'].shift(-26)

    return tenkan, kijun, span_a, span_b, chikou


def resample_ohlc(df, rule):
    ohlc = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return ohlc.dropna()


def plot_ichimoku_rsi(df, title, rr_data=None, show_rr=True):
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)
    rsi = calculate_rsi(df['Close'])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax_price, ax_rsi = axes

    ax_price.plot(df.index, df['Close'], color='#222', lw=1.2, label='Close')
    ax_price.plot(tenkan.index, tenkan, color='#e67e22', lw=1, label='Tenkan (9)')
    ax_price.plot(kijun.index, kijun, color='#2980b9', lw=1, label='Kijun (26)')
    ax_price.plot(span_a.index, span_a, color='#27ae60', lw=0.9, label='Span A')
    ax_price.plot(span_b.index, span_b, color='#c0392b', lw=0.9, label='Span B')
    ax_price.plot(chikou.index, chikou, color='#7f8c8d', lw=0.8, label='Chikou')

    ax_price.fill_between(span_a.index, span_a, span_b, where=span_a >= span_b, color='#2ecc71', alpha=0.08)
    ax_price.fill_between(span_a.index, span_a, span_b, where=span_a < span_b, color='#e74c3c', alpha=0.08)

    if rr_data and show_rr:
        entry = rr_data.get('entry')
        targets = rr_data.get('targets', [])
        stops = rr_data.get('stops', [])

        if entry:
            ax_price.axhline(entry, color='#2980b9', lw=2, ls='--', label='Entry', alpha=0.7)

        colors = ['#27ae60', '#f39c12', '#e74c3c']
        styles = [':', '--', '-']
        names = ['Scalp', 'Swing', 'Trend']
        for i, (tp, sl) in enumerate(zip(targets, stops)):
            ax_price.axhline(tp, color=colors[i], ls=styles[i], lw=1.5, alpha=0.6, label=f'{names[i]} TP')
            ax_price.axhline(sl, color=colors[i], ls=styles[i], lw=1.5, alpha=0.6, label=f'{names[i]} SL')

    ax_price.set_title(title, fontsize=10)
    ax_price.legend(loc='upper left', fontsize=6, ncol=4)
    ax_price.grid(True, alpha=0.3)

    ax_rsi.plot(rsi.index, rsi, color='#8e44ad', lw=1)
    ax_rsi.axhline(70, color='#c0392b', ls='--', lw=0.8)
    ax_rsi.axhline(30, color='#27ae60', ls='--', lw=0.8)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI', fontsize=8)
    ax_rsi.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_dynamic_ichimoku_rsi(df, title, entry_price=None, rr_data=None, plot_candlestick=False, show_rr=True):
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)
    rsi = calculate_rsi(df['Close'])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "RSI")
    )
    if plot_candlestick:
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     increasing=dict(line=dict(color=CANDLE_UP_COLOR, width=1), fillcolor=CANDLE_UP_COLOR),
                                     decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR, width=1), fillcolor=CANDLE_DOWN_COLOR),
                                     name='Candlestick'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#1f2937', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tenkan.index, y=tenkan, name='Tenkan', line=dict(color='#f59e0b', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kijun.index, y=kijun, name='Kijun', line=dict(color='#3b82f6', width=1)), row=1, col=1)
    # Cloud (Span A/B)
    fig.add_trace(go.Scatter(x=span_a.index, y=span_a, name='Span A', line=dict(color='#10b981', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=span_b.index, y=span_b, name='Span B', line=dict(color='#ef4444', width=1), fill='tonexty', fillcolor='rgba(16,185,129,0.15)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chikou.index, y=chikou, name='Chikou', line=dict(color='#6b7280', width=1)), row=1, col=1)

    if entry_price:
        fig.add_hline(y=entry_price, line=dict(color='#0ea5e9', width=2, dash='dash'), row=1, col=1)

    if rr_data and show_rr:
        targets = rr_data.get('targets', [])
        stops = rr_data.get('stops', [])
        colors = ['#16a34a', '#f59e0b', '#ef4444']
        for i, tp in enumerate(targets):
            fig.add_hline(y=tp, line=dict(color=colors[i % len(colors)], width=1, dash='dot'), row=1, col=1)
        for i, sl in enumerate(stops):
            fig.add_hline(y=sl, line=dict(color=colors[i % len(colors)], width=1, dash='dot'), row=1, col=1)

    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='#8b5cf6', width=1)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='#ef4444', width=1, dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='#10b981', width=1, dash='dash'), row=2, col=1)

    fig.update_layout(height=520, showlegend=True, legend_orientation='h', legend_y=1.02, legend_x=0,
                      legend=dict(font=dict(color='#E0E0E0')),
                      plot_bgcolor='#131722', # TradingView dark theme background
                      paper_bgcolor='#131722', # TradingView dark theme paper background
                      font=dict(color='#E0E0E0'), # Light text for dark theme
                      xaxis=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39'),
                      yaxis=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39')
                     )

    return fig


def plot_cloud_bbands_rr(df, title, entry_price=None, rr_data=None, plot_candlestick=False, show_rr=True):
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)

    fig = go.Figure()

    if plot_candlestick:
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     increasing=dict(line=dict(color=CANDLE_UP_COLOR, width=1), fillcolor=CANDLE_UP_COLOR),
                                     decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR, width=1), fillcolor=CANDLE_DOWN_COLOR),
                                     name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#1f2937', width=2)))

    # Ichimoku cloud only
    fig.add_trace(go.Scatter(x=span_a.index, y=span_a, name='Span A', line=dict(color='#10b981', width=1)))
    fig.add_trace(go.Scatter(x=span_b.index, y=span_b, name='Span B', line=dict(color='#ef4444', width=1), fill='tonexty', fillcolor='rgba(16,185,129,0.15)'))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=upper.index, y=upper, name='BB Upper', line=dict(color='#6366f1', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=ma20.index, y=ma20, name='BB Mid', line=dict(color='#6366f1', width=1)))
    fig.add_trace(go.Scatter(x=lower.index, y=lower, name='BB Lower', line=dict(color='#6366f1', width=1, dash='dot')))

    if entry_price:
        fig.add_hline(y=entry_price, line=dict(color='#0ea5e9', width=2, dash='dash'))

    if rr_data and show_rr:
        targets = rr_data.get('targets', [])
        stops = rr_data.get('stops', [])
        colors = ['#16a34a', '#f59e0b', '#ef4444']
        for i, tp in enumerate(targets):
            fig.add_hline(y=tp, line=dict(color=colors[i % len(colors)], width=1, dash='dot'))
        for i, sl in enumerate(stops):
            fig.add_hline(y=sl, line=dict(color=colors[i % len(colors)], width=1, dash='dot'))

    fig.update_layout(height=520, showlegend=True, legend_orientation='h', legend_y=1.02, legend_x=0, title=title,
                      legend=dict(font=dict(color='#E0E0E0')),
                      plot_bgcolor='#131722', # TradingView dark theme background
                      paper_bgcolor='#131722', # TradingView dark theme paper background
                      font=dict(color='#E0E0E0'), # Light text for dark theme
                      xaxis=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39'),
                      yaxis=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39')
                     )
    return fig


def _prophet_df_from_price(df):
    df_reset = df.reset_index()
    date_col = None
    for candidate in ['Date', 'date', '날짜']:
        if candidate in df_reset.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df_reset.columns[0]
    df_prophet = df_reset[[date_col, 'Close']].rename(columns={date_col: 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    # Regressors: RSI, MA20, MA60
    close_series = pd.to_numeric(df_reset['Close'], errors='coerce')
    rsi = calculate_rsi(close_series)
    ma20 = close_series.rolling(20).mean()
    ma60 = close_series.rolling(60).mean()
    df_prophet['rsi'] = rsi.values
    df_prophet['ma20'] = ma20.values
    df_prophet['ma60'] = ma60.values
    # RSI/이평 결측 보정 (모멘텀 RSI 알고리즘 기반 값 사용)
    df_prophet['rsi'] = df_prophet['rsi'].fillna(50)
    df_prophet['ma20'] = df_prophet['ma20'].fillna(df_prophet['y']).ffill().bfill()
    df_prophet['ma60'] = df_prophet['ma60'].fillna(df_prophet['y']).ffill().bfill()
    df_prophet = df_prophet.dropna(subset=['ds', 'y', 'rsi', 'ma20', 'ma60'])
    return df_prophet


def plot_prophet_forecast(df, periods=30, title="Prophet Forecast"):
    df_prophet = _prophet_df_from_price(df)
    if len(df_prophet) < 30:
        raise ValueError("Prophet 예측을 위한 데이터가 부족합니다.")
    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
    model.add_regressor('rsi')
    model.add_regressor('ma20')
    model.add_regressor('ma60')
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    # Future regressors: hold last available value
    last_rsi = df_prophet['rsi'].dropna().iloc[-1] if not df_prophet['rsi'].dropna().empty else 50
    last_ma20 = df_prophet['ma20'].dropna().iloc[-1] if not df_prophet['ma20'].dropna().empty else df_prophet['y'].iloc[-1]
    last_ma60 = df_prophet['ma60'].dropna().iloc[-1] if not df_prophet['ma60'].dropna().empty else df_prophet['y'].iloc[-1]
    future['rsi'] = list(df_prophet['rsi']) + [last_rsi] * (len(future) - len(df_prophet))
    future['ma20'] = list(df_prophet['ma20']) + [last_ma20] * (len(future) - len(df_prophet))
    future['ma60'] = list(df_prophet['ma60']) + [last_ma60] * (len(future) - len(df_prophet))
    forecast = model.predict(future)
    return forecast


def build_forecast_chart(df, forecast, title="Prophet Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#1f2937', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0ea5e9', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper', line=dict(color='rgba(14,165,233,0.2)', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower', line=dict(color='rgba(14,165,233,0.2)', width=1), fill='tonexty', fillcolor='rgba(14,165,233,0.15)', showlegend=False))
    fig.update_layout(height=520, showlegend=True, legend_orientation='h', legend_y=1.02, legend_x=0, title=title)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    return fig


def compute_prophet_forecast(df, periods=30):
    return plot_prophet_forecast(df, periods=periods, title="Prophet Forecast")


def compute_neuralprophet_forecast(df, periods=30):
    """NeuralProphet 모델을 사용하여 미래 가격을 예측합니다."""
    try:
        from neuralprophet import NeuralProphet
        import torch
    except ImportError:
        raise ImportError("NeuralProphet이 설치되지 않았습니다. pip install neuralprophet 실행 필요")
    
    # 랜덤 시드 고정으로 일관된 결과 생성
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    df_prophet = _prophet_df_from_price(df)
    df_prophet = df_prophet[['ds', 'y', 'rsi', 'ma20', 'ma60']].copy()
    df_prophet = df_prophet.sort_values('ds')
    
    if len(df_prophet) <= 30:
        raise ValueError(f"학습 데이터 길이({len(df_prophet)})가 30일보다 짧아 예측할 수 없습니다.")
    
    # 단순하고 안정적인 1-step 예측 방식 사용
    model = NeuralProphet(
        n_lags=30,
        n_forecasts=1,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        learning_rate=0.01,
        epochs=150
    )
    
    # RSI, MA20, MA60을 lagged regressor로 등록 (미래 값을 알 수 없으므로)
    model.add_lagged_regressor('rsi')
    model.add_lagged_regressor('ma20')
    model.add_lagged_regressor('ma60')
    
    model.fit(df_prophet, freq='D')
    
    # 과거 fitted 값 먼저 구하기
    historic_predictions = model.predict(df_prophet)
    
    # Iterative 방식으로 미래 예측 (한 단계씩 예측하고 결과를 다음 입력으로 사용)
    current_df = df_prophet.copy()
    future_predictions = []
    
    for i in range(periods):
        # 1일 예측
        future_1step = model.make_future_dataframe(current_df, periods=1, n_historic_predictions=0)
        pred = model.predict(future_1step)
        
        # 예측 결과 저장
        if not pred.empty:
            last_pred = pred.iloc[-1]
            future_predictions.append({
                'ds': last_pred['ds'],
                'yhat': last_pred.get('yhat1', last_pred.get('yhat', 0))
            })
            
            # 예측 결과를 다음 입력 데이터에 추가 (autoregressive)
            next_row = pd.DataFrame({
                'ds': [last_pred['ds']],
                'y': [last_pred.get('yhat1', last_pred.get('yhat', current_df['y'].iloc[-1]))],
                'rsi': [current_df['rsi'].iloc[-1]],
                'ma20': [current_df['ma20'].iloc[-1]],
                'ma60': [current_df['ma60'].iloc[-1]]
            })
            current_df = pd.concat([current_df, next_row], ignore_index=True)
    
    # 과거 + 미래 결합
    if 'yhat1' in historic_predictions.columns:
        historic_predictions['yhat'] = historic_predictions['yhat1']
    
    future_df = pd.DataFrame(future_predictions)
    forecast = pd.concat([historic_predictions[['ds', 'yhat']], future_df], ignore_index=True)
    
    # 신뢰구간 추가 (NeuralProphet은 기본 제공 안함, yhat의 ±5%로 설정)
    if 'yhat' in forecast.columns:
        forecast['yhat_lower'] = forecast['yhat'] * 0.95
        forecast['yhat_upper'] = forecast['yhat'] * 1.05
    
    return forecast


def compute_xgboost_forecast(df, periods=5):
    """XGBoost 분류 모델을 사용하여 미래 상승/하락 확률을 예측합니다."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost가 설치되지 않았습니다. pip install xgboost 실행 필요")
    
    from sklearn.preprocessing import StandardScaler
    
    # 랜덤 시드 고정으로 일관된 결과 생성
    np.random.seed(42)
    
    df_clean = df[['Close']].copy()
    df_clean['MA5'] = df_clean['Close'].rolling(5).mean()
    df_clean['MA20'] = df_clean['Close'].rolling(20).mean()
    df_clean['MA60'] = df_clean['Close'].rolling(60).mean()
    df_clean['RSI'] = calculate_rsi(df_clean['Close'])
    df_clean = df_clean.dropna()
    
    if len(df_clean) < 100:
        raise ValueError("XGBoost 예측을 위한 데이터가 부족합니다.")
    
    # 특징 생성: 이전 5일 가격 + 기술지표
    X, y = [], []
    lookback = 5
    for i in range(lookback, len(df_clean) - 1):  # -1: 다음날 라벨을 위해
        features = list(df_clean['Close'].iloc[i-lookback:i].values)
        features.extend([
            df_clean['MA5'].iloc[i],
            df_clean['MA20'].iloc[i],
            df_clean['MA60'].iloc[i],
            df_clean['RSI'].iloc[i]
        ])
        X.append(features)
        # 타겟: 다음날 상승(1) or 하락(0)
        next_close = df_clean['Close'].iloc[i + 1]
        current_close = df_clean['Close'].iloc[i]
        y.append(1 if next_close > current_close else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    # 학습/검증 분리
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # XGBoost 분류 모델 학습
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 미래 상승 확률 예측 (iterative)
    predictions = []  # 상승 확률 저장
    last_sequence = list(df_clean['Close'].iloc[-lookback:].values)
    last_ma5 = df_clean['MA5'].iloc[-1]
    last_ma20 = df_clean['MA20'].iloc[-1]
    last_ma60 = df_clean['MA60'].iloc[-1]
    last_rsi = df_clean['RSI'].iloc[-1]
    
    for i in range(periods):
        features = last_sequence[-lookback:] + [last_ma5, last_ma20, last_ma60, last_rsi]
        features_scaled = scaler.transform([features])
        # 상승 확률 (클래스 1의 확률)
        prob_up = model.predict_proba(features_scaled)[0][1]
        predictions.append(prob_up)
        
        # 다음 예측을 위해 가격 업데이트 (확률 기반 예측가)
        current_price = last_sequence[-1]
        # 상승 확률이 0.5 이상이면 1% 상승, 아니면 1% 하락 가정
        next_price = current_price * (1.01 if prob_up > 0.5 else 0.99)
        last_sequence.append(next_price)
    
    # 결과 반환 (확률 데이터)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    result = pd.DataFrame({
        'ds': future_dates,
        'probability': predictions  # 상승 확률
    })
    
    return result


def plot_support_resistance(df, order=20, title="Support/Resistance"):
    close = df['Close'].values
    local_max_idx = argrelextrema(close, np.greater, order=order)[0]
    local_min_idx = argrelextrema(close, np.less, order=order)[0]

    local_max_prices = close[local_max_idx]
    local_min_prices = close[local_min_idx]

    current_price = close[-1]
    nearest_support = local_min_prices[local_min_prices < current_price].max() if any(local_min_prices < current_price) else current_price * 0.9
    nearest_resistance = local_max_prices[local_max_prices > current_price].min() if any(local_max_prices > current_price) else current_price * 1.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=close, name='Close', line=dict(color='#1f2937', width=2)))
    if len(local_max_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[local_max_idx], y=local_max_prices, mode='markers', name='Resistance', marker=dict(color='red', size=6, symbol='triangle-down')))
    if len(local_min_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[local_min_idx], y=local_min_prices, mode='markers', name='Support', marker=dict(color='green', size=6, symbol='triangle-up')))

    fig.add_hline(y=nearest_support, line=dict(color='green', width=1, dash='dash'))
    fig.add_hline(y=nearest_resistance, line=dict(color='red', width=1, dash='dash'))

    fig.update_layout(height=520, showlegend=True, legend_orientation='h', legend_y=1.02, legend_x=0, title=title)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    return fig, nearest_support, nearest_resistance


def ui_card_header(title, status, reason):
    color = "red" if "상승" in status else "orange" if "중립" in status else "blue"
    icon = "🔴" if "상승" in status else "🟠" if "중립" in status else "🔵"
    c1, c2 = st.columns([1.5, 1])
    with c1: st.markdown(f"**{title}**")
    with c2: st.markdown(f"{icon} :{color}[**{status}**] <span style='font-size:0.8em; color:gray'>({reason})</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0.3rem 0;'>", unsafe_allow_html=True)


def ui_target_row(rank, name, code, weight, price, is_us=False):
    with stylable_container(
        key=f"target_{code}_{rank}",
        css_styles="""
            [data-testid="stHorizontalBlock"] > div {
                min-width: 0 !important;
            }
            @media (max-width: 640px) {
                button {
                    font-size: 0.8rem !important;
                    padding: 0.2rem 0.3rem !important;
                    height: 1.8rem !important;
                }
            }
        """
    ):
        c1, c2, c3, c4 = st.columns([2.5, 2, 1.2, 0.8])
        with c1:
            st.markdown(f"<div style='margin-bottom: -0.5rem;'>{rank}. {name}</div>", unsafe_allow_html=True)
            if code and code != name: st.caption(f"{code}")
        with c2:
            st.progress(weight)
            st.caption(f"{weight*100:.0f}%")
        with c3:
            if is_us: st.write(f"${price:,.2f}")
            else: st.write(f"{int(price):,}원")
        with c4:
            if code and code != "N/A":
                st.button("🔍", key=f"btn_{code}_{rank}_{int(time.time())}", on_click=set_analysis_target, args=(code, price))


def ui_ranking_list(rank_data, is_us=False, limit=50):
    unique_key = f"ranking_header_{id(rank_data)}"
    with stylable_container(
        key=unique_key,
        css_styles="""
            [data-testid="stHorizontalBlock"] > div {
                min-width: 0 !important;
            }
        """
    ):
        c1, c2, c3, c4, c5 = st.columns([0.7, 2.0, 1.2, 1.5, 1.7])
        c1.caption("No.")
        c2.caption("종목명")
        c3.caption("점수")
        c4.caption("현재가")
        c5.caption("분석")
    st.markdown("<hr style='margin: 0.1rem 0;'>", unsafe_allow_html=True)

    for item in rank_data[:limit]:
        with stylable_container(
            key=f"ranking_{item['code']}_{item['rank']}",
            css_styles="""
                [data-testid="stHorizontalBlock"] > div {
                    min-width: 0 !important;
                }
            """
        ):
            c1, c2, c3, c4, c5 = st.columns([0.7, 2.0, 1.2, 1.5, 1.7])
            with c1: st.write(f"**{item['rank']}**")
            with c2: st.write(f"{item['name']}")
            with c3:
                color = "red" if item['score'] > 0 else "blue"
                st.markdown(f":{color}[{item['score']:.2f}]")
            with c4:
                if is_us: st.write(f"${item['price']:,.2f}")
                else: st.write(f"{int(item['price']):,}원")
            with c5:
                code_label = item['code'] if item['code'] and item['code'] != "N/A" else "N/A"
                if code_label != "N/A":
                    st.button(f"{code_label}", key=f"rk_btn_{item['code']}_{item['rank']}_{int(time.time())}", on_click=set_analysis_target, args=(item['code'], item['price']), use_container_width=True)
                else: st.caption("-")
        st.markdown("<hr style='margin: 0.1rem 0; opacity: 0.3;'>", unsafe_allow_html=True)


def set_analysis_target(ticker, price):
    st.session_state['ticker_for_rr'] = ticker
    st.session_state['price_for_rr'] = float(price)


def sync_show_rr_lines(src_key):
    st.session_state['show_rr_lines'] = st.session_state.get(src_key, True)


def sync_use_candlestick(src_key):
    st.session_state['use_candlestick'] = st.session_state.get(src_key, True)


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


def render_left_card(title, data, asset_type):
    with st.container(border=True):
        if not data:
            st.warning(f"{title} 데이터 없음")
            return

        ui_card_header(title, data['status'], data['reason'])

        is_us_asset = (asset_type == 'us')

        for i, (name, weight) in enumerate(data['targets']):
            if asset_type == 'etf':
                code = cfg.ETF_TICKERS.get(name, "N/A")
            elif asset_type == 'stock':
                code = data['tickers_map'].get(name, name)
            else:
                code = name
            price = data['raw_data_last'].get(name, 0)
            ui_target_row(i+1, name, code, weight, price, is_us=is_us_asset)

        with st.expander("🔻 전체 순위 보기 (Top 50)"):
            if data['rankings']:
                ui_ranking_list(data['rankings'], is_us=is_us_asset, limit=50)
            else:
                st.info("순위 데이터가 없습니다.")


class UniversalRiskRewardCalculator:
    def calculate_atr(self, df, period):
        tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def analyze(self, ticker, entry_price):
        df = fdr.DataReader(ticker, end=datetime.now().strftime('%Y-%m-%d'), start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        if df.empty: return None, None

        current_price = df['Close'].iloc[-1]
        if entry_price == 0: entry_price = current_price

        strategies = [
            {"name": "Scalping", "atr_period": 14, "risk_mult": 1.5, "reward_ratio": 1.5, "style": ":"},
            {"name": "Swing", "atr_period": 22, "risk_mult": 2.5, "reward_ratio": 2.0, "style": "--"},
            {"name": "Trend", "atr_period": 60, "risk_mult": 3.5, "reward_ratio": 3.0, "style": "-"}
        ]
        results = []
        targets = []
        stops = []
        for s in strategies:
            atr = self.calculate_atr(df, s['atr_period']).iloc[-1]
            risk = atr * s['risk_mult']
            stop, tp = entry_price - risk, entry_price + (risk * s['reward_ratio'])
            results.append({"Mode": s['name'], "Target": tp, "Stop": stop, "R/R": f"1:{s['reward_ratio']}", "Risk": f"-{(entry_price-stop)/entry_price*100:.1f}%"})
            targets.append(tp)
            stops.append(stop)

        rr_data = {
            'entry': entry_price,
            'targets': targets,
            'stops': stops
        }

        return pd.DataFrame(results), rr_data


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


tabs = st.tabs(["보유종목", "모멘텀"])

with tabs[0]:
    st.markdown("### 보유종목")
    st.caption("보유종목을 추가/편집/저장하고, 한눈에 성과와 펀더멘탈을 확인합니다.")

    name_to_ticker, ticker_to_name = get_ticker_name_map()

    # 좌우 2컬럼 배치: 보유종목 추가 | 보유종목 리스트
    col_add, col_list = st.columns([1, 2])
    
    with col_add:
        with st.expander("보유종목 추가", expanded=False):
            r1c1, r1c2, r1c3 = st.columns([2, 1, 1])
            with r1c1:
                add_input = st.text_input("종목명/티커", value="005930")
            with r1c2:
                add_qty = st.number_input("수량", min_value=0.0, value=0.0, step=1.0)
            with r1c3:
                add_avg = st.number_input("평균단가", min_value=0.0, value=0.0, step=100.0)

            r2c1, r2c2 = st.columns([3, 1])
            with r2c1:
                add_memo = st.text_input("메모", value="")
            with r2c2:
                add_btn = st.button("추가", use_container_width=True)

            if add_btn:
                resolved, unresolved = normalize_holdings(add_input, name_to_ticker)
                if unresolved:
                    st.warning(f"인식하지 못한 항목: {', '.join(unresolved)}")
                if not resolved:
                    st.info("추가할 종목이 없습니다.")
                else:
                    for t in resolved:
                        row = {
                            "티커": t,
                            "종목명": ticker_to_name.get(t, t),
                            "보유수량": float(add_qty),
                            "평균단가": float(add_avg),
                            "메모": add_memo,
                            "삭제": False,
                        }
                        st.session_state["holdings"].append(row)
                    st.success("추가 완료")
    
    with col_list:
        st.markdown("**보유종목 리스트**")
        holdings_df = pd.DataFrame(st.session_state["holdings"])
        if holdings_df.empty:
            holdings_df = pd.DataFrame(columns=["티커", "종목명", "보유수량", "평균단가", "메모", "삭제"])

        edited_df = st.data_editor(
            holdings_df,
            use_container_width=True,
            hide_index=True,
            height=240,
            column_config={
                "티커": st.column_config.TextColumn("티커"),
                "종목명": st.column_config.TextColumn("종목명"),
                "보유수량": st.column_config.NumberColumn("보유수량", format="%,.0f"),
                "평균단가": st.column_config.NumberColumn("평균단가", format="%,.0f"),
                "메모": st.column_config.TextColumn("메모"),
                "삭제": st.column_config.CheckboxColumn("삭제"),
            },
        )

        c_save, c_delete, c_refresh = st.columns([1, 1, 1])
        with c_save:
            save_btn = st.button("저장")
        with c_delete:
            delete_btn = st.button("선택 삭제")
        with c_refresh:
            reload_btn = st.button("새로고침")

        if save_btn:
            rows = edited_df.to_dict(orient="records")
            rows = [r for r in rows if str(r.get("티커", "")).strip()]
            if save_holdings_to_disk(rows):
                st.session_state["holdings"] = rows
                st.success("저장 완료")
            else:
                st.error("저장 실패")

        if delete_btn:
            rows = edited_df.to_dict(orient="records")
            rows = [r for r in rows if not r.get("삭제")]
            st.session_state["holdings"] = rows
            if save_holdings_to_disk(rows):
                st.success("삭제 후 저장 완료")
            else:
                st.error("삭제 저장 실패")

        if reload_btn:
            st.session_state["holdings"] = load_holdings_from_disk()
            st.info("디스크에서 다시 불러왔습니다.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>조회 옵션</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        price_lookback_days = st.number_input("가격 조회 기간(일)", min_value=30, max_value=1825, value=365, step=30)
    with col2:
        fundamental_lookback_years = st.number_input("펀더멘탈 조회 기간(년)", min_value=1, max_value=10, value=3, step=1)
    with col3:
        run = st.button("펀더멘탈/차트 조회")
    with col4:
        history_rows = st.selectbox("히스토리 표시 개수", options=[12, 24, 60, "ALL"], index=3)

    if run:
        st.session_state["holdings_query"] = True

    if st.session_state.get("holdings_query"):
        current_df = pd.DataFrame(st.session_state["holdings"])
        tickers = [t for t in current_df.get("티커", []).tolist() if isinstance(t, str) and t.strip()]

        if not tickers:
            st.info("조회할 종목이 없습니다.")
        else:
            date_str, funda = get_latest_fundamental()
            if date_str is None or funda.empty:
                st.error("펀더멘탈 데이터를 가져오지 못했습니다.")
            else:
                funda = funda.copy()
                funda.index.name = "티커"
                funda = funda.loc[funda.index.intersection(tickers)]

                if funda.empty:
                    st.warning("해당 종목의 펀더멘탈 데이터가 없습니다.")
                else:
                    funda["ROE(%)"] = (funda["EPS"] / funda["BPS"]).replace([pd.NA, pd.NaT, float("inf")], pd.NA) * 100
                    funda.insert(0, "종목명", [ticker_to_name.get(t, t) for t in funda.index])
                    funda.insert(1, "티커", funda.index)

                    st.markdown(f"<span class='pill'>기준일 {date_str}</span>", unsafe_allow_html=True)
                    summary_df = funda[["종목명", "티커", "EPS", "PER", "PBR", "BPS", "DIV", "DPS", "ROE(%)"]].reset_index(drop=True)
                    summary_fmt = {
                        "EPS": "{:,.2f}",
                        "PER": "{:,.2f}",
                        "PBR": "{:,.2f}",
                        "BPS": "{:,.2f}",
                        "DIV": "{:,.2f}",
                        "DPS": "{:,.2f}",
                        "ROE(%)": "{:,.2f}",
                    }
                    st.dataframe(summary_df.style.format(summary_fmt), use_container_width=True)

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.subheader("!벌어야 한다!")

                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=int(price_lookback_days))).strftime("%Y-%m-%d")
                    f_start_date = (datetime.now() - timedelta(days=int(fundamental_lookback_years) * 365)).strftime("%Y%m%d")
                    f_end_date = datetime.now().strftime("%Y%m%d")

                    holdings_map = {r.get("티커"): r for r in st.session_state.get("holdings", [])}

                    def slice_hist(df):
                        if history_rows == "ALL":
                            return df
                        return df.tail(int(history_rows))

                    for t in tickers:
                        name = ticker_to_name.get(t, t)
                        st.markdown(f"<div class='card'><div class='section-title'>{name} ({t})</div>", unsafe_allow_html=True)

                        price_df = load_price_data(t, start_date, end_date)
                        current_price = None
                        if not price_df.empty and "Close" in price_df.columns:
                            current_price = float(price_df["Close"].iloc[-1])

                        holding = holdings_map.get(t, {})
                        qty = float(holding.get("보유수량", 0) or 0)
                        avg = float(holding.get("평균단가", 0) or 0)
                        cost = qty * avg
                        value = qty * current_price if current_price is not None else None
                        pnl = (value - cost) if value is not None else None
                        pnl_pct = (pnl / cost * 100) if cost > 0 and pnl is not None else None

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("현재가", f"{current_price:,.0f}원" if current_price is not None else "-")
                        m2.metric("보유수량", f"{qty:,.0f}")
                        m3.metric("평가금액", f"{value:,.0f}원" if value is not None else "-")
                        m4.metric("평가손익", f"{pnl:,.0f}원" if pnl is not None else "-", f"{pnl_pct:.2f}%" if pnl_pct is not None else None)

                        tab_overview, tab_price, tab_funda, tab_foreign, tab_sr = st.tabs(["요약", "가격", "펀더멘탈", "외인", "지지/예측"])

                        with tab_overview:
                            overview_df = funda.loc[[t]][["종목명", "티커", "EPS", "PER", "PBR", "BPS", "DIV", "DPS", "ROE(%)"]].reset_index(drop=True)
                            overview_fmt = {
                                "EPS": "{:,.2f}",
                                "PER": "{:,.2f}",
                                "PBR": "{:,.2f}",
                                "BPS": "{:,.2f}",
                                "DIV": "{:,.2f}",
                                "DPS": "{:,.2f}",
                                "ROE(%)": "{:,.2f}",
                            }
                            st.dataframe(overview_df.style.format(overview_fmt), use_container_width=True)

                        with tab_price:
                            if price_df.empty or "Close" not in price_df.columns:
                                st.caption("가격 데이터를 가져오지 못했습니다.")
                            else:
                                candle_key = f"use_candlestick_holdings_{t}"
                                st.checkbox(
                                    "봉차트 표시",
                                    value=st.session_state['use_candlestick'],
                                    key=candle_key,
                                    on_change=sync_use_candlestick,
                                    args=(candle_key,),
                                )
                                rr_key = f"show_rr_lines_holdings_{t}"
                                st.checkbox(
                                    "손익비 라인 표시",
                                    value=st.session_state['show_rr_lines'],
                                    key=rr_key,
                                    on_change=sync_show_rr_lines,
                                    args=(rr_key,),
                                )
                                entry_for_rr = avg if avg > 0 else float(price_df["Close"].iloc[-1])
                                rr_table, rr_data = UniversalRiskRewardCalculator().analyze(t, entry_for_rr)
                                view = price_df.copy()
                                st.plotly_chart(
                                    plot_cloud_bbands_rr(
                                        view,
                                        f"{name} 가격",
                                        entry_for_rr,
                                        rr_data,
                                        plot_candlestick=st.session_state['use_candlestick'],
                                        show_rr=st.session_state['show_rr_lines'],
                                    ),
                                    use_container_width=True
                                )

                                st.markdown("**가격 히스토리**")
                                price_view = price_df[["Close", "Volume"]] if "Volume" in price_df.columns else price_df[["Close"]]
                                price_hist = slice_hist(price_view).reset_index()
                                price_fmt = {"Close": "{:,.2f}"}
                                if "Volume" in price_hist.columns:
                                    price_fmt["Volume"] = "{:,.0f}"
                                st.dataframe(price_hist.style.format(price_fmt), use_container_width=True)

                        with tab_sr:
                            if price_df.empty or "Close" not in price_df.columns:
                                st.caption("가격 데이터를 가져오지 못했습니다.")
                            else:
                                applied_key = f"sr_order_applied_{t}"
                                if applied_key not in st.session_state:
                                    st.session_state[applied_key] = 5

                                with st.form(key=f"sr_form_{t}"):
                                    order_input = st.slider(
                                        "지지/저항 민감도",
                                        min_value=5,
                                        max_value=60,
                                        value=int(st.session_state[applied_key]),
                                        step=5,
                                        key=f"sr_order_input_{t}",
                                    )
                                    apply_order = st.form_submit_button("민감도 적용")

                                if apply_order:
                                    st.session_state[applied_key] = order_input

                                # 지지/저항 차트
                                fig_sr, sup, res = plot_support_resistance(
                                    price_df,
                                    order=int(st.session_state[applied_key]),
                                    title=f"{name} 지지/저항",
                                )
                                s1, s2, s3 = st.columns(3)
                                s1.metric("현재가", f"{float(price_df['Close'].iloc[-1]):,.0f}원")
                                s2.metric("지지선", f"{float(sup):,.0f}원")
                                s3.metric("저항선", f"{float(res):,.0f}원")
                                st.plotly_chart(fig_sr, use_container_width=True)
                                
                                st.divider()
                                st.markdown("**📈 AI 예측 모델 (30일)**")

                                ai_cache_key = f"ai_forecast_cache_{t}"
                                ai_sig = (
                                    len(price_df),
                                    str(price_df.index.max()),
                                    float(price_df['Close'].iloc[-1])
                                )

                                if (
                                    ai_cache_key not in st.session_state
                                    or st.session_state[ai_cache_key].get("sig") != ai_sig
                                ):
                                    try:
                                        with st.spinner("AI 모델 계산 중..."):
                                            forecast_prophet = compute_prophet_forecast(price_df, periods=30)
                                            forecast_np = compute_neuralprophet_forecast(price_df, periods=5)
                                            forecast_xgb = compute_xgboost_forecast(price_df, periods=5)
                                        st.session_state[ai_cache_key] = {
                                            "sig": ai_sig,
                                            "prophet": forecast_prophet,
                                            "neural": forecast_np,
                                            "xgboost": forecast_xgb,
                                        }
                                    except Exception as e:
                                        st.error(f"예측 실패: {e}")
                                        st.session_state[ai_cache_key] = None

                                if ai_cache_key in st.session_state and st.session_state[ai_cache_key]:
                                    cached = st.session_state[ai_cache_key]

                                    # 3개 모델을 2행으로 배치 (Prophet | NeuralProphet / XGBoost | 빈공간)
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("**Prophet**")
                                        try:
                                            fig_pf = build_forecast_chart(price_df, cached["prophet"], title=f"{name} Prophet")
                                            st.plotly_chart(fig_pf, use_container_width=True)
                                            last = cached["prophet"].iloc[-1]
                                            st.caption(f"예측: {last['yhat']:.2f} / 하단: {last.get('yhat_lower', 0):.2f} / 상단: {last.get('yhat_upper', 0):.2f}")
                                        except Exception as e:
                                            st.error(f"예측 실패: {e}")

                                    with col2:
                                        st.markdown("**NeuralProphet**")
                                        try:
                                            fig_np = build_forecast_chart(price_df, cached["neural"], title=f"{name} NeuralProphet")
                                            st.plotly_chart(fig_np, use_container_width=True)
                                            last_np = cached["neural"].iloc[-1]
                                            st.caption(f"예측: {last_np['yhat']:.2f}")
                                        except Exception as e:
                                            st.error(f"예측 실패: {e}")

                                    col3, col4 = st.columns(2)

                                    with col3:
                                        st.markdown("**XGBoost (상승확률)**")
                                        try:
                                            for idx, row in cached["xgboost"].iterrows():
                                                date_str = row['ds'].strftime('%m/%d')
                                                prob = row['probability']
                                                color = "green" if prob > 0.5 else "red"
                                                st.markdown(f"{date_str}: <span style='color:{color};font-weight:bold'>{prob*100:.1f}%</span> 상승", unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"예측 실패: {e}")

                        with tab_funda:
                            funda_hist = load_fundamental_history(t, f_start_date, f_end_date)
                            if funda_hist is None or funda_hist.empty:
                                st.caption("펀더멘탈 히스토리를 가져오지 못했습니다.")
                            else:
                                funda_hist = funda_hist.copy()
                                funda_hist["ROE(%)"] = (funda_hist["EPS"] / funda_hist["BPS"]).replace([pd.NA, pd.NaT, float("inf")], pd.NA) * 100

                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**EPS / BPS**")
                                    st.line_chart(funda_hist[["EPS", "BPS"]], use_container_width=True)
                                with c2:
                                    st.markdown("**PER / PBR**")
                                    st.line_chart(funda_hist[["PER", "PBR"]], use_container_width=True)

                                chart_src = funda_hist[["ROE(%)", "DIV", "DPS"]].copy()
                                chart_src = chart_src.apply(pd.to_numeric, errors="coerce")
                                chart_src = chart_src.replace([float("inf"), float("-inf")], pd.NA)

                                if chart_src.dropna(how="all").empty:
                                    st.caption("표시할 데이터가 없습니다.")
                                else:
                                    c3, c4, c5 = st.columns(3)
                                    with c3:
                                        st.markdown("**ROE(%)**")
                                        if chart_src[["ROE(%)"]].dropna(how="all").empty:
                                            st.caption("ROE 데이터가 없습니다.")
                                        else:
                                            st.line_chart(chart_src[["ROE(%)"]], use_container_width=True)
                                    with c4:
                                        st.markdown("**DIV**")
                                        if chart_src[["DIV"]].dropna(how="all").empty:
                                            st.caption("DIV 데이터가 없습니다.")
                                        else:
                                            st.line_chart(chart_src[["DIV"]], use_container_width=True)
                                    with c5:
                                        st.markdown("**DPS**")
                                        if chart_src[["DPS"]].dropna(how="all").empty:
                                            st.caption("DPS 데이터가 없습니다.")
                                        else:
                                            st.line_chart(chart_src[["DPS"]], use_container_width=True)

                                st.markdown("**펀더멘탈 히스토리**")
                                funda_hist_view = slice_hist(funda_hist[["EPS", "BPS", "PER", "PBR", "DIV", "DPS", "ROE(%)"]]).reset_index()
                                funda_hist_fmt = {
                                    "EPS": "{:,.2f}",
                                    "BPS": "{:,.2f}",
                                    "PER": "{:,.2f}",
                                    "PBR": "{:,.2f}",
                                    "DIV": "{:,.2f}",
                                    "DPS": "{:,.2f}",
                                    "ROE(%)": "{:,.2f}",
                                }
                                st.dataframe(funda_hist_view.style.format(funda_hist_fmt), use_container_width=True)

                        with tab_foreign:
                            foreign_hist = load_foreign_history(t, f_start_date, f_end_date)
                            if foreign_hist is None or foreign_hist.empty:
                                st.caption("외인 보유 데이터를 가져오지 못했습니다.")
                            else:
                                foreign_hist = foreign_hist.copy()
                                latest = foreign_hist.iloc[-1]

                                f1, f2 = st.columns(2)
                                if "지분율" in foreign_hist.columns:
                                    f1.metric("외인 지분율(%)", f"{latest.get('지분율', 0):.2f}")
                                else:
                                    f1.metric("외인 지분율(%)", "-")

                                if "보유수량" in foreign_hist.columns:
                                    f2.metric("외인 보유수량", f"{latest.get('보유수량', 0):,.0f}")
                                else:
                                    f2.metric("외인 보유수량", "-")

                                left_cols = [c for c in ["지분율"] if c in foreign_hist.columns]
                                right_cols = [c for c in ["보유수량"] if c in foreign_hist.columns]

                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**지분율**")
                                    if left_cols:
                                        st.line_chart(foreign_hist[left_cols], use_container_width=True)
                                    else:
                                        st.caption("표시할 컬럼이 없습니다.")
                                with c2:
                                    st.markdown("**보유수량**")
                                    if right_cols:
                                        st.line_chart(foreign_hist[right_cols], use_container_width=True)
                                    else:
                                        st.caption("표시할 컬럼이 없습니다.")

                                hist_cols = [c for c in ["지분율", "보유수량", "상장주식수"] if c in foreign_hist.columns]
                                if hist_cols:
                                    st.markdown("**외인 히스토리**")
                                    foreign_view = slice_hist(foreign_hist[hist_cols]).reset_index()
                                    foreign_fmt = {}
                                    if "지분율" in foreign_view.columns:
                                        foreign_fmt["지분율"] = "{:,.2f}"
                                    if "보유수량" in foreign_view.columns:
                                        foreign_fmt["보유수량"] = "{:,.0f}"
                                    if "상장주식수" in foreign_view.columns:
                                        foreign_fmt["상장주식수"] = "{:,.0f}"
                                    st.dataframe(foreign_view.style.format(foreign_fmt), use_container_width=True)

                        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("모멘텀 대시보드")

    current_data = st.session_state['cached_data']

    st.write("##### 🔄 데이터 갱신 (섹터별 개별 실행)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    with c1:
        btn_etf = st.button("🇰🇷 ETF 갱신", use_container_width=True)
    with c2:
        btn_stock = st.button("🇰🇷 개별주 갱신", use_container_width=True)
    with c3:
        btn_us = st.button("🇺🇸 미국주식 갱신", use_container_width=True)
    with c4:
        ts = current_data.get('last_update', '-')
        st.info(f"🕒 마지막 저장: {ts}")

    target_sector = None
    if btn_etf: target_sector = 'etf'
    elif btn_stock: target_sector = 'stock'
    elif btn_us: target_sector = 'us'

    if target_sector:
        with st.spinner(f"[{target_sector.upper()}] 데이터를 수집 및 분석 중입니다..."):
            if target_sector == 'etf':
                new_part = calculate_etf_data()
            elif target_sector == 'stock':
                new_part = calculate_stock_data()
            else:
                new_part = calculate_us_data()

            if new_part:
                current_data[target_sector] = new_part
                kst = timezone(timedelta(hours=9))
                current_data['last_update'] = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')

                if save_momentum_data_to_disk(current_data):
                    st.session_state['cached_data'] = current_data
                    st.success(f"✅ {target_sector.upper()} 데이터 갱신 완료!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("데이터 수집 실패. 잠시 후 다시 시도해주세요.")

    st.divider()

    col_left, col_right = st.columns([0.85, 1.15])

    with col_left:
        st.subheader("모멘텀 신호")

        if current_data.get('etf'):
            render_left_card("🇰🇷 한국 ETF", current_data['etf'], 'etf')
        else:
            st.warning("🇰🇷 ETF 데이터가 없습니다. 위의 [ETF 갱신] 버튼을 눌러주세요.")

        if current_data.get('stock'):
            render_left_card("🇰🇷 한국 개별주", current_data['stock'], 'stock')
        else:
            st.warning("🇰🇷 개별주 데이터가 없습니다. 위의 [개별주 갱신] 버튼을 눌러주세요.")

        if current_data.get('us'):
            render_left_card("🇺🇸 미국 주식", current_data['us'], 'us')
        else:
            st.warning("🇺🇸 미국 주식 데이터가 없습니다. 위의 [미국주식 갱신] 버튼을 눌러주세요.")

    with col_right:
        st.subheader("종목 분석")
        with st.container(border=True):
            st.markdown("##### 📊 차트 분석 + ⚖️ 손익비")

            default_ticker = st.session_state.get('ticker_for_rr', '005930')
            default_price = st.session_state.get('price_for_rr', 0.0)
            if default_ticker == "N/A":
                default_ticker = ""

            c1, c2 = st.columns(2)
            ticker = c1.text_input("종목코드", value=default_ticker).strip().upper()
            entry_price = c2.number_input("매수단가 (0=현재가)", value=default_price)

            history_rows_m = st.selectbox("차트 데이터 표시 개수", options=[60, 120, 240, 500, "ALL"], index=1, key="momentum_history_rows")
            st.checkbox(
                "봉차트 표시",
                value=st.session_state['use_candlestick'],
                key="use_candlestick_momentum",
                on_change=sync_use_candlestick,
                args=("use_candlestick_momentum",),
            )
            st.checkbox(
                "손익비 라인 표시",
                value=st.session_state['show_rr_lines'],
                key="show_rr_lines_momentum",
                on_change=sync_show_rr_lines,
                args=("show_rr_lines_momentum",),
            )

            run_btn = st.button("분석 실행", use_container_width=True)

            should_run = run_btn
            if not run_btn and ticker and ticker != "N/A":
                if ticker == st.session_state.get('ticker_for_rr'):
                    should_run = True

            if should_run and ticker:
                try:
                    calc = UniversalRiskRewardCalculator()
                    res, rr_data = calc.analyze(ticker, entry_price)

                    if res is not None and rr_data is not None:
                        is_kr_stock = ticker.isdigit()
                        df_disp = res.copy()

                        if is_kr_stock:
                            df_disp["Target"] = df_disp["Target"].apply(lambda x: f"{int(x):,}원")
                            df_disp["Stop"] = df_disp["Stop"].apply(lambda x: f"{int(x):,}원")
                        else:
                            df_disp["Target"] = df_disp["Target"].apply(lambda x: f"${x:,.2f}")
                            df_disp["Stop"] = df_disp["Stop"].apply(lambda x: f"${x:,.2f}")

                        st.dataframe(
                            df_disp,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Mode": "전략",
                                "Target": "익절가",
                                "Stop": "손절가",
                                "R/R": "손익비",
                                "Risk": "예상손실"
                            }
                        )

                        st.markdown("---")

                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')

                        df_daily = load_price_data(ticker, start_date, end_date)
                        def slice_hist_m(df):
                            if history_rows_m == "ALL":
                                return df
                            return df.tail(int(history_rows_m))

                        if not df_daily.empty:
                            tab_daily, tab_weekly, tab_monthly, tab_sr = st.tabs(["일차트", "주차트", "월차트", "지지/예측"])

                            with tab_daily:
                                if len(df_daily) < 80:
                                    st.warning("데이터가 부족하여 지표 정확도가 떨어질 수 있습니다.")
                                view = slice_hist_m(df_daily)
                                st.plotly_chart(
                                    plot_dynamic_ichimoku_rsi(
                                        view,
                                        f"[{ticker}] 일차트",
                                        entry_price if entry_price else None,
                                        rr_data,
                                        plot_candlestick=st.session_state['use_candlestick'],
                                        show_rr=st.session_state['show_rr_lines'],
                                    ),
                                    use_container_width=True
                                )
                                ohlc_view = view[["Open", "High", "Low", "Close", "Volume"]].reset_index()
                                ohlc_fmt = {
                                    "Open": "{:,.2f}",
                                    "High": "{:,.2f}",
                                    "Low": "{:,.2f}",
                                    "Close": "{:,.2f}",
                                    "Volume": "{:,.0f}",
                                }
                                st.dataframe(ohlc_view.style.format(ohlc_fmt), use_container_width=True)
                                with st.expander("기술지표(일목/RSI)"):
                                    fig = plot_ichimoku_rsi(df_daily, f"[{ticker}] 일차트 + 손익비", rr_data, show_rr=st.session_state['show_rr_lines'])
                                    st.pyplot(fig)

                            with tab_weekly:
                                df_weekly = resample_ohlc(df_daily, 'W-FRI')
                                if len(df_weekly) < 80:
                                    st.warning("데이터가 부족하여 지표 정확도가 떨어질 수 있습니다.")
                                view = slice_hist_m(df_weekly)
                                st.plotly_chart(
                                    plot_dynamic_ichimoku_rsi(
                                        view,
                                        f"[{ticker}] 주차트",
                                        entry_price if entry_price else None,
                                        rr_data,
                                        plot_candlestick=st.session_state['use_candlestick'],
                                        show_rr=st.session_state['show_rr_lines'],
                                    ),
                                    use_container_width=True
                                )
                                ohlc_view = view[["Open", "High", "Low", "Close", "Volume"]].reset_index()
                                ohlc_fmt = {
                                    "Open": "{:,.2f}",
                                    "High": "{:,.2f}",
                                    "Low": "{:,.2f}",
                                    "Close": "{:,.2f}",
                                    "Volume": "{:,.0f}",
                                }
                                st.dataframe(ohlc_view.style.format(ohlc_fmt), use_container_width=True)
                                with st.expander("기술지표(일목/RSI)"):
                                    fig = plot_ichimoku_rsi(df_weekly, f"[{ticker}] 주차트 + 손익비", rr_data, show_rr=st.session_state['show_rr_lines'])
                                    st.pyplot(fig)

                            with tab_monthly:
                                df_monthly = resample_ohlc(df_daily, 'M')
                                if len(df_monthly) < 80:
                                    st.warning("데이터가 부족하여 지표 정확도가 떨어질 수 있습니다.")
                                view = slice_hist_m(df_monthly)
                                st.plotly_chart(
                                    plot_dynamic_ichimoku_rsi(
                                        view,
                                        f"[{ticker}] 월차트",
                                        entry_price if entry_price else None,
                                        rr_data,
                                        plot_candlestick=st.session_state['use_candlestick'],
                                        show_rr=st.session_state['show_rr_lines'],
                                    ),
                                    use_container_width=True
                                )
                                ohlc_view = view[["Open", "High", "Low", "Close", "Volume"]].reset_index()
                                ohlc_fmt = {
                                    "Open": "{:,.2f}",
                                    "High": "{:,.2f}",
                                    "Low": "{:,.2f}",
                                    "Close": "{:,.2f}",
                                    "Volume": "{:,.0f}",
                                }
                                st.dataframe(ohlc_view.style.format(ohlc_fmt), use_container_width=True)
                                with st.expander("기술지표(일목/RSI)"):
                                    fig = plot_ichimoku_rsi(df_monthly, f"[{ticker}] 월차트 + 손익비", rr_data, show_rr=st.session_state['show_rr_lines'])
                                    st.pyplot(fig)

                            with tab_sr:
                                applied_key = f"sr_order_applied_m_{ticker}"
                                if applied_key not in st.session_state:
                                    st.session_state[applied_key] = 5

                                with st.form(key=f"sr_form_m_{ticker}"):
                                    order_input = st.slider(
                                        "지지/저항 민감도",
                                        min_value=5,
                                        max_value=60,
                                        value=int(st.session_state[applied_key]),
                                        step=5,
                                        key=f"sr_order_input_m_{ticker}",
                                    )
                                    apply_order = st.form_submit_button("민감도 적용")

                                if apply_order:
                                    st.session_state[applied_key] = order_input

                                # 지지/저항 차트
                                fig_sr, sup, res = plot_support_resistance(
                                    df_daily,
                                    order=int(st.session_state[applied_key]),
                                    title=f"[{ticker}] 지지/저항",
                                )
                                s1, s2, s3 = st.columns(3)
                                s1.metric("현재가", f"{float(df_daily['Close'].iloc[-1]):,.2f}" if not ticker.isdigit() else f"{float(df_daily['Close'].iloc[-1]):,.0f}원")
                                s2.metric("지지선", f"{float(sup):,.2f}" if not ticker.isdigit() else f"{float(sup):,.0f}원")
                                s3.metric("저항선", f"{float(res):,.2f}" if not ticker.isdigit() else f"{float(res):,.0f}원")
                                st.plotly_chart(fig_sr, use_container_width=True)
                                
                                st.divider()
                                st.markdown("**📈 AI 예측 모델 (30일)**")

                                ai_cache_key = f"ai_forecast_cache_m_{ticker}"
                                ai_sig = (
                                    len(df_daily),
                                    str(df_daily.index.max()),
                                    float(df_daily['Close'].iloc[-1])
                                )

                                if (
                                    ai_cache_key not in st.session_state
                                    or st.session_state[ai_cache_key].get("sig") != ai_sig
                                ):
                                    try:
                                        with st.spinner("AI 모델 계산 중..."):
                                            forecast_prophet = compute_prophet_forecast(df_daily, periods=30)
                                            forecast_np = compute_neuralprophet_forecast(df_daily, periods=5)
                                            forecast_xgb = compute_xgboost_forecast(df_daily, periods=5)
                                        st.session_state[ai_cache_key] = {
                                            "sig": ai_sig,
                                            "prophet": forecast_prophet,
                                            "neural": forecast_np,
                                            "xgboost": forecast_xgb,
                                        }
                                    except Exception as e:
                                        st.error(f"예측 실패: {e}")
                                        st.session_state[ai_cache_key] = None

                                if ai_cache_key in st.session_state and st.session_state[ai_cache_key]:
                                    cached = st.session_state[ai_cache_key]

                                    # 3개 모델을 2행으로 배치 (Prophet | NeuralProphet / XGBoost | 빈공간)
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("**Prophet**")
                                        try:
                                            fig_pf = build_forecast_chart(df_daily, cached["prophet"], title=f"[{ticker}] Prophet")
                                            st.plotly_chart(fig_pf, use_container_width=True)
                                            last = cached["prophet"].iloc[-1]
                                            st.caption(f"예측: {last['yhat']:.2f} / 하단: {last.get('yhat_lower', 0):.2f} / 상단: {last.get('yhat_upper', 0):.2f}")
                                        except Exception as e:
                                            st.error(f"예측 실패: {e}")

                                    with col2:
                                        st.markdown("**NeuralProphet**")
                                        try:
                                            fig_np = build_forecast_chart(df_daily, cached["neural"], title=f"[{ticker}] NeuralProphet")
                                            st.plotly_chart(fig_np, use_container_width=True)
                                            last_np = cached["neural"].iloc[-1]
                                            st.caption(f"예측: {last_np['yhat']:.2f}")
                                        except Exception as e:
                                            st.error(f"예측 실패: {e}")

                                    col3, col4 = st.columns(2)

                                    with col3:
                                        st.markdown("**XGBoost (상승확률)**")
                                        try:
                                            for idx, row in cached["xgboost"].iterrows():
                                                date_str = row['ds'].strftime('%m/%d')
                                                prob = row['probability']
                                                color = "green" if prob > 0.5 else "red"
                                                st.markdown(f"{date_str}: <span style='color:{color};font-weight:bold'>{prob*100:.1f}%</span> 상승", unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"예측 실패: {e}")
                    else:
                        st.error("데이터를 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"오류: {e}")
            elif not ticker:
                st.caption("왼쪽 리스트에서 종목을 선택하거나 코드를 입력하세요.")
# streamlit run auto_bot/dashboard_local/app.py
