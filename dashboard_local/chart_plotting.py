# chart_plotting.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import platform
import os
import torch
from datetime import timedelta
from scipy.signal import argrelextrema

from technical_indicators import calculate_ichimoku, calculate_rsi # noqa: E402
from dashboard_config import CANDLE_UP_COLOR, CANDLE_DOWN_COLOR # noqa: E402

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

def plot_dynamic_ichimoku_rsi(df, title, entry_price=None, rr_data=None, plot_candlestick=False, show_rr=True, visible_tail_rows=None, show_bb=False):
    # 1. Future Cloud를 위해 데이터프레임 확장 (26일)
    last_date = df.index[-1]
    # 인덱스가 DatetimeIndex인 경우와 아닌 경우 처리
    if isinstance(last_date, pd.Timestamp):
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=26, freq='B')
        extension_index = future_dates
    else:
        # 날짜 인덱스가 아니면 정수 인덱스 확장 등으로 처리 (일단 날짜라고 가정)
        # fallback for integer index
        start_idx = df.index[-1] + 1 if isinstance(df.index[-1], int) else len(df)
        extension_index = pd.RangeIndex(start=start_idx, stop=start_idx + 26)

    # 빈 행이 추가된 확장 DF 생성
    future_df = pd.DataFrame(index=extension_index, columns=df.columns)
    df_extended = pd.concat([df, future_df])

    # 2. 지표 계산 (확장된 DF 기반)
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df_extended)
    rsi = calculate_rsi(df_extended['Close'])

    # 볼린저 밴드 계산
    ma20 = df_extended['Close'].rolling(20).mean()
    std20 = df_extended['Close'].rolling(20).std()
    bb_upper = ma20 + (std20 * 2)
    bb_lower = ma20 - (std20 * 2)

    # 3. 보여줄 범위 슬라이싱 (Tail)
    # visible_tail_rows가 지정되면 과거 데이터는 자르되, 미래 확장분(26일)은 포함해야 함
    if visible_tail_rows and visible_tail_rows != "ALL":
        try:
            rows = int(visible_tail_rows)
            # 원본 데이터에서의 시작 위치 계산
            start_pos = max(0, len(df) - rows)
            # 슬라이싱: (원본의 마지막 rows개) + (확장된 26개)
            # df_extended의 길이는 len(df) + 26
            # 우리가 원하는건 index[start_pos:]
            df_disp = df_extended.iloc[start_pos:]
            tenkan = tenkan.iloc[start_pos:]
            kijun = kijun.iloc[start_pos:]
            span_a = span_a.iloc[start_pos:]
            span_b = span_b.iloc[start_pos:]
            chikou = chikou.iloc[start_pos:]
            rsi = rsi.iloc[start_pos:]
            bb_upper = bb_upper.iloc[start_pos:]
            bb_lower = bb_lower.iloc[start_pos:]
            ma20 = ma20.iloc[start_pos:]
        except ValueError:
            df_disp = df_extended
    else:
        df_disp = df_extended

    # 4. Plotly 차트 생성 (3단: Price / Volume / RSI)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25],
        subplot_titles=(title, "Volume", "RSI")
    )

    # --- Row 1: Price & Ichimoku & BB ---
    # 캔들/라인
    if plot_candlestick:
        fig.add_trace(go.Candlestick(x=df_disp.index,
                                     open=df_disp['Open'],
                                     high=df_disp['High'],
                                     low=df_disp['Low'],
                                     close=df_disp['Close'],
                                     increasing=dict(line=dict(color=CANDLE_UP_COLOR, width=1), fillcolor=CANDLE_UP_COLOR),
                                     decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR, width=1), fillcolor=CANDLE_DOWN_COLOR),
                                     name='Candlestick'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df_disp.index, y=df_disp['Close'], name='Close', line=dict(color='#e2e8f0', width=2)), row=1, col=1)

    # 볼린저 밴드 (옵션)
    if show_bb:
        fig.add_trace(go.Scatter(x=bb_upper.index, y=bb_upper, name='BB Upper', line=dict(color='#6366f1', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ma20.index, y=ma20, name='BB Mid', line=dict(color='#6366f1', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_lower.index, y=bb_lower, name='BB Lower', line=dict(color='#6366f1', width=1, dash='dot')), row=1, col=1)

    # 일목균형표
    fig.add_trace(go.Scatter(x=tenkan.index, y=tenkan, name='Tenkan', line=dict(color='#f59e0b', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kijun.index, y=kijun, name='Kijun', line=dict(color='#3b82f6', width=1)), row=1, col=1)
    
    # 구름대 (Span A, Span B)
    # Span A, B는 미래 영역까지 값이 있음. fill='tonexty'를 위해 순서 중요.
    # 보통 Span A를 먼저 그리고 Span B를 그릴 때 fill='tonexty'하면 두 선 사이가 채워짐.
    fig.add_trace(go.Scatter(x=span_a.index, y=span_a, name='Span A', line=dict(color='#10b981', width=1), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=span_b.index, 
        y=span_b, 
        name='Span B', 
        line=dict(color='#ef4444', width=1), 
        fill='tonexty', 
        fillcolor='rgba(128, 128, 128, 0.1)', # 회색 반투명 채우기 (상승/하락 구분은 복잡하므로 단순화)
        showlegend=True
    ), row=1, col=1)
    
    # 후행스팬 (Chikou)
    fig.add_trace(go.Scatter(x=chikou.index, y=chikou, name='Chikou', line=dict(color='#6b7280', width=1)), row=1, col=1)

    # 진입가/손익비 라인
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

    # --- Row 2: Volume ---
    # 색상: 전일 대비 상승이면 빨강(한국기준) or Green, 하락이면 파랑 or Red. 
    # 여기선 단순화를 위해 회색이나 고정색, 혹은 캔들 색상 따라가기.
    # Close가 Open보다 크면 양봉(UP), 작으면 음봉(DOWN) 색상 사용
    colors_vol = [CANDLE_UP_COLOR if (o <= c) else CANDLE_DOWN_COLOR for o, c in zip(df_disp['Open'], df_disp['Close'])]
    # df_disp의 마지막 26개는 NaN이므로 색상 계산시 주의. NaN이면 기본색
    # zip은 짧은 쪽 기준이므로 NaN 데이터(Close/Open이 NaN)에 대해서는 루프가 돌지 않거나 에러 날 수 있음.
    # 안전하게 처리:
    vol_colors = []
    for i in range(len(df_disp)):
        if pd.isna(df_disp['Close'].iloc[i]) or pd.isna(df_disp['Open'].iloc[i]):
            vol_colors.append('#555555')
        else:
            if df_disp['Close'].iloc[i] >= df_disp['Open'].iloc[i]:
                vol_colors.append(CANDLE_UP_COLOR)
            else:
                vol_colors.append(CANDLE_DOWN_COLOR)
    
    fig.add_trace(go.Bar(
        x=df_disp.index, 
        y=df_disp['Volume'], 
        name='Volume', 
        marker_color=vol_colors,
        showlegend=False
    ), row=2, col=1)

    # --- Row 3: RSI ---
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='#8b5cf6', width=1)), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='#ef4444', width=1, dash='dash'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='#10b981', width=1, dash='dash'), row=3, col=1)

    # Layout 설정
    fig.update_layout(
        height=700, # 높이 증가
        showlegend=True, 
        legend_orientation='h', 
        legend_y=1.02, 
        legend_x=0,
        legend=dict(font=dict(color='#E0E0E0')),
        plot_bgcolor='#131722', 
        paper_bgcolor='#131722', 
        font=dict(color='#E0E0E0'),
        xaxis=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39', rangeslider=dict(visible=False)),
        xaxis2=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39'),
        xaxis3=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39'),
        yaxis=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39'),
        yaxis2=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39', title="Vol"),
        yaxis3=dict(showgrid=True, gridcolor='#2A2E39', zerolinecolor='#2A2E39', range=[0, 100]),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Remove range slider from all xaxes if present
    fig.update_xaxes(rangeslider_visible=False)

    return fig



def plot_support_resistance(df, order=20, title="Support/Resistance", plot_candlestick=False):
    close = df['Close'].values
    local_max_idx = argrelextrema(close, np.greater, order=order)[0]
    local_min_idx = argrelextrema(close, np.less, order=order)[0]

    local_max_prices = close[local_max_idx]
    local_min_prices = close[local_min_idx]

    current_price = close[-1]
    nearest_support = local_min_prices[local_min_prices < current_price].max() if any(local_min_prices < current_price) else current_price * 0.9
    nearest_resistance = local_max_prices[local_max_prices > current_price].min() if any(local_max_prices > current_price) else current_price * 1.1

    fig = go.Figure()
    if plot_candlestick:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing=dict(line=dict(color=CANDLE_UP_COLOR, width=1), fillcolor=CANDLE_UP_COLOR),
            decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR, width=1), fillcolor=CANDLE_DOWN_COLOR),
            name='Candlestick'
        ))
    else:
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
    from prophet import Prophet
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


def build_forecast_chart(df, forecast, title="Prophet Forecast", plot_candlestick=False):
    fig = go.Figure()
    if plot_candlestick:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing=dict(line=dict(color=CANDLE_UP_COLOR, width=1), fillcolor=CANDLE_UP_COLOR),
            decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR, width=1), fillcolor=CANDLE_DOWN_COLOR),
            name='Candlestick'
        ))
    else:
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
    from datetime import timedelta
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    result = pd.DataFrame({
        'ds': future_dates,
        'probability': predictions  # 상승 확률
    })
    
    return result
