import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import time
import pickle
import os
import logging

# [중요] 페이지 설정을 최상단으로 이동 (반드시 다른 st.명령어보다 먼저 나와야 함)
st.set_page_config(layout="wide", page_title="모멘텀 봇 대시보드", page_icon="📈")

# [설정] 스레드 컨텍스트 경고 메시지 차단 (기능에는 영향 없음)
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# 기존 프로젝트의 공통 모듈 및 설정을 가져옵니다.
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common import fetch_data_in_parallel
import config as cfg

# ----------------------------------------------------------------------
# [설정] 파일 저장 경로 및 스타일
# ----------------------------------------------------------------------
DATA_FILE = "dashboard_data.pkl"  # 데이터를 저장할 파일명

# [수정됨] 폰트 설정 로직 (파일 우선 로드)
def init_font():
    font_filename = "NanumGothic.ttf" 
    
    # 현재 파일(app.py)과 같은 폴더에서 폰트 파일을 찾음
    font_path = os.path.join(os.path.dirname(__file__), font_filename)
    
    if os.path.exists(font_path):
        # 폰트 파일이 있으면 로드 (서버/배포 환경용)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        # 폰트 파일이 없으면 시스템 폰트 사용 (로컬 환경용)
        if platform.system() == 'Darwin': 
            plt.rc('font', family='AppleGothic')
        else: 
            plt.rc('font', family='Nanum Gothic')
            
    plt.rcParams['axes.unicode_minus'] = False

init_font() # 폰트 설정 실행
plt.style.use('ggplot')

# 스타일 설정 (set_page_config 이후에 실행되어야 안전함)
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
        div[data-testid="stVerticalBlock"] > div {gap: 0.2rem;}
        .stButton button {height: 2em; padding-top: 0; padding-bottom: 0;}
        .element-container {margin-bottom: 0.2rem !important;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# [유틸리티] 파일 입출력 (디스크 저장/로드)
# ----------------------------------------------------------------------
def save_data_to_disk(data):
    """데이터를 파일로 저장 (영구 보존)"""
    try:
        with open(DATA_FILE, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"데이터 저장 실패: {e}")
        return False

def load_data_from_disk():
    """파일에서 데이터 불러오기"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# ----------------------------------------------------------------------
# [유틸리티] 차트 지표 계산
# ----------------------------------------------------------------------
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

@st.cache_data(ttl=60 * 60)
def load_price_data(ticker, start_date, end_date):
    return fdr.DataReader(ticker, start=start_date, end=end_date)

def plot_ichimoku_rsi(df, title, rr_data=None):
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

    # 손익비 라인 추가
    if rr_data:
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

# ----------------------------------------------------------------------
# [유틸리티] UI 컴포넌트 & 상태 관리
# ----------------------------------------------------------------------
def set_analysis_target(ticker, price):
    st.session_state['ticker_for_rr'] = ticker
    st.session_state['price_for_rr'] = float(price)

def ui_card_header(title, status, reason):
    color = "red" if "상승" in status else "orange" if "중립" in status else "blue"
    icon = "🔴" if "상승" in status else "🟠" if "중립" in status else "🔵"
    c1, c2 = st.columns([1.5, 1])
    with c1: st.markdown(f"**{title}**")
    with c2: st.markdown(f"{icon} :{color}[**{status}**] <span style='font-size:0.8em; color:gray'>({reason})</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0.3rem 0;'>", unsafe_allow_html=True)

def ui_target_row(rank, name, code, weight, price, is_us=False):
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
            st.button("🔍", key=f"btn_{code}_{rank}_{int(time.time())}", 
                     help="오른쪽 화면에서 분석", 
                     on_click=set_analysis_target, args=(code, price))

def ui_ranking_list(rank_data, is_us=False, limit=50):
    c1, c2, c3, c4, c5 = st.columns([0.7, 2.5, 1.2, 1.5, 1.2])
    c1.caption("No.")
    c2.caption("종목명")
    c3.caption("점수")
    c4.caption("현재가")
    c5.caption("분석")
    st.markdown("<hr style='margin: 0.1rem 0;'>", unsafe_allow_html=True)

    for item in rank_data[:limit]:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([0.7, 2.5, 1.2, 1.5, 1.2])
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
                    st.button(f"{code_label}", key=f"rk_btn_{item['code']}_{item['rank']}_{int(time.time())}", 
                              on_click=set_analysis_target, args=(item['code'], item['price']), use_container_width=True)
                else: st.caption("-")
        st.markdown("<hr style='margin: 0.1rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# [로직] 데이터 계산
# ----------------------------------------------------------------------
def calculate_etf_data():
    etf_tickers = cfg.ETF_TICKERS
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        market_df = fdr.DataReader(cfg.ETF_MARKET_INDEX, start=start_date, end=end_date)
        market_index = market_df['Close'].ffill()
        raw_data = fetch_data_in_parallel(etf_tickers, start_date, end_date)
        if raw_data.empty: return None
    except: return None

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
    except: return None

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        market_df = fdr.DataReader(cfg.STOCK_MARKET_INDEX, start=start_date, end=end_date)
        raw_data = fetch_data_in_parallel(tickers, start_date, end_date)
        valid_cols = [c for c in raw_data.columns if raw_data[c].count() >= 120]
        raw_data = raw_data[valid_cols]
    except: return None

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
        # [수정] S&P 500 전종목 + 나스닥 100 조합 (약 530~550개) - 우량주 누락 방지
        df_sp = fdr.StockListing('S&P500')
        sp500_tickers = set(df_sp['Symbol'].tolist())
        
        df_nasdaq = fdr.StockListing('NASDAQ')
        nasdaq100_tickers = set(df_nasdaq.head(100)['Symbol'].tolist())
        
        combined_tickers = sp500_tickers.union(nasdaq100_tickers)
        
        tickers = {t: t for t in combined_tickers}
        tickers[cfg.US_DEFENSE_ASSET] = cfg.US_DEFENSE_ASSET
    except: return None

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        market_df = fdr.DataReader(cfg.US_MARKET_INDEX, start=start_date, end=end_date)
        raw_data = fetch_data_in_parallel(tickers, start_date, end_date)
    except: return None

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

# ----------------------------------------------------------------------
# [렌더링] 왼쪽 컬럼 카드
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# [로직] 손익비 분석기
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# [메인] 대시보드 구조 (개별 섹터 갱신 기능 적용)
# ----------------------------------------------------------------------
def main():
    # st.set_page_config는 여기서 제거하고 파일 최상단으로 이동했습니다.
    
    st.title("📈 모멘텀 봇 대시보드")
    
    # [1] 초기 데이터 로드 (파일 -> 메모리)
    if 'cached_data' not in st.session_state:
        loaded_data = load_data_from_disk()
        if loaded_data:
            st.session_state['cached_data'] = loaded_data
            last_update = loaded_data.get('last_update', '알 수 없음')
            st.toast(f"📂 저장된 데이터를 불러왔습니다. (Last Save: {last_update})")
        else:
            # 파일이 없으면 빈 껍데기 생성
            st.session_state['cached_data'] = {'etf': None, 'stock': None, 'us': None, 'last_update': '-'}

    # 현재 메모리에 있는 데이터 가져오기
    current_data = st.session_state['cached_data']

    # [2] 상단 컨트롤 패널 (3분할 버튼)
    st.write("##### 🔄 데이터 갱신 (섹터별 개별 실행)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    
    with c1:
        btn_etf = st.button("🇰🇷 ETF 갱신", use_container_width=True)
    with c2:
        btn_stock = st.button("🇰🇷 개별주 갱신", use_container_width=True)
    with c3:
        btn_us = st.button("🇺🇸 미국주식 갱신", use_container_width=True)
    with c4:
        # 마지막 업데이트 시간 표시
        ts = current_data.get('last_update', '-')
        st.info(f"🕒 마지막 저장: {ts}")

    # [3] 갱신 로직 (선택된 섹터만 계산 후 합치기)
    target_sector = None
    
    if btn_etf: target_sector = 'etf'
    elif btn_stock: target_sector = 'stock'
    elif btn_us: target_sector = 'us'

    if target_sector:
        with st.spinner(f"[{target_sector.upper()}] 데이터를 수집 및 분석 중입니다..."):
            
            # 1. 해당 섹터만 새로 계산
            if target_sector == 'etf':
                new_part = calculate_etf_data()
            elif target_sector == 'stock':
                new_part = calculate_stock_data()
            elif target_sector == 'us':
                new_part = calculate_us_data()
            
            # 2. 기존 데이터에 덮어쓰기 (Merge)
            if new_part:
                current_data[target_sector] = new_part
                current_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 3. 파일 저장
                if save_data_to_disk(current_data):
                    st.session_state['cached_data'] = current_data
                    st.success(f"✅ {target_sector.upper()} 데이터 갱신 완료!")
                    time.sleep(1)
                    st.rerun() # 화면 새로고침
            else:
                st.error("데이터 수집 실패. 잠시 후 다시 시도해주세요.")

    st.divider()

    # --- 메인 레이아웃 (좌우 2단 분할) ---
    col_left, col_right = st.columns([0.85, 1.15])

    # [왼쪽] 모멘텀 신호 카드 스택
    with col_left:
        st.subheader("모멘텀 신호")
        
        # 데이터가 있으면 그리고, 없으면 안내 문구
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

    # [오른쪽] 통합 차트 분석 (손익비 + 일목균형표 + RSI)
    with col_right:
        st.subheader("종목 분석")
        with st.container(border=True):
            st.markdown("##### 📊 차트 분석 + ⚖️ 손익비")
            
            default_ticker = st.session_state.get('ticker_for_rr', '005930')
            default_price = st.session_state.get('price_for_rr', 0.0)
            if default_ticker == "N/A": default_ticker = ""

            c1, c2 = st.columns(2)
            ticker = c1.text_input("종목코드", value=default_ticker).strip().upper()
            entry_price = c2.number_input("매수단가 (0=현재가)", value=default_price)
            
            run_btn = st.button("분석 실행", use_container_width=True)
            
            should_run = run_btn
            if not run_btn and ticker and ticker != "N/A":
                if ticker == st.session_state.get('ticker_for_rr'):
                    should_run = True
            
            if should_run and ticker:
                try:
                    # 손익비 계산
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
                        
                        # 차트 분석 (손익비 라인 포함)
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
                        
                        df_daily = load_price_data(ticker, start_date, end_date)
                        if not df_daily.empty:
                            tab_daily, tab_weekly, tab_monthly = st.tabs(["일차트", "주차트", "월차트"])

                            with tab_daily:
                                if len(df_daily) < 80:
                                    st.warning("데이터가 부족하여 지표 정확도가 떨어질 수 있습니다.")
                                fig = plot_ichimoku_rsi(df_daily, f"[{ticker}] 일차트 + 손익비", rr_data)
                                st.pyplot(fig)

                            with tab_weekly:
                                df_weekly = resample_ohlc(df_daily, 'W-FRI')
                                if len(df_weekly) < 80:
                                    st.warning("데이터가 부족하여 지표 정확도가 떨어질 수 있습니다.")
                                fig = plot_ichimoku_rsi(df_weekly, f"[{ticker}] 주차트 + 손익비", rr_data)
                                st.pyplot(fig)

                            with tab_monthly:
                                df_monthly = resample_ohlc(df_daily, 'M')
                                if len(df_monthly) < 80:
                                    st.warning("데이터가 부족하여 지표 정확도가 떨어질 수 있습니다.")
                                fig = plot_ichimoku_rsi(df_monthly, f"[{ticker}] 월차트 + 손익비", rr_data)
                                st.pyplot(fig)
                    else:
                        st.error("데이터를 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"오류: {e}")
            elif not ticker:
                st.caption("왼쪽 리스트에서 돋보기(🔍)를 누르거나 코드를 입력하세요.")

    st.divider()
    st.caption("이 페이지에는 네이버에서 제공한 나눔 고딕 글꼴이 적용되어 있습니다.")

if __name__ == "__main__":
    main()
# streamlit run auto_bot/dashboard/app.py
