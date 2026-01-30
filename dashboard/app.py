import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import platform

# 기존 프로젝트의 공통 모듈 및 설정을 가져옵니다.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common import fetch_data_in_parallel
import config as cfg

# Matplotlib 폰트 및 스타일 설정
if platform.system() == 'Darwin': 
    plt.rc('font', family='AppleGothic')
else: 
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
# 차트 스타일을 좀 더 모던하게
plt.style.use('ggplot')

# ----------------------------------------------------------------------
# 유틸리티 함수: UI 컴포넌트 & 세션 관리
# ----------------------------------------------------------------------
def set_analysis_target(ticker, price):
    st.session_state['ticker_for_rr'] = ticker
    st.session_state['price_for_rr'] = float(price)
    st.toast(f"✅ [{ticker}] 분석 준비 완료! '손익비 분석' 탭으로 이동하세요.", icon="👉")

def ui_market_status(status, reason):
    """시장 상태를 예쁜 박스로 보여주는 UI 함수"""
    color = "red" if "상승" in status else "orange" if "중립" in status else "blue"
    icon = "🔴" if "상승" in status else "🟠" if "중립" in status else "🔵"
    
    with st.container():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"#### {icon} : {color}[{status}]")
        with c2:
            st.info(f"💡 전략: **{reason}**")

def ui_asset_row(idx, name, code, weight, current_price):
    """매수 종목 한 줄을 예쁘게 그려주는 UI 함수"""
    with st.container():
        c1, c2, c3, c4 = st.columns([3, 2, 1.5, 1])
        with c1:
            st.markdown(f"**{idx}. {name}**")
            st.caption(f"Code: {code}")
        with c2:
            # 비중을 Progress Bar로 시각화
            st.progress(weight)
            st.caption(f"비중 {weight*100:.0f}%")
        with c3:
            st.markdown(f"**{int(current_price):,}원**")
        with c4:
            if code and code != "N/A":
                st.button(
                    "🔍", 
                    key=f"btn_{code}_{idx}",
                    help="손익비 분석으로 이동",
                    on_click=set_analysis_target,
                    args=(code, current_price)
                )
            else:
                st.write("-")
    st.divider()

# ----------------------------------------------------------------------
# 1. 한국 ETF 봇 로직
# ----------------------------------------------------------------------
def generate_etf_signals():
    st.markdown("### 🇰🇷 한국 ETF 가중모멘텀")
    
    with st.spinner("데이터 분석 및 차트 그리는 중..."):
        etf_tickers = cfg.ETF_TICKERS
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            market_df = fdr.DataReader(cfg.ETF_MARKET_INDEX, start=start_date, end=end_date)
            market_index = market_df['Close'].ffill()
            raw_data = fetch_data_in_parallel(etf_tickers, start_date, end_date)
            if raw_data.empty: raise Exception("데이터 수집 실패")
        except Exception as e:
            st.error(f"데이터 수집 오류: {e}"); return

        # 알고리즘 계산 (기존 유지)
        w1, w2, w3 = cfg.MOMENTUM_WEIGHTS
        mom_1m = raw_data.pct_change(20).iloc[-1]
        mom_3m = raw_data.pct_change(60).iloc[-1]
        mom_6m = raw_data.pct_change(120).iloc[-1]
        weighted_score = (mom_1m.fillna(0) * w1) + (mom_3m.fillna(0) * w2) + (mom_6m.fillna(0) * w3)

        ma_series = market_index.rolling(window=60).mean()
        current_ma, prev_ma = ma_series.iloc[-1], ma_series.iloc[-6]
        current_market_index = market_index.iloc[-1]
        
        is_bull_market = current_market_index > current_ma
        is_neutral_market = not is_bull_market and (current_ma > prev_ma)
        market_status = "상승장" if is_bull_market else "중립장" if is_neutral_market else "하락장"

        final_targets, reason, all_rankings = [], "", []
        defense_asset = cfg.ETF_DEFENSE_ASSET
        scores = weighted_score.drop(defense_asset, errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        for rank, (name, score) in enumerate(top_assets.items(), 1):
            all_rankings.append({'rank': rank, 'name': name, 'score': round(score, 3), 'price': int(raw_data[name].iloc[-1])})

        selected = [name for name, score in top_assets.items() if score > 0][:2]
        
        if is_bull_market:
            reason = "적극 투자 (주식형 100%)"
            if not selected: final_targets = [(defense_asset, 1.0)]
            elif len(selected) == 1: final_targets = [(selected[0], 1.0)]
            else: final_targets = [(selected[0], 0.5), (selected[1], 0.5)]
        elif is_neutral_market:
            reason = "분산 투자 (주식형 50% + 채권 50%)"
            if not selected: final_targets = [(defense_asset, 1.0)]
            elif len(selected) == 1: final_targets = [(selected[0], 0.5), (defense_asset, 0.5)]
            else: final_targets = [(selected[0], 0.25), (selected[1], 0.25), (defense_asset, 0.5)]
        else:
            final_targets, reason = [(defense_asset, 1.0)], "보수적 운용 (현금성 100%)"

    # --- UI 렌더링 ---
    with st.container(border=True):
        ui_market_status(market_status, reason)
        
        st.write("##### 🎯 매수 추천 포트폴리오")
        if final_targets:
            for i, (name, weight) in enumerate(final_targets):
                # 코드 찾기 로직
                ticker_code = etf_tickers.get(name, "N/A")
                price = raw_data[name].iloc[-1] if name in raw_data.columns else 0
                
                ui_asset_row(i+1, name, ticker_code, weight, price)

        with st.expander("📊 전체 모멘텀 순위표"):
            st.dataframe(
                pd.DataFrame(all_rankings).set_index('rank'),
                column_config={
                    "score": st.column_config.NumberColumn("모멘텀 점수", format="%.3f"),
                    "price": st.column_config.NumberColumn("현재가", format="%d원")
                },
                use_container_width=True
            )

# ----------------------------------------------------------------------
# 2. 한국 개별주 봇 로직
# ----------------------------------------------------------------------
def generate_stock_signals():
    st.markdown("### 🇰🇷 한국 우량주 변동성조절")
    
    with st.spinner("KOSPI/KOSDAQ 데이터 스캔 중..."):
        try:
            df_kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSPI)
            df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSDAQ)
            target_tickers = {row['Name']: row['Code'] for _, row in pd.concat([df_kospi, df_kosdaq]).iterrows()}
            def_asset_code = cfg.ETF_TICKERS.get(cfg.STOCK_DEFENSE_ASSET, '261240')
            target_tickers[cfg.STOCK_DEFENSE_ASSET] = def_asset_code
        except Exception as e:
            st.error(f"리스트 확보 실패: {e}"); return

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            market_df = fdr.DataReader(cfg.STOCK_MARKET_INDEX, start=start_date, end=end_date)
            market_index = market_df['Close'].ffill()
            raw_data = fetch_data_in_parallel(target_tickers, start_date, end_date)
            valid_cols = [col for col in raw_data.columns if raw_data[col].count() >= 120]
            raw_data = raw_data[valid_cols]
            if raw_data.empty: raise Exception("데이터 부족")
        except Exception as e:
            st.error(f"데이터 다운로드 오류: {e}"); return

        # 알고리즘 (기존 유지)
        daily_rets = raw_data.pct_change()
        vol_3m = daily_rets.rolling(60).std().iloc[-1]
        weighted_score = ((raw_data.pct_change(60).iloc[-1] / (vol_3m + 1e-6)).fillna(0) * 0.5) + \
                         ((raw_data.pct_change(120).iloc[-1] / (vol_3m + 1e-6)).fillna(0) * 0.5)

        ma_series = market_index.rolling(window=60).mean()
        is_bull_market = market_index.iloc[-1] > ma_series.iloc[-1]
        is_neutral_market = not is_bull_market and (ma_series.iloc[-1] > ma_series.iloc[-6])
        market_status = "상승장" if is_bull_market else "중립장" if is_neutral_market else "하락장"

        final_targets, reason = [], ""
        defense_asset = cfg.STOCK_DEFENSE_ASSET
        scores = weighted_score.drop(defense_asset, errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        selected = [name for name, score in top_assets.items() if score > 0][:cfg.STOCK_TOP_N]

        if is_bull_market:
            reason = "적극 투자 (N빵)"
            if selected: final_targets = [(s, 1.0 / len(selected)) for s in selected]
            else: final_targets = [(defense_asset, 1.0)]
        elif is_neutral_market:
            reason = "주식 50% + 현금 50%"
            if selected:
                final_targets = [(s, 0.5 / len(selected)) for s in selected]
                final_targets.append((defense_asset, 0.5))
            else: final_targets = [(defense_asset, 1.0)]
        else:
            final_targets, reason = [(defense_asset, 1.0)], "전량 현금 방어"

    # --- UI 렌더링 ---
    with st.container(border=True):
        ui_market_status(market_status, reason)
        st.write("##### 🎯 매수 추천 포트폴리오")
        
        if final_targets:
            for i, (name, weight) in enumerate(final_targets):
                ticker_code = target_tickers.get(name, name)
                price = raw_data[name].iloc[-1] if name in raw_data.columns else 0
                ui_asset_row(i+1, name, ticker_code, weight, price)

# ----------------------------------------------------------------------
# 3. 미국 주식 봇 로직
# ----------------------------------------------------------------------
def generate_us_signals():
    st.markdown("### 🇺🇸 미국 주식 가중모멘텀")
    
    with st.spinner("S&P500 데이터 분석 중..."):
        try:
            df_sp500 = fdr.StockListing('S&P500').head(200)
            target_tickers = {row['Symbol']: row['Symbol'] for _, row in df_sp500.iterrows()}
            target_tickers[cfg.US_DEFENSE_ASSET] = cfg.US_DEFENSE_ASSET
        except: st.error("종목 리스트 에러"); return

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            market_df = fdr.DataReader(cfg.US_MARKET_INDEX, start=start_date, end=end_date)
            raw_data = fetch_data_in_parallel(target_tickers, start_date, end_date)
            if raw_data.empty: raise Exception("데이터 수집 실패")
        except Exception as e: st.error(f"다운로드 에러: {e}"); return

        # 알고리즘
        w1, w2, w3 = cfg.MOMENTUM_WEIGHTS
        weighted_score = ((raw_data.pct_change(20).iloc[-1].fillna(0) * w1) + 
                          (raw_data.pct_change(60).iloc[-1].fillna(0) * w2) + 
                          (raw_data.pct_change(120).iloc[-1].fillna(0) * w3))

        market_index = market_df['Close'].ffill()
        ma_series = market_index.rolling(window=60).mean()
        is_bull_market = market_index.iloc[-1] > ma_series.iloc[-1]
        is_neutral_market = not is_bull_market and (ma_series.iloc[-1] > ma_series.iloc[-6])
        market_status = "상승장" if is_bull_market else "중립장" if is_neutral_market else "하락장"

        final_targets, reason = [], ""
        defense_asset = cfg.US_DEFENSE_ASSET
        scores = weighted_score.drop(defense_asset, errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        selected = [name for name, score in top_assets.items() if score > 0][:cfg.US_TOP_N]

        if is_bull_market: reason = "적극 투자"; final_targets = [(s, 1.0/len(selected)) for s in selected] if selected else [(defense_asset, 1.0)]
        elif is_neutral_market: reason = "분산 투자"; final_targets = [(s, 0.5/len(selected)) for s in selected] + [(defense_asset, 0.5)] if selected else [(defense_asset, 1.0)]
        else: reason = "방어"; final_targets = [(defense_asset, 1.0)]

    # --- UI 렌더링 ---
    with st.container(border=True):
        ui_market_status(market_status, reason)
        st.write("##### 🎯 매수 추천 포트폴리오")
        if final_targets:
            for i, (name, weight) in enumerate(final_targets):
                price = raw_data[name].iloc[-1] if name in raw_data.columns else 0
                ui_asset_row(i+1, name, name, weight, price)

# ----------------------------------------------------------------------
# 4. 손익비 분석기
# ----------------------------------------------------------------------
class UniversalRiskRewardCalculator:
    def calculate_atr(self, df, period):
        tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def analyze(self, ticker, entry_price):
        # 
        df = fdr.DataReader(ticker, end=datetime.now().strftime('%Y-%m-%d'), start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        if df.empty: return None, None
        
        current_price = df['Close'].iloc[-1]
        if entry_price == 0: entry_price = current_price

        strategies = [
            {"name": "단기 (Scalping)", "atr_period": 14, "risk_mult": 1.5, "reward_ratio": 1.5, "style": ":", "alpha": 0.6},
            {"name": "스윙 (Swing)", "atr_period": 22, "risk_mult": 2.5, "reward_ratio": 2.0, "style": "--", "alpha": 0.8},
            {"name": "추세 (Trend)", "atr_period": 60, "risk_mult": 3.5, "reward_ratio": 3.0, "style": "-", "alpha": 1.0}
        ]
        
        results = []
        for s in strategies:
            atr_val = self.calculate_atr(df, s['atr_period']).iloc[-1]
            risk_width = atr_val * s['risk_mult']
            stop_loss = entry_price - risk_width
            take_profit = entry_price + (risk_width * s['reward_ratio'])
            results.append({
                "전략": s['name'],
                "익절가 (Target)": take_profit,
                "손절가 (Stop)": stop_loss,
                "손익비": f"1 : {s['reward_ratio']}",
                "예상손실폭": f"-{(entry_price-stop_loss)/entry_price*100:.1f}%"
            })
        
        fig = self.plot_chart(df.tail(120), ticker, entry_price, strategies, results)
        return pd.DataFrame(results), fig

    def plot_chart(self, plot_data, ticker, entry_price, strategies, results):
        # 차트 디자인 개선
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 캔들스틱 대신 단순 라인차트지만 예쁘게
        ax.plot(plot_data.index, plot_data['Close'], label='Close Price', color='#333333', linewidth=1.5, alpha=0.9)
        ax.axhline(y=entry_price, color='#2980b9', linestyle='-', linewidth=2, label=f'Entry: {entry_price:,.0f}')
        ax.fill_between(plot_data.index, plot_data['Close'], min(plot_data['Close']), color='#ecf0f1', alpha=0.3)

        # 전략별 라인
        colors = ['#27ae60', '#e67e22', '#c0392b'] # 초록, 주황, 빨강
        for i, strat in enumerate(strategies):
            tp = results[i]['익절가 (Target)']
            sl = results[i]['손절가 (Stop)']
            # 텍스트 라벨 대신 범례 활용
            ax.axhline(y=tp, color=colors[i], linestyle=strat['style'], alpha=0.8, linewidth=1, label=f"{strat['name']} TP")
            ax.axhline(y=sl, color=colors[i], linestyle=strat['style'], alpha=0.8, linewidth=1, label=f"{strat['name']} SL")

        # 추세 구간 강조
        trend_tp = results[2]['익절가 (Target)']
        trend_sl = results[2]['손절가 (Stop)']
        ax.axhspan(entry_price, trend_tp, color='#2ecc71', alpha=0.05) # 이익구간
        ax.axhspan(trend_sl, entry_price, color='#e74c3c', alpha=0.05) # 손실구간

        ax.set_title(f"📊 [{ticker}] Risk/Reward Analysis", fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig

def run_rr_analysis():
    # 상단 헤더 컨테이너
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### ⚖️ 만능 손익비 계산기")
            st.caption("진입가 기준 ATR 기반의 최적 익절/손절 라인을 계산합니다.")
        
        # 입력 폼
        default_ticker = st.session_state.get('ticker_for_rr', '005930')
        default_price = st.session_state.get('price_for_rr', 0.0)
        
        if default_ticker == "N/A": default_ticker = "005930"

        with st.form("rr_form"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                ticker = st.text_input("종목코드", value=default_ticker)
            with col2:
                entry_price = st.number_input("진입단가 (0=현재가)", value=default_price)
            with col3:
                st.write("") # 간격 맞추기용
                st.write("") 
                submit = st.form_submit_button("🚀 분석 실행", use_container_width=True)

    # 실행 로직
    should_run = submit
    if 'ticker_for_rr' in st.session_state and st.session_state['ticker_for_rr'] and st.session_state['ticker_for_rr'] != "N/A":
         # 세션에 값이 변경되어 리로드된 경우 자동 실행 조건 (Form 안이라 자동실행이 까다로울 수 있어, 세션 체크 추가)
         # 하지만 폼 제출 버튼이 UX상 깔끔하므로 버튼 클릭 위주로 하되, 탭 전환 직후를 위해 아래 로직 유지
         pass 
    
    # 세션 상태에 티커가 있으면 자동 실행 (Form 밖에서 처리)
    if not submit and 'ticker_for_rr' in st.session_state:
         ticker = st.session_state.get('ticker_for_rr', '005930')
         entry_price = st.session_state.get('price_for_rr', 0.0)
         if ticker and ticker != "N/A":
             should_run = True

    if should_run:
        if not ticker or ticker == "N/A":
            st.warning("종목 코드를 확인해주세요.")
            return

        calculator = UniversalRiskRewardCalculator()
        try:
            results_df, fig = calculator.analyze(ticker, entry_price)
            if results_df is None:
                st.error("데이터를 찾을 수 없습니다.")
            else:
                # 결과 UI
                st.markdown("#### 📋 전략별 가이드라인")
                st.dataframe(
                    results_df,
                    column_config={
                        "익절가 (Target)": st.column_config.NumberColumn(format="%.0f원"),
                        "손절가 (Stop)": st.column_config.NumberColumn(format="%.0f원"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                st.pyplot(fig)
        except Exception as e:
            st.error(f"오류 발생: {e}")

# ----------------------------------------------------------------------
# 대시보드 메인
# streamlit run auto_bot/dashboard/app.py
# ----------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="모멘텀 봇 대시보드", page_icon="📈")
    
    # 사이드바 (옵션)
    with st.sidebar:
        st.header("설정 및 정보")
        st.info("이 대시보드는 모멘텀 전략과 ATR 기반 손익비 분석을 제공합니다.")
        st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.title("📈 Momentum Trading Dashboard")
    
    if 'analysis_executed' not in st.session_state:
        st.session_state['analysis_executed'] = False

    tab1, tab2 = st.tabs(["🚀 모멘텀 시그널", "⚖️ 손익비 분석"])

    with tab1:
        # 상단 액션 버튼 영역
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔄 시그널 갱신", type="primary", use_container_width=True):
                st.session_state['analysis_executed'] = True
        
        st.divider()

        if st.session_state['analysis_executed']:
            # 3단 레이아웃 (ETF / 국장 / 미장)
            col_etf, col_kor, col_us = st.columns(3)
            
            with col_etf:
                generate_etf_signals()
            with col_kor:
                generate_stock_signals()
            with col_us:
                generate_us_signals()
        else:
            st.info("좌측 상단의 '시그널 갱신' 버튼을 눌러 최신 데이터를 불러오세요.")
    
    with tab2:
        run_rr_analysis()

if __name__ == "__main__":
    main()