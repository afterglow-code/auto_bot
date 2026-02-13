# streamlit_ui.py

import streamlit as st
import time
import os
import sys
from streamlit_extras.stylable_container import stylable_container

from session_management import set_analysis_target # noqa: E402

# Adjust path to import config from parent directory
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)


def render_support_resistance_and_forecast(ticker, price_df, name=None, key_suffix="", plot_candlestick=False):
    """
    지지/저항 민감도 슬라이더 + 차트 + AI 예측 모델을 통합 렌더링
    
    Args:
        ticker: 종목 코드
        price_df: 가격 데이터프레임
        name: 종목명 (None이면 ticker 사용)
        key_suffix: 세션 상태 키 구분용 접미사
        plot_candlestick: 봉차트 사용 여부
    """
    from data_utilities import get_ai_forecasts
    
    if name is None:
        name = ticker
    
    is_kr_stock = ticker.isdigit()
    
    # 지지/저항 민감도
    applied_key = f"sr_order_applied_{key_suffix}_{ticker}"
    if applied_key not in st.session_state:
        st.session_state[applied_key] = 5

    order_input = st.slider(
        "지지/저항 민감도",
        min_value=5,
        max_value=60,
        value=int(st.session_state[applied_key]),
        step=5,
        key=f"sr_order_input_{key_suffix}_{ticker}",
    )
    if order_input != st.session_state[applied_key]:
        st.session_state[applied_key] = order_input

    # 지지/저항 차트 (cached 함수는 app.py에서만 사용 가능하므로 직접 import)
    from chart_plotting import plot_support_resistance
    fig_sr, sup, res = plot_support_resistance(
        price_df,
        order=int(st.session_state[applied_key]),
        title=f"{name} 지지/저항",
        plot_candlestick=plot_candlestick,
    )
    
    s1, s2, s3 = st.columns(3)
    if is_kr_stock:
        s1.metric("현재가", f"{float(price_df['Close'].iloc[-1]):,.0f}원")
        s2.metric("지지선", f"{float(sup):,.0f}원")
        s3.metric("저항선", f"{float(res):,.0f}원")
    else:
        s1.metric("현재가", f"${float(price_df['Close'].iloc[-1]):,.2f}")
        s2.metric("지지선", f"${float(sup):,.2f}")
        s3.metric("저항선", f"${float(res):,.2f}")
    
    st.plotly_chart(fig_sr, use_container_width=True)
    
    st.divider()
    st.markdown("**📈 AI 예측 모델 (30일)**")

    # AI 예측 캐싱
    ai_cache_key = f"ai_forecast_cache_{key_suffix}_{ticker}"
    ai_sig = (
        len(price_df),
        str(price_df.index.max()),
        float(price_df['Close'].iloc[-1])
    )

    cache_entry = st.session_state.get(ai_cache_key)
    if cache_entry is None or cache_entry.get("sig") != ai_sig:
        try:
            with st.spinner("AI 모델 계산 중..."):
                forecasts = get_ai_forecasts(price_df, prophet_periods=30, neural_periods=5, xgb_periods=5)
            st.session_state[ai_cache_key] = {
                "sig": ai_sig,
                **forecasts,
            }
        except Exception as e:
            st.error(f"예측 실패: {e}")
            st.session_state[ai_cache_key] = None

    if ai_cache_key in st.session_state and st.session_state[ai_cache_key]:
        cached = st.session_state[ai_cache_key]
        from chart_plotting import build_forecast_chart

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Prophet**")
            try:
                fig_pf = build_forecast_chart(price_df, cached["prophet"], title=f"[{ticker}] Prophet", plot_candlestick=plot_candlestick)
                st.plotly_chart(fig_pf, use_container_width=True)
                last = cached["prophet"].iloc[-1]
                st.caption(f"예측: {last['yhat']:.2f} / 하단: {last.get('yhat_lower', 0):.2f} / 상단: {last.get('yhat_upper', 0):.2f}")
            except Exception as e:
                st.error(f"예측 실패: {e}")

        with col2:
            st.markdown("**NeuralProphet**")
            try:
                fig_np = build_forecast_chart(price_df, cached["neural"], title=f"[{ticker}] NeuralProphet", plot_candlestick=plot_candlestick)
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
                st.button("🔍", key=f"btn_{code}_{rank}", on_click=set_analysis_target, args=(code, price))


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
                    st.button(f"{code_label}", key=f"rk_btn_{item['code']}_{item['rank']}", on_click=set_analysis_target, args=(item['code'], item['price']), use_container_width=True)
                else: st.caption("-")
        st.markdown("<hr style='margin: 0.1rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

def render_left_card(title, data, asset_type):
    with st.container(border=True):
        if not data:
            st.warning(f"{title} 데이터 없음")
            return

        ui_card_header(title, data['status'], data['reason'])

        is_us_asset = (asset_type == 'us')

        if asset_type == 'etf':
            import config as cfg # Assuming config is in parent dir
            ticker_map = cfg.ETF_TICKERS
        elif asset_type == 'stock':
            ticker_map = data['tickers_map']
        else:
            ticker_map = {} # US stocks use name as code

        for i, (name, weight) in enumerate(data['targets']):
            code = ticker_map.get(name, name if asset_type == 'us' else "N/A")
            price = data['raw_data_last'].get(name, 0)
            ui_target_row(i+1, name, code, weight, price, is_us=is_us_asset)

        with st.expander("🔻 전체 순위 보기 (Top 50)"):
            if data['rankings']:
                ui_ranking_list(data['rankings'], is_us=is_us_asset, limit=50)
            else:
                st.info("순위 데이터가 없습니다.")