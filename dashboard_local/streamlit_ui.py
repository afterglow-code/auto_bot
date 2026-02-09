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