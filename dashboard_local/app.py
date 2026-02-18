import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time
import logging

from session_management import (
    initialize_session_state,
    load_holdings_from_disk,
    save_holdings_to_disk,
    save_momentum_data_to_disk,
    load_momentum_data_from_disk,
    set_analysis_target,
    sync_show_rr_lines,
    sync_use_candlestick,
)
from data_utilities import (
    get_ticker_name_map,
    get_latest_fundamental,
    load_price_data,
    load_fundamental_history,
    load_foreign_history,
    calculate_etf_data,
    calculate_stock_data,
    calculate_us_data,
    normalize_holdings,
    get_rr_analysis,
    get_ai_forecasts,
)
from chart_plotting import (
    init_font,
    plot_ichimoku_rsi,
    plot_dynamic_ichimoku_rsi,
    plot_support_resistance,
    compute_prophet_forecast,
    compute_neuralprophet_forecast,
    compute_xgboost_forecast,
    build_forecast_chart,
)
from technical_indicators import resample_ohlc, InstitutionalExecution, calculate_atr_targets
from streamlit_ui import render_left_card
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="대시보드", layout="wide", page_icon="📊")
st.divider()
st.markdown("# 대시보드")
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

initialize_session_state()

# [설정] 스레드 컨텍스트 경고 메시지 차단 (기능에는 영향 없음)
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)



init_font()


def _freeze_rr(rr_data):
    if not rr_data:
        return None
    return (
        float(rr_data.get("entry", 0)),
        tuple(rr_data.get("targets", [])),
        tuple(rr_data.get("stops", [])),
    )


@st.cache_data(show_spinner=False)
def cached_support_resistance(price_df, order, title, plot_candlestick):
    return plot_support_resistance(
        price_df,
        order=order,
        title=title,
        plot_candlestick=plot_candlestick,
    )


@st.cache_data(show_spinner=False)
def cached_dynamic_ichimoku_rsi(view, title, entry, rr_frozen, plot_candlestick, show_rr, visible_tail_rows=None, show_bb=False):
    rr_data = None
    if rr_frozen:
        entry_f, targets, stops = rr_frozen
        rr_data = {"entry": entry_f, "targets": list(targets), "stops": list(stops)}
    return plot_dynamic_ichimoku_rsi(
        view,
        title,
        entry,
        rr_data,
        plot_candlestick=plot_candlestick,
        show_rr=show_rr,
        visible_tail_rows=visible_tail_rows,
        show_bb=show_bb,
    )


@st.cache_data(show_spinner=False)
def cached_forecast_chart(price_df, forecast_df, title, plot_candlestick=False):
    return build_forecast_chart(price_df, forecast_df, title=title, plot_candlestick=plot_candlestick)


tabs = st.tabs(["보유종목", "타점분석기", "모멘텀"])

with tabs[0]:
    st.markdown("### 보유종목")
    st.caption("보유종목을 추가/편집/저장하고, 한눈에 성과와 펀더멘탈을 확인합니다.")

    name_to_ticker, ticker_to_name = get_ticker_name_map()

    with stylable_container(
        key="holdings_top_panel",
        css_styles="""
            /* 보유종목 상단(추가/리스트) 스타일 커스터마이즈 영역 */
            div[data-testid="stStylableContainer"][data-key="holdings_top_panel"] {
                padding: 0.2rem 0.1rem;
            }
            div[data-testid="stStylableContainer"][data-key="holdings_top_panel"] [data-testid="stDataFrame"] {
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            }
        """,
    ):
        # 좌우 2컬럼 배치: 보유종목 추가 | 보유종목 리스트
        col_list, col_add = st.columns([2, 1])
        
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

            with stylable_container(
                key="holdings_query_panel",
                css_styles="""
                    /* 보유종목 조회 옵션 스타일 커스터마이즈 영역 */
                    div[data-testid="stStylableContainer"][data-key="holdings_query_panel"] {
                        background: var(--card-bg);
                        border: 1px solid var(--card-border);
                        border-radius: 14px;
                        padding: 0.7rem 0.8rem 0.4rem 0.8rem;
                        margin-top: 0.6rem;
                    }
                """,
            ):
                with st.expander("조회 옵션", expanded=False):
                    with st.form("holdings_query_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            price_lookback_days = st.number_input(
                                "가격 조회 기간(일)",
                                min_value=30,
                                max_value=1825,
                                value=365,
                                step=30,
                            )
                            fundamental_lookback_years = st.number_input(
                                "펀더멘탈 조회 기간(년)",
                                min_value=1,
                                max_value=10,
                                value=3,
                                step=1,
                            )
                        with col2:
                            history_rows = st.selectbox(
                                "히스토리 표시 개수",
                                options=[12, 24, 60, "ALL"],
                                index=3,
                            )
                            run = st.form_submit_button("펀더멘탈/차트 조회", use_container_width=True)
        
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
                    st.subheader("상세 정보 및 차트")

                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=int(price_lookback_days))).strftime("%Y-%m-%d")
                    f_start_date = (datetime.now() - timedelta(days=int(fundamental_lookback_years) * 365)).strftime("%Y%m%d")
                    f_end_date = datetime.now().strftime("%Y%m%d")

                    holdings_map = {r.get("티커"): r for r in st.session_state.get("holdings", [])}

                    def slice_hist(df):
                        if history_rows == "ALL":
                            return df
                        return df.tail(int(history_rows))

                    with stylable_container(
                        key="holdings_cards_panel",
                        css_styles="""
                            /* 보유종목 카드 영역 스타일 커스터마이즈 */
                            div[data-testid="stStylableContainer"][data-key="holdings_cards_panel"] .card {
                                max-width: 980px;
                                margin: 0.25rem auto 0.8rem auto;
                                padding: 0.85rem 1rem 1rem 1rem;
                            }
                            div[data-testid="stStylableContainer"][data-key="holdings_cards_panel"] [data-testid="stMetric"] {
                                padding: 0.45rem 0.6rem;
                            }
                            div[data-testid="stStylableContainer"][data-key="holdings_cards_panel"] [data-testid="stMetric"] label {
                                font-size: 0.72rem;
                            }
                            div[data-testid="stStylableContainer"][data-key="holdings_cards_panel"] .section-title {
                                margin-bottom: 0.1rem;
                            }
                        """,
                    ):
                        for t in tickers:
                            name = ticker_to_name.get(t, t)
                            card_left, card_center, card_right = st.columns([0.02, 0.96, 0.02])
                            with card_center:
                                st.divider()

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

                                header_left, header_right = st.columns([1.2, 2.8])
                                with header_left:
                                    st.markdown(f"<div class='section-title'>{name} ({t})</div>", unsafe_allow_html=True)
                                with header_right:
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
                                        rr_table, rr_data = get_rr_analysis(t, entry_for_rr)
                                        view = price_df.copy()
                                        rr_frozen = _freeze_rr(rr_data)
                                        st.plotly_chart(
                                            cached_dynamic_ichimoku_rsi(
                                                view,
                                                f"{name} 가격",
                                                entry_for_rr,
                                                rr_frozen,
                                                st.session_state['use_candlestick'],
                                                st.session_state['show_rr_lines'],
                                                visible_tail_rows="ALL",
                                                show_bb=True,
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
                                        from streamlit_ui import render_support_resistance_and_forecast
                                        render_support_resistance_and_forecast(
                                            ticker=t,
                                            price_df=price_df,
                                            name=name,
                                            key_suffix="holdings",
                                            plot_candlestick=st.session_state['use_candlestick']
                                        )

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


with tabs[1]:
    st.subheader("타점분석기 (Institutional Risk Manager)")
    st.caption("리스크 기반 포지션 사이징과 피라미딩 계획을 확인합니다.")

    inst_top_left, inst_top_right = st.columns(2)

    with inst_top_left:
        with st.expander("보유종목", expanded=True):
            inst_holdings_df = pd.DataFrame(st.session_state.get("holdings", []))
            if inst_holdings_df.empty:
                st.info("보유종목이 없습니다.")
            else:
                st.dataframe(
                    inst_holdings_df,
                    use_container_width=True,
                    hide_index=True,
                )

    with inst_top_right:
        with st.form("inst_calc_form"):
            # 계산 모드 선택
            calc_mode = st.radio(
                "계산 방식",
                ["손익비(R배수)", "ATR 기반"],
                horizontal=True,
                key="inst_calc_mode"
            )
            
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                inst_ticker = st.text_input(
                    "종목코드",
                    value=st.session_state.get('ticker_for_rr', ''),
                    key="inst_ticker_input",
                ).strip().upper()
            with c2:
                inst_entry = st.number_input(
                    "매수단가 (0=현재가)",
                    value=st.session_state.get('price_for_rr', 0.0),
                    key="inst_entry_price",
                )
            with c3:
                inst_lookback = st.number_input(
                    "가격 조회 기간(일)",
                    min_value=30,
                    max_value=1825,
                    value=365,
                    step=30,
                    key="inst_lookback_days",
                )

            b1, b2 = st.columns(2)
            with b1:
                total_balance = st.number_input(
                    "내 총 투자 원금 (원/$)",
                    value=10000000,
                    step=1000000,
                    key="inst_total_balance",
                )
            with b2:
                if calc_mode == "손익비(R배수)":
                    risk_tol = st.slider(
                        "허용 손실률 (Risk %)",
                        0.5,
                        5.0,
                        2.0,
                        0.5,
                        help="계좌 전체 금액 중 이 종목에서 손실볼 최대 비중",
                        key="inst_risk_tol",
                    )
                else:  # ATR 기반
                    risk_tol = 2.0  # 기본값, ATR 모드에서는 사용 안 됨
                    st.markdown("**ATR 모드**")

            b3, b4 = st.columns(2)
            with b3:
                invest_amount = st.number_input(
                    "투입 금액 (원/$)",
                    value=2000000,
                    step=100000,
                    key="inst_invest_amount",
                )
            with b4:
                if calc_mode == "손익비(R배수)":
                    target_rr = st.slider(
                        "목표 손익비",
                        1.0,
                        5.0,
                        2.0,
                        0.5,
                        key="inst_target_rr",
                    )
                else:  # ATR 기반
                    atr_mult = st.number_input(
                        "ATR 배수 (익절)",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.5,
                        key="inst_atr_mult",
                        help="익절 = 진입가 + (ATR × 배수)"
                    )
                    target_rr = 2.0  # 기본값

            # ATR 모드 추가 파라미터
            if calc_mode == "ATR 기반":
                atr_c1, atr_c2 = st.columns(2)
                with atr_c1:
                    atr_window = st.number_input(
                        "ATR 기간",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=1,
                        key="inst_atr_window",
                        help="ATR 계산을 위한 기간"
                    )
                with atr_c2:
                    atr_stop_loss = st.slider(
                        "손절 비율 (%)",
                        1.0,
                        10.0,
                        5.0,
                        0.5,
                        key="inst_atr_stop_loss",
                        help="진입가 대비 손절 비율"
                    )
            else:
                atr_window = 20
                atr_stop_loss = 5.0
                atr_mult = 3.0

            run_inst = st.form_submit_button("계산 실행", use_container_width=True)
    if run_inst and inst_ticker:
        inst_sig = (
            inst_ticker,
            float(inst_entry),
            int(inst_lookback),
            float(total_balance),
            float(risk_tol),
            float(invest_amount),
            float(target_rr),
            calc_mode,
        )
        if st.session_state.get("inst_calc_sig") != inst_sig:
            st.session_state["inst_calc_sig"] = inst_sig
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=int(inst_lookback))).strftime("%Y-%m-%d")
        price_df = load_price_data(inst_ticker, start_date, end_date)
        current_close = float(price_df["Close"].iloc[-1]) if not price_df.empty and "Close" in price_df.columns else inst_entry
        entry_price = inst_entry if inst_entry > 0 else current_close

        if entry_price and invest_amount > 0:
            qty = invest_amount / entry_price
            portfolio_pct = (invest_amount / total_balance * 100) if total_balance else 0

            # ATR 기반 계산 vs 손익비 기반 계산
            if calc_mode == "ATR 기반":
                target_price, stop_price, atr_value = calculate_atr_targets(
                    price_df,
                    entry_price,
                    atr_window=int(atr_window),
                    atr_mult=float(atr_mult),
                    stop_loss_rate=atr_stop_loss / 100.0
                )
                
                if target_price is None or stop_price is None:
                    st.error("ATR 계산에 필요한 충분한 데이터가 없습니다. 더 긴 기간을 선택하세요.")
                    st.session_state["inst_calc_cache"] = None
                else:
                    calc_method = "ATR"
                    risk_per_share = entry_price - stop_price
            else:
                # 기존 손익비 기반 계산
                allowed_loss = total_balance * (risk_tol / 100)
                risk_per_share = allowed_loss / qty if qty > 0 else 0
                stop_price = entry_price - risk_per_share
                target_price = entry_price + (risk_per_share * target_rr)
                calc_method = "R배수"

            if target_price and stop_price and risk_per_share:
                st.session_state["inst_calc_cache"] = {
                    "ticker": inst_ticker,
                    "entry": entry_price,
                    "current_close": current_close,
                    "price_df": price_df,
                    "qty": qty,
                    "invest_amount": invest_amount,
                    "portfolio_pct": portfolio_pct,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "total_balance": total_balance,
                    "risk_tol": risk_tol,
                    "target_rr": target_rr,
                    "calc_method": calc_method,
                    "risk_per_share": risk_per_share,
                }
            else:
                st.session_state["inst_calc_cache"] = None
                st.error("입력값을 확인해주세요.")
        else:
            st.session_state["inst_calc_cache"] = None
            st.error("입력값을 확인해주세요.")

    inst_left, inst_right = st.columns(2)

    with inst_left:
        st.markdown("#### 계산 결과")
        calc = st.session_state.get("inst_calc_cache")
        if not calc:
            st.caption("계산 실행을 눌러 결과를 확인하세요.")
        else:
            if calc.get("qty"):
                calc_method = calc.get("calc_method", "R배수")
                if calc_method == "ATR":
                    st.caption(
                        f"💡 방식: **ATR 기반** | 진입가 ± ATR 배수로 목표가/손절가 결정"
                    )
                else:
                    st.caption(
                        f"💡 원칙: 이 트레이딩이 실패해도 계좌 전체에서 **{int(calc['total_balance'] * calc['risk_tol'] / 100):,}** 이상 잃지 않습니다."
                    )
                k1, k2, k3 = st.columns(3)
                k1.metric("적정 매수 수량", f"{int(calc['qty']):,} 주")
                if calc["ticker"].isdigit():
                    k2.metric("총 투입 금액", f"{int(calc['invest_amount']):,} 원")
                else:
                    k2.metric("총 투입 금액", f"${calc['invest_amount']:,.2f}")
                k3.metric("포트 비중", f"{calc['portfolio_pct']:.1f} %")

                s1, s2, s3 = st.columns(3)
                expected_profit = (calc["target_price"] - calc["entry"]) * calc["qty"]
                if calc["ticker"].isdigit():
                    s1.metric("손절가", f"{int(calc['stop_price']):,} 원")
                    s2.metric("익절가", f"{int(calc['target_price']):,} 원")
                    s3.metric("예상수익", f"{int(expected_profit):,} 원")
                else:
                    s1.metric("손절가", f"${calc['stop_price']:,.2f}")
                    s2.metric("익절가", f"${calc['target_price']:,.2f}")
                    s3.metric("예상수익", f"${expected_profit:,.2f}")

                if calc['portfolio_pct'] > 30:
                    st.warning("⚠️ 경고: 한 종목 비중이 너무 높습니다. 손절폭을 좁히거나 리스크 %를 낮추세요.")

                with st.expander("🔻 기관식 피라미딩(분할매수) 계획", expanded=True):
                    plan_df = InstitutionalExecution(calc["total_balance"], calc["risk_tol"]).get_pyramiding_plan(
                        calc["entry"],
                        calc["qty"],
                        stop_price=calc["stop_price"],
                        target_price=calc["target_price"],
                    )
                    if not plan_df.empty:
                        is_kr = calc["ticker"].isdigit()
                        plan_df = plan_df.copy()
                        plan_df['가격'] = plan_df['가격'].apply(lambda x: f"{int(x):,}원" if is_kr else f"${x:,.2f}")
                        plan_df['수량'] = plan_df['수량'].apply(lambda x: f"{int(x):,}주")
                        plan_df["손절가"] = plan_df["손절가"].apply(lambda x: f"{int(x):,}원" if is_kr else f"${x:,.2f}")
                        plan_df["익절가"] = plan_df["익절가"].apply(lambda x: f"{int(x):,}원" if is_kr else f"${x:,.2f}")
                        st.table(plan_df)

                st.caption(f"목표 손익비: {calc['target_rr']:.1f}R")
            else:
                st.caption("계산 결과가 없습니다.")

    with inst_right:
        st.markdown("#### AI 예측 (30일)")
        calc = st.session_state.get("inst_calc_cache")
        if not calc:
            st.caption("계산 실행 후 예측을 표시합니다.")
        else:
            price_df = calc.get("price_df")
            if price_df is None or price_df.empty:
                st.caption("가격 데이터가 없습니다.")
            else:
                ai_cache_key = f"ai_forecast_cache_inst_{calc['ticker']}"
                ai_sig = (
                    len(price_df),
                    str(price_df.index.max()),
                    float(price_df['Close'].iloc[-1]) if "Close" in price_df.columns else 0.0,
                )

                if (
                    ai_cache_key not in st.session_state
                    or st.session_state[ai_cache_key].get("sig") != ai_sig
                ):
                    try:
                        with st.spinner("AI 모델 계산 중..."):
                            forecasts = get_ai_forecasts(price_df, prophet_periods=30, neural_periods=5, xgb_periods=5)
                        st.session_state[ai_cache_key] = {"sig": ai_sig, **forecasts}
                    except Exception as e:
                        st.error(f"예측 실패: {e}")
                        st.session_state[ai_cache_key] = None

                if ai_cache_key in st.session_state and st.session_state[ai_cache_key]:
                    cached = st.session_state[ai_cache_key]

                    # 캔들스틱 토글 추가
                    st.checkbox(
                        "봉차트 표시",
                        value=st.session_state.get('use_candlestick', False),
                        key="use_candlestick_inst",
                        on_change=sync_use_candlestick,
                        args=("use_candlestick_inst",),
                    )

                    def add_inst_levels(fig):
                        entry_price = calc["entry"]
                        levels = [
                            ("손절", calc["stop_price"], "#ef4444"),
                            ("익절", calc["target_price"], "#22c55e"),
                            ("1차", entry_price, "#0ea5e9"),
                            ("2차", entry_price * 1.02, "#8b5cf6"),
                            ("3차", entry_price * 1.02 * 1.02, "#f59e0b"),
                        ]
                        for label, price, color in levels:
                            fig.add_hline(
                                y=price,
                                line_dash="dot",
                                line_color=color,
                                annotation_text=label,
                                annotation_position="top left",
                            )
                        return fig

                    st.markdown("**Prophet**")
                    try:
                        fig_pf = cached_forecast_chart(
                            price_df, 
                            cached["prophet"], 
                            title=f"{calc['ticker']} Prophet",
                            plot_candlestick=st.session_state.get('use_candlestick', False)
                        )
                        fig_pf = add_inst_levels(fig_pf)
                        st.plotly_chart(fig_pf, use_container_width=True)
                    except Exception as e:
                        st.error(f"예측 실패: {e}")

                    st.markdown("**NeuralProphet**")
                    try:
                        fig_np = cached_forecast_chart(
                            price_df, 
                            cached["neural"], 
                            title=f"{calc['ticker']} NeuralProphet",
                            plot_candlestick=st.session_state.get('use_candlestick', False)
                        )
                        fig_np = add_inst_levels(fig_np)
                        st.plotly_chart(fig_np, use_container_width=True)
                    except Exception as e:
                        st.error(f"예측 실패: {e}")

                    st.markdown("**XGBoost (상승확률)**")
                    try:
                        for _, row in cached["xgboost"].iterrows():
                            date_str = row['ds'].strftime('%m/%d')
                            prob = row['probability']
                            color = "green" if prob > 0.5 else "red"
                            st.markdown(
                                f"{date_str}: <span style='color:{color};font-weight:bold'>{prob*100:.1f}%</span> 상승",
                                unsafe_allow_html=True,
                            )
                    except Exception as e:
                        st.error(f"예측 실패: {e}")

with tabs[2]:
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

            with st.form("momentum_analysis_form"):
                if "momentum_ticker_input" not in st.session_state:
                    st.session_state["momentum_ticker_input"] = default_ticker
                if "momentum_entry_price" not in st.session_state:
                    st.session_state["momentum_entry_price"] = default_price
                c1, c2 = st.columns(2)
                ticker = c1.text_input("종목코드", key="momentum_ticker_input").strip().upper()
                entry_price = c2.number_input("매수단가 (0=현재가)", key="momentum_entry_price")

                history_rows_m = st.selectbox("차트 데이터 표시 개수", options=[60, 120, 240, 500, "ALL"], index=1, key="momentum_history_rows")
                use_candlestick_m = st.checkbox(
                    "봉차트 표시",
                    value=st.session_state['use_candlestick'],
                    key="use_candlestick_momentum",
                )
                show_rr_lines_m = st.checkbox(
                    "손익비 라인 표시",
                    value=st.session_state['show_rr_lines'],
                    key="show_rr_lines_momentum",
                )
                show_bb_m = st.checkbox(
                    "볼린저 밴드 표시",
                    value=st.session_state.get('show_bb', False),
                    key="show_bb_momentum",
                )

                run_btn = st.form_submit_button("분석 실행", use_container_width=True)

            if run_btn:
                st.session_state['use_candlestick'] = use_candlestick_m
                st.session_state['show_rr_lines'] = show_rr_lines_m
                st.session_state['show_bb'] = show_bb_m
                st.session_state['momentum_analysis_running'] = True
                st.session_state['momentum_saved_ticker'] = ticker
                st.session_state['momentum_saved_entry'] = entry_price
                st.session_state['momentum_saved_history_rows'] = history_rows_m

            # session_state에 저장된 상태가 있으면 계속 표시
            should_run = st.session_state.get('momentum_analysis_running', False)
            if should_run:
                ticker = st.session_state.get('momentum_saved_ticker', ticker)
                entry_price = st.session_state.get('momentum_saved_entry', entry_price)
                history_rows_m = st.session_state.get('momentum_saved_history_rows', history_rows_m)

            if should_run and ticker:
                try:
                    res, rr_data = get_rr_analysis(ticker, entry_price)

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
                                rr_frozen = _freeze_rr(rr_data)
                                st.plotly_chart(
                                    cached_dynamic_ichimoku_rsi(
                                        view,
                                        f"[{ticker}] 일차트",
                                        entry_price if entry_price else None,
                                        rr_frozen,
                                        st.session_state['use_candlestick'],
                                        st.session_state['show_rr_lines'],
                                        show_bb=st.session_state.get('show_bb', False),
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
                                rr_frozen = _freeze_rr(rr_data)
                                st.plotly_chart(
                                    cached_dynamic_ichimoku_rsi(
                                        view,
                                        f"[{ticker}] 주차트",
                                        entry_price if entry_price else None,
                                        rr_frozen,
                                        st.session_state['use_candlestick'],
                                        st.session_state['show_rr_lines'],
                                        show_bb=st.session_state.get('show_bb', False),
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
                                rr_frozen = _freeze_rr(rr_data)
                                st.plotly_chart(
                                    cached_dynamic_ichimoku_rsi(
                                        view,
                                        f"[{ticker}] 월차트",
                                        entry_price if entry_price else None,
                                        rr_frozen,
                                        st.session_state['use_candlestick'],
                                        st.session_state['show_rr_lines'],
                                        show_bb=st.session_state.get('show_bb', False),
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
                                from streamlit_ui import render_support_resistance_and_forecast
                                render_support_resistance_and_forecast(
                                    ticker=ticker,
                                    price_df=df_daily,
                                    name=f"[{ticker}]",
                                    key_suffix="momentum",
                                    plot_candlestick=st.session_state['use_candlestick']
                                )
                    else:
                        st.error("데이터를 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"오류: {e}")
            elif not ticker:
                st.caption("왼쪽 리스트에서 종목을 선택하거나 코드를 입력하세요.")
st.divider()
st.caption("이 페이지에는 네이버에서 제공한 나눔 고딕 글꼴이 적용되어 있습니다.")
# streamlit run auto_bot/dashboard_local/app.py
