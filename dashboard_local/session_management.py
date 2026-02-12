# session_management.py

import streamlit as st
import json
import os
import pickle
from datetime import datetime, timedelta, timezone

# Assuming dashboard_config is in the same directory
from dashboard_config import MOMENTUM_DATA_FILE, HOLDINGS_FILE

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

def set_analysis_target(ticker, price):
    st.session_state['ticker_for_rr'] = ticker
    st.session_state['price_for_rr'] = float(price)
    st.session_state['momentum_analysis_running'] = True
    st.session_state['momentum_saved_ticker'] = ticker
    st.session_state['momentum_saved_entry'] = float(price)
    st.session_state['momentum_ticker_input'] = ticker
    st.session_state['momentum_entry_price'] = float(price)

def sync_show_rr_lines(src_key):
    st.session_state['show_rr_lines'] = st.session_state.get(src_key, True)

def sync_use_candlestick(src_key):
    st.session_state['use_candlestick'] = st.session_state.get(src_key, True)

def update_momentum_cache(target_sector, new_part):
    current_data = st.session_state['cached_data']
    if new_part:
        current_data[target_sector] = new_part
        kst = timezone(timedelta(hours=9))
        current_data['last_update'] = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')

        if save_momentum_data_to_disk(current_data):
            st.session_state['cached_data'] = current_data
            return True
    return False
