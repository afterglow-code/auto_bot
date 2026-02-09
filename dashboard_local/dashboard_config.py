# dashboard_config.py

import os

# 파일 경로 설정
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MOMENTUM_DATA_FILE = os.path.join(DATA_DIR, "momentum_dashboard.pkl")
HOLDINGS_FILE = os.path.join(DATA_DIR, "holdings.json")

# 차트 색상 설정
CANDLE_UP_COLOR = '#26A69A' # Green for increasing candles
CANDLE_DOWN_COLOR = '#EF5350' # Red for decreasing candles
