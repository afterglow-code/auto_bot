import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
# =========================================================
# [사용자 설정 영역]
# =========================================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

MY_TOTAL_ASSETS = 1000000 
CURRENT_HOLDING = "KODEX 미국달러선물" 

# ⭐ [핵심 추가] 리밸런싱을 할 날짜 (매월 며칠에 할지?)
# 예: 1이면 매월 1일, 25이면 매월 25일에만 '매매 신호'를 줍니다.
# -1로 설정하면 날짜 상관없이 매일 매매 신호를 줍니다 (테스트용)
REBALANCE_DAY = 1  
# =========================================================

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
    try: requests.get(url); print("전송 완료")
    except: pass

def get_todays_signal():
    print("데이터 분석 중...")
    
    # 1. 데이터 준비 (전략 동일)
    etf_tickers = {
        'KODEX 200': '069500.KS',
        'KODEX 반도체': '091160.KS',
        'KODEX 2차전지': '305720.KS',
        'KODEX 헬스케어': '266420.KS',
        'KODEX 미국달러선물': '261240.KS'
    }
    
    # 데이터 다운로드
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
    
    kospi = yf.download(['^KS11'], start=start_date, end=end_date, progress=False)['Close'].ffill()
    tickers = list(etf_tickers.values())
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'].ffill().dropna()
    
    inv_map = {v: k for k, v in etf_tickers.items()}
    raw_data.columns = [inv_map.get(x, x) for x in raw_data.columns]

    # 2. 전략 로직 (모멘텀 + 마켓타이밍)
    momentum_score = raw_data.pct_change(60).iloc[-1]
    kospi_ma120 = kospi.rolling(window=120).mean().iloc[-1]
    current_kospi = kospi.iloc[-1]
    
    if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
    if hasattr(kospi_ma120, 'item'): kospi_ma120 = kospi_ma120.item()

    is_bull_market = current_kospi > kospi_ma120

    # 3. 목표 종목 선정
    target_stock = ""
    reason = ""
    
    if is_bull_market:
        scores = momentum_score.drop('KODEX 미국달러선물', errors='ignore')
        best_etf = scores.idxmax()
        if scores[best_etf] < 0:
            target_stock = "KODEX 미국달러선물"
            reason = "주도주 부재(모두 하락) -> 달러 방어"
        else:
            target_stock = best_etf
            reason = f"주도주 모멘텀 1위 ({scores[best_etf]*100:.1f}%)"
    else:
        target_stock = "KODEX 미국달러선물"
        reason = "하락장 방어(코스피 이탈)"

    # 4. 날짜 체크 및 메시지 생성
    today_dt = datetime.now()
    is_trading_day = (today_dt.day == REBALANCE_DAY) or (REBALANCE_DAY == -1)
    
    current_price = raw_data[target_stock].iloc[-1]
    buy_qty = int(MY_TOTAL_ASSETS // current_price)
    
    msg = f"📅 [{today_dt.strftime('%Y-%m-%d')}] 투자 비서\n"
    msg += f"상태: {'🔴상승장' if is_bull_market else '🔵하락장'}\n"
    msg += f"1위 종목: {target_stock}\n"
    msg += "-" * 20 + "\n"

    # ⭐ [핵심] 오늘이 리밸런싱 날인지에 따라 다른 행동 지시
    if is_trading_day:
        msg += "📢 [오늘은 리밸런싱 하는 날!]\n"
        if target_stock != CURRENT_HOLDING:
            msg += f"🚨 교체 신호 발생!\n"
            msg += f"매도: {CURRENT_HOLDING}\n"
            msg += f"매수: {target_stock} (약 {buy_qty}주)\n"
        else:
            msg += "✅ 포트폴리오 유지 (매매 없음)\n"
    else:
        msg += f"👀 [오늘은 관망하는 날]\n"
        msg += f"다음 리밸런싱: {today_dt.strftime('%Y-%m')}-{REBALANCE_DAY:02d}일\n"
        if target_stock != CURRENT_HOLDING:
            msg += f"(참고: 지금 리밸런싱 한다면 '{target_stock}'이 추천됩니다)\n"

    msg += "-" * 20
    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    get_todays_signal()