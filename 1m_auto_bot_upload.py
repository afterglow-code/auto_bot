import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import platform

# =========================================================
# [사용자 설정 영역]
# =========================================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

MY_TOTAL_ASSETS = 1000000 

# 리밸런싱 기간 (매월 1일 ~ 7일 사이)
REBALANCE_PERIOD_START = 1
REBALANCE_PERIOD_END = 7
# =========================================================

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
    try: requests.get(url); print("전송 완료")
    except: pass

def get_todays_signal():
    print("데이터 분석 중...")
    
    # 1. 데이터 준비
    etf_tickers = {
        'KODEX 200': '069500.KS',
        'KODEX 미국나스닥100TR': '379810.KS',
        'ACE 미국S&P500': '360200.KS',
        'KODEX 반도체': '091160.KS',
        'KODEX 헬스케어': '266420.KS',
        'KODEX 미국달러선물': '261240.KS'
    }
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    
    try:
        kospi = yf.download(['^KS11'], start=start_date, end=end_date, progress=False)['Close'].ffill()
        tickers = list(etf_tickers.values())
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'].ffill().dropna()
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(-1)
            
        inv_map = {v: k for k, v in etf_tickers.items()}
        raw_data.columns = [inv_map.get(x, x) for x in raw_data.columns]
        
    except Exception as e:
        send_telegram(f"❌ 오류: 데이터 수집 실패\n{e}")
        return

    # 2. 전략 로직
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
        if scores.empty:
             target_stock = "KODEX 미국달러선물"
        else:
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

    # 4. 날짜 및 메시지 생성
    today_dt = datetime.now()
    
    # [수정 1] 다음 리밸런싱 날짜 정확히 계산 (다음 달 1일)
    # 현재 날짜에서 32일을 더해서 다음 달로 넘긴 후, 1일로 셋팅
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    current_price = raw_data[target_stock].iloc[-1]
    buy_qty = int(MY_TOTAL_ASSETS // current_price)
    
    msg = f"📅 [{today_dt.strftime('%Y-%m-%d')}] 투자 비서\n"
    msg += f"시장: {'🔴상승장' if is_bull_market else '🔵하락장'}\n"
    msg += "-" * 20 + "\n"
    
    # [수정 2] '현재 보유' 삭제하고 '목표 종목'만 제시
    if is_rebalance_period:
        msg += "🔔 [리밸런싱 주간입니다]\n"
        msg += "계좌를 확인하고 아래 종목으로 맞추세요.\n\n"
        msg += f"👉 목표 종목: {target_stock}\n"
        msg += f"   (사유: {reason})\n"
        msg += f"   (매수 예산: 약 {buy_qty}주)\n"
    else:
        msg += f"☕ [관망 모드]\n"
        msg += f"이번 달 목표: {target_stock}\n"
        msg += f"다음 리밸런싱: {next_rebalance_date.strftime('%Y-%m-%d')}\n"

    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    get_todays_signal()