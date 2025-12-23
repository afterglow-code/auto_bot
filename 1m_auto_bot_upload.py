import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import time

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
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 텔레그램 설정이 없습니다. 메시지를 보내지 않습니다.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
    try: 
        requests.get(url)
        print("전송 완료")
    except Exception as e: 
        print(f"전송 실패: {e}")

def get_todays_signal():
    print("데이터 분석 중 (FDR 기반)...")
    
    # 1. 데이터 준비
    # [수정] FDR 사용 시 .KS 제거 (숫자 코드만 사용)
    etf_tickers = {
        'KODEX 200': '069500',
        'KODEX 미국나스닥100TR': '379810',
        'ACE 미국S&P500': '360200',
        'KODEX 반도체': '091160',
        'KODEX 헬스케어': '266420',
        'KODEX 미국달러선물': '261240',
        'KODEX AI전력핵심설비' : '487240',
        'ACE 구글벨류체인액티브' : '483340',
        'PLUS K방산': '449170',
        #'TIGER 조선TOP10': '494670',
        'KODEX 미국30년국채액티브(H)': '484790',
        #'ACE KRX 금현물': '411060'
    }
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    # 지표(120일 이평선) 계산을 위해 넉넉히 500일 전부터 조회
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    
    kospi = None
    raw_data = pd.DataFrame()

    try:
        # 1-1. KOSPI 지수 가져오기 (FDR 코드는 'KS11')
        kospi_df = fdr.DataReader('KS11', start=start_date, end=end_date)
        kospi = kospi_df['Close'].ffill()

        # 1-2. ETF 데이터 가져오기 (반복문 사용)
        df_list = []
        for name, code in etf_tickers.items():
            # 데이터 수집
            df = fdr.DataReader(code, start=start_date, end=end_date)
            
            # 데이터가 있으면 'Close' 컬럼만 뽑아서 리스트에 추가
            if not df.empty:
                series = df['Close'].rename(name)
                df_list.append(series)
            
            # [중요] 차단 방지를 위해 0.1초 쉬어줌
            time.sleep(0.1)
        
        # 1-3. 데이터 합치기
        if df_list:
            raw_data = pd.concat(df_list, axis=1).ffill().dropna()
        else:
            raise Exception("ETF 데이터를 하나도 가져오지 못했습니다.")

    except Exception as e:
        error_msg = f"❌ 오류: 데이터 수집 실패\n{e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 2. 전략 로직 (기존과 동일)
    momentum_score = raw_data.pct_change(60).iloc[-1]
    kospi_ma120 = kospi.rolling(window=120).mean().iloc[-1]
    current_kospi = kospi.iloc[-1]
    
    # 안전장치: 단일 값 추출
    if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
    if hasattr(kospi_ma120, 'item'): kospi_ma120 = kospi_ma120.item()

    is_bull_market = current_kospi > kospi_ma120

    # 3. 목표 종목 선정
    target_stock = ""
    reason = ""
    
    if is_bull_market:
        # 달러를 제외한 종목 중 모멘텀 1등 찾기
        scores = momentum_score.drop('KODEX 미국달러선물', errors='ignore')
        
        if scores.empty:
             target_stock = "KODEX 미국달러선물"
        else:
            best_etf = scores.idxmax()
            # 1등조차 모멘텀이 마이너스라면(전부 하락세), 현금성 자산(달러) 대피
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
    
    # 다음 리밸런싱 날짜 계산 (현재 날짜 + 32일 후의 달 1일)
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    
    # 오늘이 리밸런싱 기간(1일~7일)인지 확인
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    current_price = raw_data[target_stock].iloc[-1]
    buy_qty = int(MY_TOTAL_ASSETS // current_price)
    
    msg = f"📅 [{today_dt.strftime('%Y-%m-%d')}] 투자 비서\n"
    msg += f"시장: {'🔴상승장' if is_bull_market else '🔵하락장'}\n"
    msg += "-" * 20 + "\n"
    
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