import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import time
import random

# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 설정 파라미터 (config 파일 의존성 제거 및 내부화) ---
MOSIG_TOP_N_US = 500  # S&P 500 전체 스캔
MOSIG_MAX_WORKERS = 4 # 야후 파이낸스 차단 방지를 위해 스레드 4개로 제한
MOSIG_PICK_COUNT = 5  # 메시지에 표시할 상위 종목 수

ATR_WINDOW = 20
ATR_MULT = 3.0        # 익절 목표 (ATR의 3배)
STOP_LOSS_RATE = 0.05 # 손절 (-5%)
VOL_MULT = 2.0        # 거래량 급증 기준 (20일 평균 대비 2배)

# 텔레그램 설정이 있다면 여기에 입력 (테스트 시 print로 확인 가능)
TELEGRAM_TOKEN = "여기에_토큰_입력"
CHAT_ID = "여기에_챗ID_입력"

def send_telegram(message):
    """간단한 텔레그램 발송 함수 내장"""
    import requests
    if TELEGRAM_TOKEN == "여기에_토큰_입력":
        return # 토큰이 없으면 콘솔 출력만 수행
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")

def analyze_mosig_strategy_us():
    """미국 S&P 500 대상 모멘텀 돌파 종목 스캔"""
    print(f"[{datetime.datetime.now()}] 🇺🇸 미국장 MOSIG 스캔 시작...")
    
    try:
        # 미국 S&P 500 종목 리스트 로드
        df_us = fdr.StockListing('S&P500')
        target_stocks = df_us.head(MOSIG_TOP_N_US)
        print(f"✅ 스캔 대상: S&P 500 {len(target_stocks)}개 종목")
    except Exception as e:
        print(f"❌ 대상 종목 선정 실패: {e}")
        return []

    candidates = []
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
    total = len(target_stocks)

    # API 차단 방지를 위한 멀티스레딩 제한
    with ThreadPoolExecutor(max_workers=MOSIG_MAX_WORKERS) as executor:
        # 미국 주식은 'Code' 대신 'Symbol'을 사용합니다.
        future_to_stock = {
            executor.submit(_fetch_and_check, row['Symbol'], row['Name'], start_date): row['Name']
            for _, row in target_stocks.iterrows()
        }
        
        for i, future in enumerate(as_completed(future_to_stock)):
            stock_name = future_to_stock[future]
            print(f"\r   분석 진행률: {i+1}/{total} ({stock_name[:15]:<15})", end='', flush=True)
            
            result = future.result()
            if result:
                candidates.append(result)
    
    print("\n✅ 🇺🇸 미국장 분석 완료!")
    return candidates

def _fetch_and_check(symbol, name, start_date):
    """야후 파이낸스 API 차단 방어가 적용된 데이터 수집 및 신호 분석"""
    # 동시 요청 분산을 위한 0.2 ~ 0.5초 랜덤 대기
    time.sleep(random.uniform(0.2, 0.5)) 
    
    # 3회 재시도 로직 적용
    for attempt in range(3):
        try:
            df = fdr.DataReader(symbol, start_date)
            if len(df) < 30: return None

            is_breakout, stock_info = check_breakout_signal(df, symbol, name)
            if is_breakout:
                return stock_info
            break # 에러 없이 처리 완료되면 재시도 루프 탈출
        except Exception:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1)) # 지수 백오프
            else:
                return None
    return None

def check_breakout_signal(df, symbol, name):
    """모멘텀 돌파 + 동적 거래량 급증 확인 로직 (기존과 수학적으로 동일)"""
    df['Momentum'] = (df['Close'] / df['Close'].shift(10)) * 100
    df['Signal'] = df['Momentum'].rolling(window=9).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=ATR_WINDOW).mean()

    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    if pd.isna(df.iloc[-1]['Momentum']) or pd.isna(df.iloc[-2]['Vol_MA20']) or pd.isna(df.iloc[-1]['ATR']):
        return False, None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    # 1. 모멘텀 돌파
    is_momentum_break = (today['Momentum'] >= 100) and \
                        (yesterday['Momentum'] < 100) and \
                        (today['Momentum'] > today['Signal'])
    
    # 2. 거래량 폭증
    vol_ma_baseline = yesterday['Vol_MA20']
    if vol_ma_baseline > 0:
        is_volume_spike = today['Volume'] >= (vol_ma_baseline * VOL_MULT)
    else:
        is_volume_spike = False

    # 최종 진입 조건 판별 및 가격 포맷팅 (달러 소수점 둘째 자리 적용)
    if is_momentum_break and is_volume_spike:
        current_price = float(today['Close'])
        atr_value = float(today['ATR'])
        
        target_price = current_price + (atr_value * ATR_MULT)
        stop_price = current_price * (1 - STOP_LOSS_RATE)
        target_pct = ((target_price - current_price) / current_price) * 100
        vol_ratio = today['Volume'] / vol_ma_baseline if vol_ma_baseline > 0 else 0

        return True, {
            'Symbol': symbol, 
            'Name': name, 
            'Price': round(current_price, 2),
            'TargetPrice': round(target_price, 2),
            'StopPrice': round(stop_price, 2),
            'TargetPct': target_pct,
            'Momentum': today['Momentum'], 
            'VolumeRatio': vol_ratio,
            'ATR': round(atr_value, 2)
        }
    
    return False, None

def format_message(candidates):
    """달러($) 기호가 적용된 텔레그램 메시지 포맷팅"""
    if not candidates:
        return "📉 오늘은 포착된 🇺🇸미국장 하이브리드 돌파 종목이 없습니다."
    
    candidates.sort(key=lambda x: x['Momentum'], reverse=True)
    top_list = candidates[:MOSIG_PICK_COUNT]
    
    # 미국 현지 시간(동부 표준시) 병기
    kst_time = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%m-%d %H:%M KST')
    est_time = datetime.datetime.now(pytz.timezone('America/New_York')).strftime('%m-%d %H:%M EST')
    
    msg = f"🗽 *[US Mosig Hybrid Signal]*\n"
    msg += f"시간: {kst_time} ({est_time})\n"
    msg += f"조건: 거래량 {VOL_MULT}배↑ / 손절 -{int(STOP_LOSS_RATE*100)}%\n"
    msg += "-" * 30 + "\n"
    
    for i, stock in enumerate(top_list):
        msg += f"*{i+1}. {stock['Name']}* ({stock['Symbol']})\n"
        msg += f"   💰 현  재: ${stock['Price']}\n"
        msg += f"   🎯 목  표: *${stock['TargetPrice']}* (+{stock['TargetPct']:.1f}%)\n"
        msg += f"   🛡️ 손  절: ${stock['StopPrice']}\n"
        msg += f"   📊 M: {stock['Momentum']:.1f} / Vol: {stock['VolumeRatio']:.1f}배 / ATR: ${stock['ATR']}\n\n"
    
    msg += "-" * 30
    msg += f"\n총 {len(candidates)}개 종목 포착됨 (상위 {len(top_list)}개 출력)"
    
    return msg

if __name__ == "__main__":
    # 1. 스캔 실행
    detected_stocks = analyze_mosig_strategy_us()
    
    # 2. 결과 포맷팅
    message_text = format_message(detected_stocks)
    print("\n" + "="*45)
    print(message_text)
    print("="*45)
    
    # 3. 텔레그램 전송 (토큰 세팅 시 작동)
    send_telegram(message_text)