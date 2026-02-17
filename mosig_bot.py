# dev/mosig_bot.py

import FinanceDataReader as fdr
import pandas as pd
import numpy as np  # ATR 계산을 위해 추가
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import time

# 리팩토링된 공통 모듈 및 설정 가져오기
from common import send_telegram
import config as cfg

# --- 백테스트에서 검증된 파라미터 ---
ATR_WINDOW = 20
ATR_MULT = 3.0        # 익절 목표 (ATR의 3배)
STOP_LOSS_RATE = 0.05 # 손절 (5%)
VOL_MULT = 2.0        # 거래량 급증 기준 (2배)

def analyze_mosig_strategy():
    """모멘텀 돌파 종목을 병렬로 스캔하고 결과 리스트를 반환하는 함수"""
    print(f"[{datetime.datetime.now()}] 모멘텀 돌파(Hybrid) 스캔 시작...")
    
    # 1. 대상 종목 선정
    try:
        df_kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSPI)
        df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSDAQ)
        target_stocks = pd.concat([df_kospi, df_kosdaq])
        print(f"✅ 스캔 대상: {len(target_stocks)}개 종목")
    except Exception as e:
        error_msg = f"❌ [모시그 봇] 대상 종목 선정 실패: {e}"
        print(error_msg)
        return []

    # 결과 담을 리스트
    candidates = []
    # ATR 계산(20일)을 위해 데이터 여유있게 90일치 로드
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
    total = len(target_stocks)

    # --- 병렬 처리 로직 ---
    with ThreadPoolExecutor(max_workers=cfg.MOSIG_MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(_fetch_and_check, row['Code'], row['Name'], start_date): row['Name']
            for _, row in target_stocks.iterrows()
        }
        
        for i, future in enumerate(as_completed(future_to_stock)):
            stock_name = future_to_stock[future]
            # 진행 상황 표시 (선택사항)
            # print(f"\r   분석 진행률: {i+1}/{total} ({stock_name})", end='', flush=True)
            
            result = future.result()
            if result:
                candidates.append(result)
    
    print("\n✅ 분석 완료!")
    return candidates

def _fetch_and_check(code, name, start_date):
    """(내부 함수) 단일 종목 데이터 수집 및 신호 분석"""
    try:
        time.sleep(cfg.MOSIG_REQUEST_DELAY)
        df = fdr.DataReader(code, start_date)
        # ATR 계산 및 모멘텀 계산을 위해 최소 30일 이상 데이터 필요
        if len(df) < 30: return None

        is_breakout, stock_info = check_breakout_signal(df, code, name)
        if is_breakout:
            return stock_info
    except Exception:
        return None
    return None

def check_breakout_signal(df, code, name):
    """
    데이터프레임을 받아 Hybrid 모멘텀 신호(거래량+ATR)를 확인하고
    익절/손절가를 계산하여 반환합니다.
    """
    # 1. 모멘텀 지표 계산
    df['Momentum'] = (df['Close'] / df['Close'].shift(10)) * 100
    df['Signal'] = df['Momentum'].rolling(window=9).mean()
    
    # 2. ATR(변동성) 계산 - 익절가 산정용
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=ATR_WINDOW).mean()

    # 데이터 유효성 체크
    if pd.isna(df.iloc[-1]['Momentum']) or pd.isna(df.iloc[-2]['Momentum']) or pd.isna(df.iloc[-1]['ATR']):
        return False, None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    # --- [조건 검증] ---
    # 1) 모멘텀 돌파 (기존 로직)
    is_momentum_break = (today['Momentum'] >= 100) and \
                        (yesterday['Momentum'] < 100) and \
                        (today['Momentum'] > today['Signal'])
    
    # 2) 거래량 폭증 (백테스트 승률 개선 핵심)
    # 거래량이 0인 경우 방지 및 2배수 확인
    if yesterday['Volume'] > 0:
        is_volume_spike = today['Volume'] >= (yesterday['Volume'] * VOL_MULT)
    else:
        is_volume_spike = False

    # 최종 진입 조건 (모멘텀 + 거래량)
    if is_momentum_break and is_volume_spike:
        current_price = int(today['Close'])
        atr_value = today['ATR']
        
        # --- [익절/손절가 계산] ---
        # 익절: ATR * 3배 위
        target_price = int(current_price + (atr_value * ATR_MULT))
        # 손절: -5% 아래 (고정)
        stop_price = int(current_price * (1 - STOP_LOSS_RATE))
        
        # 수익률(%)로 환산해서 보여주기 위함
        target_pct = ((target_price - current_price) / current_price) * 100
        
        return True, {
            'Code': code, 
            'Name': name, 
            'Price': current_price,
            'TargetPrice': target_price,
            'StopPrice': stop_price,
            'TargetPct': target_pct,
            'Momentum': today['Momentum'], 
            'VolumeRatio': today['Volume'] / yesterday['Volume'] if yesterday['Volume'] > 0 else 0
        }
    
    return False, None

def format_message(candidates):
    """텔레그램 메시지 포맷팅 (익절/손절가 포함)"""
    if not candidates:
        return "📉 오늘은 포착된 하이브리드(Hybrid) 돌파 종목이 없습니다."
    
    # 모멘텀 강한 순으로 정렬
    candidates.sort(key=lambda x: x['Momentum'], reverse=True)

    # 상위 N개
    top_list = candidates[:cfg.MOSIG_PICK_COUNT]
    
    msg = f"🚀 *[Mosig Hybrid Signal]*\n"
    msg += f"기준: {datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M')}\n"
    msg += f"조건: 거래량 {VOL_MULT}배↑ / 손절 -{STOP_LOSS_RATE*100}%\n"
    msg += "-" * 28 + "\n"
    
    for i, stock in enumerate(top_list):
        msg += f"*{i+1}. {stock['Name']}* ({stock['Code']})\n"
        msg += f"   💰 현  재: {stock['Price']:,}원\n"
        msg += f"   🎯 목  표: *{stock['TargetPrice']:,}원* (+{stock['TargetPct']:.1f}%)\n"
        msg += f"   🛡️ 손  절: {stock['StopPrice']:,}원\n"
        msg += f"   📊 M: {stock['Momentum']:.1f} / Vol: {stock['VolumeRatio']:.1f}배\n\n"
    
    msg += "-" * 28
    msg += f"\n총 {len(candidates)}개 종목 포착됨"
    
    return msg

# --- 메인 실행 ---
if __name__ == "__main__":
    # 1. 종목 스캔
    detected_stocks = analyze_mosig_strategy()
    
    # 2. 메시지 만들기
    message_text = format_message(detected_stocks)
    print("------------------------------------------")
    print(message_text)
    print("------------------------------------------")
    
    # 3. 텔레그램 전송
    send_telegram(message_text, chat_id=cfg.CHAT_ID_1P, parse_mode='Markdown')