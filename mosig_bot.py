# dev/mosig_bot.py

import FinanceDataReader as fdr
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import time

# 리팩토링된 공통 모듈 및 설정 가져오기
from common import send_telegram
import config as cfg

def analyze_mosig_strategy():
    """모멘텀 돌파 종목을 병렬로 스캔하고 결과 리스트를 반환하는 함수"""
    print(f"[{datetime.datetime.now()}] 모멘텀 돌파 스캔 시작...")
    
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
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
    total = len(target_stocks)

    # --- 병렬 처리 로직 ---
    with ThreadPoolExecutor(max_workers=cfg.MOSIG_MAX_WORKERS) as executor:
        # 개별 종목 데이터 수집 및 분석 작업을 제출
        future_to_stock = {
            executor.submit(_fetch_and_check, row['Code'], row['Name'], start_date): row['Name']
            for _, row in target_stocks.iterrows()
        }
        
        for i, future in enumerate(as_completed(future_to_stock)):
            stock_name = future_to_stock[future]
            print(f"\r   분석 진행률: {i+1}/{total} ({stock_name})", end='', flush=True)
            
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
        if len(df) < 20: return None

        is_breakout, stock_info = check_breakout_signal(df, code, name)
        if is_breakout:
            return stock_info
    except Exception:
        return None
    return None

def check_breakout_signal(df, code, name):
    """
    데이터프레임을 받아 모멘텀 돌파 신호를 확인하고 정보를 반환합니다.
    """
    # 지표 계산
    df['Momentum'] = (df['Close'] / df['Close'].shift(10)) * 100
    df['Signal'] = df['Momentum'].rolling(window=9).mean()
    df['Slope'] = df['Momentum'] - df['Momentum'].shift(1)
    
    if pd.isna(df.iloc[-1]['Momentum']) or pd.isna(df.iloc[-2]['Momentum']):
        return False, None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    is_100_breakout = (today['Momentum'] >= 100) and \
                      (yesterday['Momentum'] < 100) and \
                      (today['Momentum'] > today['Signal'])
                  
    if is_100_breakout:
        return True, {
            'Code': code, 'Name': name, 'Price': int(today['Close']),
            'Momentum': today['Momentum'], 'Signal': today['Signal'], 'Slope': today['Slope']
        }
    
    return False, None

def format_message(candidates):
    """텔레그램 메시지 포맷팅"""
    if not candidates:
        return "📉 오늘은 포착된 모멘텀 돌파(Golden Cross) 종목이 없습니다."
    
    # 정렬 (설정한 우선순위에 따라)
    strategy = cfg.MOSIG_STRATEGY
    if strategy == 'value':
        candidates.sort(key=lambda x: x['Momentum'], reverse=True)
        title_emoji, strategy_name = "🚀", "강한 돌파 (High Value)"
    elif strategy == 'slope':
        candidates.sort(key=lambda x: x['Slope'], reverse=True)
        title_emoji, strategy_name = "📈", "급등 출발 (High Slope)"
    else: # 기본값
        candidates.sort(key=lambda x: x['Momentum'], reverse=True)
        title_emoji, strategy_name = "🔎", "모멘텀 알림"

    # 상위 N개 자르기
    top_list = candidates[:cfg.MOSIG_PICK_COUNT]
    
    msg = f"{title_emoji} *[모멘텀 돌파 TOP {len(top_list)}]*\n"
    msg += f"전략: {strategy_name}\n"
    msg += f"기준: {datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M')}\n"
    msg += "-" * 25 + "\n"
    
    for i, stock in enumerate(top_list):
        msg += f"*{i+1}. {stock['Name']}* ({stock['Price']:,}원)\n"
        msg += f"   M: {stock['Momentum']:.1f} / S: {stock['Signal']:.1f}\n"
    
    msg += "-" * 25
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
    
    # 3. 텔레그램 전송 (mosig_bot 전용 CHAT_ID 사용)
    send_telegram(message_text, chat_id=cfg.CHAT_ID_1P, parse_mode='Markdown')
