import FinanceDataReader as fdr
import pandas as pd
import requests
import datetime
import os
import time

# ==========================================
# [사용자 설정] 여기에 텔레그램 정보를 넣으세요
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('CHAT_ID')

# 전략 설정
TOP_N_KOSPI = 200   # 코스피 감시 대상 (시총 상위)
TOP_N_KOSDAQ = 100  # 코스닥 감시 대상 (시총 상위)
PICK_COUNT = 10     # 텔레그램으로 보낼 종목 수
STRATEGY = 'value'  # 우선순위: 'value'(모멘텀점수), 'slope'(기울기), 'marcap'(시총)

# ==========================================

def send_telegram_message(msg):
    """텔레그램 메시지 전송 함수"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': msg}
    try:
        requests.post(url, data=data)
        print("✅ 텔레그램 전송 완료")
    except Exception as e:
        print(f"❌ 텔레그램 전송 실패: {e}")

def get_latest_signals():
    print(f"[{datetime.datetime.now()}] 데이터 수집 및 분석 시작...")
    
    # 1. 대상 종목 선정
    df_kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(TOP_N_KOSPI)
    df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(TOP_N_KOSDAQ)
    target_stocks = pd.concat([df_kospi, df_kosdaq])
    
    # 결과 담을 리스트
    candidates = []
    
    # 데이터 조회 기간 (넉넉히 3달)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
    
    total = len(target_stocks)
    for idx, (code, name) in enumerate(zip(target_stocks['Code'], target_stocks['Name'])):
        print(f"\r진행률: {idx+1}/{total} ({name})", end='')
        
        try:
            # 데이터 수집
            df = fdr.DataReader(code, start_date)
            if len(df) < 20: continue # 데이터 부족하면 패스
            
            # 지표 계산
            # Momentum = (종가 / 10일전 종가) * 100
            df['Momentum'] = (df['Close'] / df['Close'].shift(10)) * 100
            # Signal = 9일 이동평균
            df['Signal'] = df['Momentum'].rolling(window=9).mean()
            # Slope = 모멘텀 변화량
            df['Slope'] = df['Momentum'] - df['Momentum'].shift(1)
            
            # 최신 데이터 (오늘, 어제, 그제)
            # 장 중이라면 iloc[-1]이 현재가, 장 마감 후라면 iloc[-1]이 오늘 종가
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            daybefore = df.iloc[-3]
            
            # ---------------------------------------------------------
            # [전략 로직 수정] 모멘텀 100선 돌파 (100 Line Breakout)
            # 조건 1: 오늘(최신) 모멘텀이 100 이상 (상승 추세 진입)
            # 조건 2: 어제(직전) 모멘텀은 100 미만 (돌파 발생)
            # 조건 3: 모멘텀 > 시그널 (정배열 상태여야 안전함)
            # ---------------------------------------------------------
            
            # 장 마감 후 기준 (today=최신봉, yesterday=직전봉)
            is_100_breakout = (today['Momentum'] >= 100) and \
                              (yesterday['Momentum'] < 100) and \
                              (today['Momentum'] > today['Signal'])
                          
            if is_100_breakout:
                candidates.append({
                    'Code': code,
                    'Name': name,
                    'Price': int(today['Close']),
                    'Momentum': today['Momentum'],
                    'Signal': today['Signal'],
                    'Slope': today['Slope']
                })
                
        except Exception as e:
            continue
            
    print("\n분석 완료!")
    return candidates

def format_message(candidates):
    """텔레그램 메시지 포맷팅"""
    if not candidates:
        return "📉 오늘은 포착된 모멘텀 돌파(Golden Cross) 종목이 없습니다."
    
    # 정렬 (설정한 우선순위에 따라)
    if STRATEGY == 'value':
        # 모멘텀 점수가 높은 순
        candidates.sort(key=lambda x: x['Momentum'], reverse=True)
        title_emoji = "🚀"
        strategy_name = "강한 돌파 (High Value)"
    elif STRATEGY == 'slope':
        # 기울기가 가파른 순
        candidates.sort(key=lambda x: x['Slope'], reverse=True)
        title_emoji = "📈"
        strategy_name = "급등 출발 (High Slope)"
    else:
        # 기본: 모멘텀 순
        candidates.sort(key=lambda x: x['Momentum'], reverse=True)
        title_emoji = "🔎"
        strategy_name = "모멘텀 알림"

    # 상위 N개 자르기
    top_list = candidates[:PICK_COUNT]
    
    msg = f"{title_emoji} [모멘텀 돌파 TOP {len(top_list)}]\n"
    msg += f"전략: {strategy_name}\n"
    msg += f"기준: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    msg += "-" * 25 + "\n"
    
    for i, stock in enumerate(top_list):
        # 예: 1. 삼성전자 (70,000)
        #     M: 105.2 / S: 101.5
        msg += f"{i+1}. {stock['Name']} ({stock['Price']:,}원)\n"
        msg += f"   M: {stock['Momentum']:.1f} / S: {stock['Signal']:.1f}\n"
    
    msg += "-" * 25
    msg += f"\n총 {len(candidates)}개 종목 포착됨"
    
    return msg

# --- 메인 실행 ---
if __name__ == "__main__":
    # 1. 종목 스캔
    detected_stocks = get_latest_signals()
    
    # 2. 메시지 만들기
    message_text = format_message(detected_stocks)
    print("------------------------------------------")
    print(message_text)
    print("------------------------------------------")
    
    # 3. 텔레그램 전송
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram_message(message_text)
    else:
        print("⚠️ 토큰이 설정되지 않아 텔레그램 전송을 건너뜁니다.")