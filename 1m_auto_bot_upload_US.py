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

# 투자 원금 (달러 기준)
MY_TOTAL_ASSETS = 10000  # $10,000 (약 1,400만원)

# 리밸런싱 기간 (매월 1일 ~ 7일 사이)
REBALANCE_PERIOD_START = 1
REBALANCE_PERIOD_END = 7
# =========================================================

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 텔레그램 설정이 없습니다. 메시지를 보내지 않습니다.")
        print(f"[메시지 미리보기]\n{msg}")
        return
        
    # [수정] URL에 msg를 직접 넣지 않고, params 딕셔너리로 분리합니다.
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {
        'chat_id': CHAT_ID,
        'text': msg
    }
    try: 
        # params=params 를 넣어주면 알아서 & 기호를 처리해줍니다.
        requests.get(url, params=params)
        print("전송 완료")
    except Exception as e: 
        print(f"전송 실패: {e}")

def get_todays_signal():
    print("🚀 [US S&P 500 전략] 데이터 분석 시작...")
    print("⏳ 미국 데이터 수집 중... (약 2~3분 소요)")
    
    # 1. 대상 종목 리스트 구성
    target_tickers = {}
    
    try:
        df_sp500 = fdr.StockListing('S&P500')
        top_200 = df_sp500.head(200)
        
        for _, row in top_200.iterrows():
            target_tickers[row['Symbol']] = row['Symbol']

        target_tickers['BIL'] = 'BIL'
        
        print(f"-> 분석 대상: 총 {len(target_tickers)}개 종목 (S&P500 Top200 + BIL)")

    except Exception as e:
        send_telegram(f"❌ 종목 리스트 확보 실패: {e}")
        return

    # 2. 데이터 다운로드
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    market_index = None # SPY
    raw_data = pd.DataFrame()
    
    try:
        spy_df = fdr.DataReader('SPY', start=start_date, end=end_date)
        market_index = spy_df['Close'].ffill()

        df_list = []
        total_count = len(target_tickers)
        
        for i, (name, code) in enumerate(target_tickers.items()):
            if i % 20 == 0: print(f"   수집 중... ({i}/{total_count})")
            
            try:
                df = fdr.DataReader(code, start=start_date, end=end_date)
                if df.empty: continue

                series = df['Close'].rename(name)
                df_list.append(series)
            except:
                continue
            
            time.sleep(0.1) 
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).ffill().dropna(how='all')
        else:
            raise Exception("유효한 데이터를 하나도 가져오지 못했습니다.")

    except Exception as e:
        send_telegram(f"❌ 데이터 다운로드 치명적 오류: {e}")
        return

    # 3. 전략 계산 (가중 평균 모멘텀)
    try:
        mom_1m = raw_data.pct_change(20).iloc[-1]
        mom_3m = raw_data.pct_change(60).iloc[-1]
        mom_6m = raw_data.pct_change(120).iloc[-1]

        weighted_score = ((mom_1m.fillna(0) * 0.2) + (mom_3m.fillna(0) * 0.3) + (mom_6m.fillna(0) * 0.5))

        spy_ma120 = market_index.rolling(window=120).mean().iloc[-1]
        current_spy = market_index.iloc[-1]
        
        if hasattr(current_spy, 'item'): current_spy = current_spy.item()
        if hasattr(spy_ma120, 'item'): spy_ma120 = spy_ma120.item()

        is_bull_market = current_spy > spy_ma120
    except Exception as e:
        send_telegram(f"❌ 지표 계산 중 오류: {e}")
        return

    # 4. 목표 종목 선정
    final_targets = [] 
    reason = ""

    if is_bull_market:
        scores = weighted_score.drop('BIL', errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [('BIL', 1.0)]
            reason = "주도주 부재 -> BIL 방어"
        else:
            selected = []
            for name, score in top_assets.items():
                if score > 0: selected.append(name)
                if len(selected) >= 3: break 
            
            count = len(selected)
            if count > 0:
                weight = 1.0 / count
                for s in selected:
                    final_targets.append((s, weight))
                reason = f"US TOP {count} 모멘텀"
            else:
                final_targets = [('BIL', 1.0)]
                reason = "대상 종목 없음 -> BIL 방어"
    else:
        final_targets = [('BIL', 1.0)]
        reason = "하락장 방어(S&P500 이탈)"

    # 5. 메시지 전송 (점수 표시 추가)
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    msg = f"🇺🇸 [{today_dt.strftime('%Y-%m-%d')}] 미국 주식 봇\n"
    msg += f"전략: S&P500 가중모멘텀 (0.2/0.3/0.5)\n"
    msg += f"시장: {'🔴상승장' if is_bull_market else '🔵하락장'} (SPY)\n"
    msg += "-" * 20 + "\n"
    
    # [수정된 메시지 생성 로직]
    target_list_msg = ""
    for name, weight in final_targets:
        # 점수 가져오기 (BIL 등 예외 처리)
        try:
            current_score = weighted_score[name]
        except:
            current_score = 0.0
        
        # 점수에 따른 이모지 (미국장은 모멘텀 숫자가 더 크게 나옴)
        score_emoji = ""
        # 미국장은 추세가 강해서 0.3 이상이면 꽤 좋은 편
        if current_score >= 0.5: score_emoji = "🔥🔥"
        elif current_score >= 0.3: score_emoji = "🔥"
        elif current_score > 0: score_emoji = "🙂"
        else: score_emoji = "🛡️"

        if name in raw_data.columns:
            current_price = raw_data[name].iloc[-1]
            buy_budget = MY_TOTAL_ASSETS * weight
            buy_qty = int(buy_budget // current_price)
            
            target_list_msg += f"👉 {name} (점수: {current_score:.2f} {score_emoji})\n"
            target_list_msg += f"   비중: {int(weight*100)}% (약 {buy_qty}주)\n"
            target_list_msg += f"   현재가: ${current_price:.2f}\n"
        else:
             target_list_msg += f"👉 {name} (점수: {current_score:.2f})\n"

    if is_rebalance_period:
        msg += "🔔 [리밸런싱 주간입니다]\n"
        msg += f"사유: {reason}\n\n"
        msg += target_list_msg
    else:
        msg += f"☕ [관망 모드]\n이번 달 목표 (실시간 순위):\n"
        msg += target_list_msg
        msg += f"\n다음 리밸런싱: {next_rebalance_date.strftime('%Y-%m-%d')}\n"

    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    get_todays_signal()