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

MY_TOTAL_ASSETS = 10000000 

REBALANCE_PERIOD_START = 1
REBALANCE_PERIOD_END = 7
# =========================================================

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 텔레그램 설정이 없습니다. 메시지를 보내지 않습니다.")
        print(f"[메시지 미리보기]\n{msg}")
        return
        
    # URL 파라미터 분리 (특수문자 & 버그 해결)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {
        'chat_id': CHAT_ID,
        'text': msg
    }
    
    try: 
        requests.get(url, params=params)
        print("전송 완료")
    except Exception as e: 
        print(f"전송 실패: {e}")

def get_todays_signal():
    print("🚀 [TOP 200 변동성조절 전략] 데이터 분석 시작...")
    
    # 1. 대상 종목 리스트 구성
    target_tickers = {}
    
    try:
        # KOSPI
        df_kospi = fdr.StockListing('KOSPI')
        top_kospi = df_kospi.sort_values('Marcap', ascending=False).head(100)
        for _, row in top_kospi.iterrows():
            target_tickers[row['Name']] = row['Code']

        # KOSDAQ
        df_kosdaq = fdr.StockListing('KOSDAQ')
        top_kosdaq = df_kosdaq.sort_values('Marcap', ascending=False).head(100)
        for _, row in top_kosdaq.iterrows():
            target_tickers[row['Name']] = row['Code']

        # 달러 선물
        target_tickers['KODEX 미국달러선물'] = '261240'
        
        print(f"-> 분석 대상: 총 {len(target_tickers)}개 종목 후보 확보")

    except Exception as e:
        send_telegram(f"❌ 종목 리스트 확보 실패: {e}")
        return

    # 2. 데이터 다운로드
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    kospi = None
    raw_data = pd.DataFrame()
    
    try:
        # 2-1. KOSPI 지수
        kospi_df = fdr.DataReader('KS11', start=start_date, end=end_date)
        kospi = kospi_df['Close'].ffill()

        # 2-2. 개별 종목 데이터 수집 Loop
        df_list = []
        total_count = len(target_tickers)
        
        for i, (name, code) in enumerate(target_tickers.items()):
            if i % 20 == 0: print(f"   수집 중... ({i}/{total_count})")
            
            try:
                df = fdr.DataReader(code, start=start_date, end=end_date)
                if df.empty or len(df) < 120: continue

                series = df['Close'].rename(name)
                df_list.append(series)
            except:
                continue
            
            time.sleep(0.05) 
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).ffill().dropna(how='all')
        else:
            raise Exception("유효한 데이터를 하나도 가져오지 못했습니다.")

    except Exception as e:
        send_telegram(f"❌ 데이터 다운로드 치명적 오류: {e}")
        return

    # 3. 전략 계산 (변동성 조절 모멘텀)
    try:
        daily_rets = raw_data.pct_change()
        
        ret_3m = raw_data.pct_change(60).iloc[-1]
        ret_6m = raw_data.pct_change(120).iloc[-1]
        
        vol_3m = daily_rets.rolling(60).std().iloc[-1]
        vol_6m = daily_rets.rolling(120).std().iloc[-1]
        
        epsilon = 1e-6 
        score_3m = ret_3m / (vol_3m + epsilon)
        score_6m = ret_6m / (vol_3m + epsilon)
        
        weighted_score = (score_3m.fillna(0) * 0.5) + (score_6m.fillna(0) * 0.5)

        kospi_ma60 = kospi.rolling(window=60).mean().iloc[-1]
        current_kospi = kospi.iloc[-1]
        
        if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
        if hasattr(kospi_ma60, 'item'): kospi_ma60 = kospi_ma60.item()

        is_bull_market = current_kospi > kospi_ma60
    except Exception as e:
        send_telegram(f"❌ 지표 계산 중 오류: {e}")
        return

    # 4. 목표 종목 선정
    final_targets = [] 
    reason = ""

    if is_bull_market:
        scores = weighted_score.drop('KODEX 미국달러선물', errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [('KODEX 미국달러선물', 1.0)]
            reason = "주도주 부재(전체 하락세) -> 달러 방어"
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
                reason = f"TOP {count} 변동성조절 모멘텀"
            else:
                final_targets = [('KODEX 미국달러선물', 1.0)]
                reason = "대상 종목 없음 -> 달러 방어"
    else:
        final_targets = [('KODEX 미국달러선물', 1.0)]
        reason = "하락장 방어(코스피 이탈)"

    # 5. 메시지 전송 (점수 표시 추가)
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    msg = f"📅 [{today_dt.strftime('%Y-%m-%d')}] 국내 주식 봇\n"
    msg += f"전략: 변동성조절 모멘텀 (TOP 3)\n"
    msg += f"시장: {'🔴상승장' if is_bull_market else '🔵하락장'}\n"
    msg += "-" * 20 + "\n"
    
    # [수정된 메시지 생성 부분]
    target_list_msg = ""
    for name, weight in final_targets:
        # 점수 가져오기 (달러선물 등 예외 처리)
        try:
            current_score = weighted_score[name]
        except:
            current_score = 0.0
        
        # 점수에 따른 이모지 (높을수록 불꽃)
        score_emoji = ""
        if current_score >= 2.0: score_emoji = "🔥🔥"
        elif current_score >= 1.0: score_emoji = "🔥"
        elif current_score > 0: score_emoji = "🙂"
        else: score_emoji = "🛡️"

        if name in raw_data.columns:
            current_price = raw_data[name].iloc[-1]
            buy_budget = MY_TOTAL_ASSETS * weight
            buy_qty = int(buy_budget // current_price)
            
            target_list_msg += f"👉 {name} (점수: {current_score:.2f} {score_emoji})\n"
            target_list_msg += f"   비중: {int(weight*100)}% ({buy_qty}주)\n"
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