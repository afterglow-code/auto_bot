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
    print("데이터 분석 중 (가중평균 + TOP2 전략)...")
    
    # 1. 데이터 준비
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
        'KODEX 미국30년국채액티브(H)': '484790'
    }
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    # 가중 평균(6개월) 계산을 위해 넉넉히 365일 전부터 조회
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    kospi = None
    raw_data = pd.DataFrame()

    try:
        # 1-1. KOSPI 지수 (시장 타이밍용)
        kospi_df = fdr.DataReader('KS11', start=start_date, end=end_date)
        kospi = kospi_df['Close'].ffill()

        # 1-2. ETF 데이터 수집
        df_list = []
        for name, code in etf_tickers.items():
            df = fdr.DataReader(code, start=start_date, end=end_date)
            if not df.empty:
                series = df['Close'].rename(name)
                df_list.append(series)
            time.sleep(0.1) # 차단 방지
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).ffill().dropna()
        else:
            raise Exception("데이터 수집 실패")

    except Exception as e:
        send_telegram(f"❌ 오류 발생: {e}")
        print(f"분석 실패: {e}")
        return

    # 2. [핵심] 가중 평균 모멘텀 계산
    # 최근 데이터(iloc[-1]) 기준으로 1개월(20일), 3개월(60일), 6개월(120일) 수익률 계산
    mom_1m = raw_data.pct_change(20).iloc[-1]
    mom_3m = raw_data.pct_change(60).iloc[-1]
    mom_6m = raw_data.pct_change(120).iloc[-1]

    # 종합 점수 (단기+중기+장기 평균)
    # 신규 상장주라 6개월 데이터가 없으면(NaN) 0점 처리하여 안전하게 제외
    weighted_score = ((mom_1m.fillna(0) * 0.2) + (mom_3m.fillna(0) * 0.3) + (mom_6m.fillna(0) * 0.5))

    # 시장 타이밍 (코스피 120일선)
    kospi_ma120 = kospi.rolling(window=120).mean().iloc[-1]
    current_kospi = kospi.iloc[-1]
    
    if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
    if hasattr(kospi_ma120, 'item'): kospi_ma120 = kospi_ma120.item()

    is_bull_market = current_kospi > kospi_ma120

    # 3. [핵심] 목표 종목 선정 (TOP 2 분산)
    final_targets = [] # [(종목명, 비중), (종목명, 비중)] 형태
    reason = ""

    if is_bull_market:
        # 달러 제외하고 점수 산출
        scores = weighted_score.drop('KODEX 미국달러선물', errors='ignore')
        
        # 점수 높은 순 정렬
        top_assets = scores.sort_values(ascending=False)
        
        # 1등이 0점 이하면 (모두 하락세) -> 달러
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [('KODEX 미국달러선물', 1.0)]
            reason = "주도주 부재(모두 하락) -> 달러 방어"
        else:
            # 1등과 2등을 뽑음 (점수가 양수인 경우만)
            selected = []
            for name, score in top_assets.items():
                if score > 0:
                    selected.append(name)
                if len(selected) >= 2: break
            
            # 종목 수에 따라 비중 결정
            if len(selected) == 1:
                final_targets = [(selected[0], 1.0)] # 1개면 몰빵
                reason = f"단독 주도주: {selected[0]}"
            else:
                final_targets = [(selected[0], 0.5), (selected[1], 0.5)] # 2개면 반반
                reason = f"TOP 2 분산: {selected[0]}, {selected[1]}"
    else:
        # 하락장 -> 달러 방어
        final_targets = [('KODEX 미국달러선물', 1.0)]
        reason = "하락장 방어(코스피 이탈)"

    # 4. 메시지 생성 (점수 표시 추가)
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    msg = f"📅 [{today_dt.strftime('%Y-%m-%d')}] 국내 ETF 봇\n"
    msg += f"시장: {'🔴상승장' if is_bull_market else '🔵하락장'} (KOSPI)\n"
    msg += f"전략: 가중모멘텀 + TOP2 분산\n"
    msg += "-" * 20 + "\n"
    
    # [수정된 목록 생성 로직]
    target_list_msg = ""
    for name, weight in final_targets:
        # 점수 가져오기 (달러선물은 weighted_score에 없을 수 있음)
        try:
            current_score = weighted_score[name]
        except:
            current_score = 0.0 # 달러선물 등
        
        # ETF용 이모지 기준 (ETF는 변동성이 낮아 기준을 낮춤)
        score_emoji = ""
        if current_score >= 1.0: score_emoji = "🔥🔥" # ETF가 1.0 넘으면 초대박
        elif current_score >= 0.5: score_emoji = "🔥"
        elif current_score > 0: score_emoji = "🙂"
        else: score_emoji = "🛡️"

        current_price = raw_data[name].iloc[-1]
        buy_budget = MY_TOTAL_ASSETS * weight
        buy_qty = int(buy_budget // current_price)
        
        target_list_msg += f"👉 {name} (점수: {current_score:.2f} {score_emoji})\n"
        target_list_msg += f"   비중: {int(weight*100)}% (약 {buy_qty}주)\n"

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