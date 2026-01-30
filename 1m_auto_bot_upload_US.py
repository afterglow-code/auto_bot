# dev/1m_auto_bot_upload_US.py

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import time
import re

# 리팩토링된 공통 모듈 및 설정 가져오기
from common import send_telegram, fetch_data_in_parallel
import config as cfg

def get_todays_signal():
    print("="*70)
    print("📊 미국 주식 가중모멘텀 전략 (S&P500 Top 200)")
    print("="*70)
    
    # 1. 대상 종목 리스트 구성
    try:
        print("⏳ 분석 대상 종목 수집 중... (S&P500 Top 200)")
        df_sp500 = fdr.StockListing('S&P500')
        top_200 = df_sp500.head(200)
        
        target_tickers = {row['Symbol']: row['Symbol'] for _, row in top_200.iterrows()}
        target_tickers[cfg.US_DEFENSE_ASSET] = cfg.US_DEFENSE_ASSET # 방어 자산 추가
        
        print(f"✅ 분석 대상: 총 {len(target_tickers)}개 종목 (S&P500 Top200 + {cfg.US_DEFENSE_ASSET})")

    except Exception as e:
        error_msg = f"❌ [미국 주식 봇] 종목 리스트 확보 실패: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 2. 데이터 다운로드 (병렬 처리로 변경)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        print("⏳ 데이터 병렬 다운로드 중...")
        # 2-1. 시장 지수
        market_df = fdr.DataReader(cfg.US_MARKET_INDEX, start=start_date, end=end_date)
        market_index = market_df['Close'].ffill()

        # 2-2. 개별 종목 데이터 병렬 수집
        raw_data = fetch_data_in_parallel(target_tickers, start_date, end_date)

        if raw_data.empty:
            raise Exception("유효한 데이터를 하나도 가져오지 못했습니다.")
            
        print(f"✅ {len(raw_data.columns)}개 종목 데이터 다운로드 완료")

    except Exception as e:
        error_msg = f"❌ [미국 주식 봇] 데이터 다운로드 치명적 오류: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 3. 전략 계산 (가중 평균 모멘텀)
    try:
        print("⏳ 전략 지표 계산 중...")
        w1, w2, w3 = cfg.MOMENTUM_WEIGHTS
        mom_1m = raw_data.pct_change(20).iloc[-1]
        mom_3m = raw_data.pct_change(60).iloc[-1]
        mom_6m = raw_data.pct_change(120).iloc[-1]

        weighted_score = (mom_1m.fillna(0) * w1) + (mom_3m.fillna(0) * w2) + (mom_6m.fillna(0) * w3)

        # 시장 타이밍 (SPY 60일선)
        ma_series = market_index.rolling(window=60).mean()
        current_ma = ma_series.iloc[-1]
        prev_ma = ma_series.iloc[-6] # 5일 전 MA
        current_market_index = market_index.iloc[-1]
        
        ma_is_rising = current_ma > prev_ma
        is_bull_market = current_market_index > current_ma
        is_neutral_market = not is_bull_market and ma_is_rising

        market_status = "🔴 상승장" if is_bull_market else "🟠 중립장" if is_neutral_market else "🔵 하락장"
        print(f"✅ 시장 판단: {market_status}")

    except Exception as e:
        error_msg = f"❌ [미국 주식 봇] 지표 계산 중 오류: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 4. 목표 종목 선정
    final_targets = [] 
    reason = ""
    defense_asset = cfg.US_DEFENSE_ASSET
    scores = weighted_score.drop(defense_asset, errors='ignore')
    top_assets = scores.sort_values(ascending=False)

    # 상승장: 공격 100%
    if is_bull_market:
        reason = "상승장 투자"
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [(defense_asset, 1.0)]
            reason = "주도주 부재 -> BIL 방어"
        else:
            selected = [name for name, score in top_assets.items() if score > 0][:cfg.US_TOP_N]
            count = len(selected)
            if count > 0:
                weight = 1.0 / count
                final_targets = [(s, weight) for s in selected]
                reason = f"US TOP {count} 모멘텀"
            else:
                final_targets = [(defense_asset, 1.0)]
                reason = "대상 종목 없음 -> BIL 방어"

    # 중립장: 공격 50%, 방어 50%
    elif is_neutral_market:
        reason = "중립장 분산 투자"
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [(defense_asset, 1.0)]
            reason = "주도주 부재 -> BIL 100% 방어"
        else:
            selected = [name for name, score in top_assets.items() if score > 0][:cfg.US_TOP_N]
            count = len(selected)
            if count > 0:
                weight = 0.5 / count # 공격 자산 비중 50%
                final_targets = [(s, weight) for s in selected]
                final_targets.append((defense_asset, 0.5)) # 방어 자산 비중 50%
                reason = f"US TOP {count} 모멘텀 (50% 공격)"
            else:
                final_targets = [(defense_asset, 1.0)]
                reason = "대상 종목 없음 -> BIL 100% 방어"

    # 하락장: 방어 100%
    else:
        final_targets = [(defense_asset, 1.0)]
        reason = f"하락장 방어({cfg.US_MARKET_INDEX} 이탈)"


    # 5. 메시지 전송
    msg = create_message(is_bull_market, is_neutral_market, final_targets, reason, weighted_score, raw_data)
    
    print("\n" + "="*70)
    print("메시지 미리보기:")
    print("="*70)
    clean_msg = re.sub('<.*?>', '', msg)
    print(clean_msg)
    print("="*70)

    send_telegram(msg, parse_mode='Markdown')

def create_message(is_bull_market, is_neutral_market, final_targets, reason, weighted_score, raw_data):
    """텔레그램 메시지를 생성하는 함수 (Markdown 포맷)"""
    today_dt = datetime.now()
    is_rebalance_period = (cfg.REBALANCE_PERIOD_START <= today_dt.day <= cfg.REBALANCE_PERIOD_END)
    
    market_status_emoji = "🔴 상승장" if is_bull_market else "🟠 중립장" if is_neutral_market else "🔵 하락장"

    msg = f"🇺🇸 *[{today_dt.strftime('%Y-%m-%d')}] 미국 주식 봇*\n"
    msg += f"전략: S&P500 가중모멘텀 (TOP {cfg.US_TOP_N})\n"
    msg += f"시장: {market_status_emoji} ({cfg.US_MARKET_INDEX})\n"
    msg += "---------------------------------\"n"
    
    target_list_msg = ""
    for name, weight in final_targets:
        score = weighted_score.get(name, 0.0)
        
        score_emoji = "🔥🔥" if score >= 0.5 else "🔥" if score >= 0.3 else "🙂" if score > 0 else "🛡️"

        if name in raw_data.columns:
            price = raw_data[name].iloc[-1]
            buy_budget = cfg.US_ASSETS * weight
            buy_qty = int(buy_budget // price) if price > 0 else 0
            
            target_list_msg += f"👉 {name} (점수: {score:.2f} {score_emoji})\n"
            target_list_msg += f"   - 비중: {int(weight*100)}% (약 {buy_qty}주)\n"
            target_list_msg += f"   - 현재가: ${price:.2f}\n"
        else:
             target_list_msg += f"👉 *{name}* (점수: {score:.2f})\n"

    if is_rebalance_period:
        msg += f"🔔 *리밸런싱 주간입니다*\n"
        msg += f"사유: {reason}\n\n"
        msg += target_list_msg
    else:
        next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
        msg += f"☕ *관망 모드*\n"
        msg += f"다음 리밸런싱: {next_rebalance_date.strftime('%Y-%m-%d')}\n\n"
        msg += "*이번 달 목표 (실시간 순위):*\n"
        msg += target_list_msg

    msg += "---------------------------------\"n"
    msg += f"_투자 원금: ${cfg.US_ASSETS:,}_"
    
    return msg

if __name__ == "__main__":
    get_todays_signal()
