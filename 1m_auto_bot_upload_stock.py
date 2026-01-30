# dev/1m_auto_bot_upload_stock.py

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
    print("📊 한국 개별주 변동성조절 모멘텀")
    print("="*70)
    
    # 1. 대상 종목 리스트 구성
    try:
        print("⏳ 분석 대상 종목 수집 중...")
        df_kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSPI)
        df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg.MOSIG_TOP_N_KOSDAQ)
        
        target_tickers = {}
        for _, row in pd.concat([df_kospi, df_kosdaq]).iterrows():
            target_tickers[row['Name']] = row['Code']

        # 방어 자산 추가
        target_tickers[cfg.STOCK_DEFENSE_ASSET] = cfg.ETF_TICKERS.get(cfg.STOCK_DEFENSE_ASSET, '261240')
        
        print(f"✅ 분석 대상: 총 {len(target_tickers)}개 종목 후보 확보")

    except Exception as e:
        error_msg = f"❌ [개별주 봇] 종목 리스트 확보 실패: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 2. 데이터 다운로드 (병렬 처리로 변경)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        print("⏳ 데이터 병렬 다운로드 중...")
        # 2-1. 시장 지수
        market_df = fdr.DataReader(cfg.STOCK_MARKET_INDEX, start=start_date, end=end_date)
        market_index = market_df['Close'].ffill()

        # 2-2. 개별 종목 데이터 병렬 수집
        raw_data = fetch_data_in_parallel(target_tickers, start_date, end_date)

        # 데이터 검증
        if raw_data.empty:
            raise Exception("유효한 데이터를 하나도 가져오지 못했습니다.")
        
        # 변동성 계산을 위해 최소 120일치 데이터가 있는 종목만 필터링
        valid_cols = [col for col in raw_data.columns if raw_data[col].count() >= 120]
        raw_data = raw_data[valid_cols]
        
        if raw_data.empty:
            raise Exception("최소 분석 기간(120일)을 충족하는 종목이 없습니다.")
            
        print(f"✅ {len(raw_data.columns)}개 종목 데이터 분석 준비 완료")

    except Exception as e:
        error_msg = f"❌ [개별주 봇] 데이터 다운로드 치명적 오류: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 3. 전략 계산 (변동성 조절 모멘텀)
    try:
        print("⏳ 전략 지표 계산 중...")
        daily_rets = raw_data.pct_change()
        
        ret_3m = raw_data.pct_change(60).iloc[-1]
        ret_6m = raw_data.pct_change(120).iloc[-1]
        
        vol_3m = daily_rets.rolling(60).std().iloc[-1]
        
        # 0으로 나누는 것을 방지하기 위한 작은 값
        epsilon = 1e-6 
        score_3m = ret_3m / (vol_3m + epsilon)
        score_6m = ret_6m / (vol_3m + epsilon)
        
        # 3개월, 6개월 점수 평균
        weighted_score = (score_3m.fillna(0) * 0.5) + (score_6m.fillna(0) * 0.5)

        # 시장 타이밍 (코스피 60일선)
        ma60 = market_index.rolling(window=60).mean().iloc[-1]
        current_market_index = market_index.iloc[-1]
        
        is_bull_market = current_market_index > ma60
        print(f"✅ 시장 판단: {'🔴 상승장' if is_bull_market else '🔵 하락장'}")

    except Exception as e:
        error_msg = f"❌ [개별주 봇] 지표 계산 중 오류: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 4. 목표 종목 선정
    final_targets = [] 
    reason = ""
    defense_asset = cfg.STOCK_DEFENSE_ASSET

    if is_bull_market:
        scores = weighted_score.drop(defense_asset, errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [(defense_asset, 1.0)]
            reason = "주도주 부재(전체 하락세) -> 달러 방어"
        else:
            selected = [name for name, score in top_assets.items() if score > 0][:cfg.STOCK_TOP_N]
            
            count = len(selected)
            if count > 0:
                weight = 1.0 / count
                final_targets = [(s, weight) for s in selected]
                reason = f"TOP {count} 변동성조절 모멘텀"
            else:
                final_targets = [(defense_asset, 1.0)]
                reason = "대상 종목 없음 -> 달러 방어"
    else:
        final_targets = [(defense_asset, 1.0)]
        reason = f"하락장 방어({cfg.STOCK_MARKET_INDEX} 이탈)"

    # 5. 메시지 전송
    msg = create_message(is_bull_market, final_targets, reason, weighted_score, raw_data)
    
    print("\n" + "="*70)
    print("메시지 미리보기:")
    print("="*70)
    clean_msg = re.sub('<.*?>', '', msg)
    print(clean_msg)
    print("="*70)

    send_telegram(msg, parse_mode='Markdown') # 이 봇은 마크다운을 사용해봄

def create_message(is_bull_market, final_targets, reason, weighted_score, raw_data):
    """텔레그램 메시지를 생성하는 함수 (Markdown 포맷)"""
    today_dt = datetime.now()
    is_rebalance_period = (cfg.REBALANCE_PERIOD_START <= today_dt.day <= cfg.REBALANCE_PERIOD_END)
    
    msg = f"📅 *[{today_dt.strftime('%Y-%m-%d')}] 한국 개별주 봇*\n"
    msg += f"전략: 변동성조절 모멘텀 (TOP {cfg.STOCK_TOP_N})"
    msg += f"시장: {'🔴 상승장' if is_bull_market else '🔵 하락장'}\n"
    msg += "---------------------------------"
    
    target_list_msg = ""
    for name, weight in final_targets:
        score = weighted_score.get(name, 0.0)
        
        score_emoji = "🔥🔥" if score >= 2.0 else "🔥" if score >= 1.0 else "🙂" if score > 0 else "🛡️"

        if name in raw_data.columns:
            price = raw_data[name].iloc[-1]
            buy_budget = cfg.STOCK_ASSETS * weight
            buy_qty = int(buy_budget // price) if price > 0 else 0
            
            target_list_msg += f"👉 {name} (점수: {score:.2f} {score_emoji})\n"
            target_list_msg += f"   - 비중: {int(weight*100)}% ({buy_qty}주)\n"
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

    msg += "---------------------------------"
    msg += f"_투자 원금: {cfg.STOCK_ASSETS:,}원_"
    
    return msg

if __name__ == "__main__":
    get_todays_signal()
