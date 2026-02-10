# dev/1m_auto_bot_upload_etf.py

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
import re

# 리팩토링된 공통 모듈 및 설정 가져오기
from common import send_telegram, fetch_data_in_parallel
import config as cfg

def get_todays_signal():
    print("="*70)
    print("📊 한국 ETF 가중모멘텀 전략")
    print("="*70)
    print("⏳ 데이터 분석 중...")
    
    # 1. 데이터 준비 (config에서 설정값 로드)
    etf_tickers = cfg.ETF_TICKERS
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        # 1-1. 시장 지수 (KOSPI)
        market_df = fdr.DataReader(cfg.ETF_MARKET_INDEX, start=start_date, end=end_date)
        market_index = market_df['Close'].ffill()

        # 1-2. ETF 데이터 병렬 수집
        raw_data = fetch_data_in_parallel(etf_tickers, start_date, end_date)

        if raw_data.empty:
            raise Exception("데이터 수집 실패: 유효한 ETF 데이터를 가져오지 못했습니다.")
            
        print(f"✅ {len(raw_data.columns)}개 ETF 데이터 수집 완료")

    except Exception as e:
        error_msg = f"❌ [ETF 봇] 데이터 수집 오류: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 2. 가중 평균 모멘텀 계산
    try:
        w1, w2, w3 = cfg.MOMENTUM_WEIGHTS
        mom_1m = raw_data.pct_change(20).iloc[-1]
        mom_3m = raw_data.pct_change(60).iloc[-1]
        mom_6m = raw_data.pct_change(120).iloc[-1]

        weighted_score = (mom_1m.fillna(0) * w1) + (mom_3m.fillna(0) * w2) + (mom_6m.fillna(0) * w3)

        # 시장 타이밍 (코스피 60일선)
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
        error_msg = f"❌ [ETF 봇] 지표 계산 오류: {e}"
        print(error_msg)
        send_telegram(error_msg)
        return

    # 3. 목표 종목 선정 (TOP 2 분산)
    final_targets = []
    reason = ""
    all_rankings = []
    defense_asset = cfg.ETF_DEFENSE_ASSET
    
    scores = weighted_score.drop(defense_asset, errors='ignore')
    top_assets = scores.sort_values(ascending=False)
    
    for rank, (name, score) in enumerate(top_assets.items(), 1):
        all_rankings.append({'rank': rank, 'name': name, 'score': score, 'price': raw_data[name].iloc[-1]})

    # 상승장: 공격 100%
    if is_bull_market:
        reason = "상승장 투자"
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [(defense_asset, 1.0)]
            reason = "주도주 부재 → 달러 방어"
        else:
            selected = [name for name, score in top_assets.items() if score > 0][:2]
            if len(selected) == 1:
                final_targets = [(selected[0], 1.0)]
                reason = "단독 주도주"
            elif len(selected) == 2:
                final_targets = [(selected[0], 0.5), (selected[1], 0.5)]
                reason = "TOP 2 분산"
            else:
                final_targets = [(defense_asset, 1.0)]
                reason = "상승 모멘텀 종목 없음 → 달러 방어"

    # 중립장: 공격 50%, 방어 50%
    elif is_neutral_market:
        reason = "중립장 분산 투자"
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [(defense_asset, 1.0)]
            reason = "주도주 부재 → 달러 100% 방어"
        else:
            selected = [name for name, score in top_assets.items() if score > 0][:2]
            if len(selected) == 1:
                final_targets = [(selected[0], 0.5), (defense_asset, 0.5)]
            elif len(selected) == 2:
                final_targets = [(selected[0], 0.25), (selected[1], 0.25), (defense_asset, 0.5)]
            else:
                final_targets = [(defense_asset, 1.0)]
                reason = "상승 모멘텀 종목 없음 → 달러 100% 방어"
    
    # 하락장: 방어 100%
    else:
        final_targets = [(defense_asset, 1.0)]
        reason = f"하락장 방어 ({cfg.ETF_MARKET_INDEX} < MA60)"


    # 4. 메시지 생성
    msg = create_message(is_bull_market, is_neutral_market, final_targets, all_rankings, reason, market_index, weighted_score, raw_data)
    
    # 콘솔 출력 (HTML 태그 제거 버전)
    print("\n" + "="*70)
    print("메시지 미리보기:")
    print("="*70)
    clean_msg = re.sub('<.*?>', '', msg)
    print(clean_msg)
    print("="*70)
    
    send_telegram(msg)

def create_message(is_bull_market, is_neutral_market, final_targets, all_rankings, reason, market_index, weighted_score, raw_data):
    """텔레그램 메시지를 생성하는 함수"""
    today_dt = datetime.now(pytz.timezone('Asia/Seoul'))
    is_rebalance_period = (cfg.REBALANCE_PERIOD_START <= today_dt.day <= cfg.REBALANCE_PERIOD_END)
    
    # --- 기본 정보 ---
    msg = f"<b>🇰🇷 한국 ETF 가중모멘텀 [{today_dt.strftime('%Y-%m-%d %H:%M')}]</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    current_market_index = market_index.iloc[-1]
    ma60 = market_index.rolling(window=60).mean().iloc[-1]
    market_change_pct = ((current_market_index - ma60) / ma60) * 100
    
    msg += f"📈 <b>시장 상태 ({cfg.ETF_MARKET_INDEX})</b>\n"
    msg += f"  • 지수: {current_market_index:,.2f}\n"
    msg += f"  • 60일선: {ma60:,.2f}\n"
    msg += f"  • 상태: {'🔴 상승장' if is_bull_market else '🟠 중립장' if is_neutral_market else '🔵 하락장'} ({market_change_pct:+.1f}%)\n\n"
    
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # --- 리밸런싱/관망 모드에 따른 메시지 ---
    if is_rebalance_period:
        msg += f"🔔 <b>리밸런싱 주간</b> (사유: {reason})\n\n"
        msg += f"💼 <b>매수 대상 ({len(final_targets)}개)</b>\n\n"
        
        # --- 실제 매수 대상 표시 (기존 로직 유지) ---
        for name, weight in final_targets:
            score = weighted_score.get(name, 0.0)
            price = raw_data[name].iloc[-1] if name in raw_data.columns else 0
            buy_budget = cfg.ETF_ASSETS * weight
            buy_qty = int(buy_budget // price) if price > 0 else 0
            
            if name == cfg.ETF_DEFENSE_ASSET:
                msg += f"<b>🛡️ {name}</b>\n"
                msg += f"  • 비중: {weight*100:.0f}%\n"
                msg += f"  • 사유: {reason}\n\n"
            else:
                rank = next((r['rank'] for r in all_rankings if r['name'] == name), '-')
                emoji = "🔥🔥" if score >= 0.15 else "🔥" if score >= 0.08 else "⭐" if score > 0 else "🛡️"
                msg += f"<b>{rank}위. {name}</b> {emoji}\n"
                msg += f"  • 가격: {price:,.0f}원 | 수량: {buy_qty}주\n"
                msg += f"  • 비중: {weight*100:.0f}% ({int(buy_budget):,}원)\n"
                msg += f"  • 점수: {score:.3f}\n\n"
        
        # --- 참고용 전체 순위 목록 추가 ---
        msg += f"📋 <b>참고 순위 (상위 10개)</b>\n\n"
        for info in all_rankings[:10]:
            is_target = any(info['name'] == target_name for target_name, _ in final_targets)
            prefix = "👉 " if is_target else ""
            
            msg += f"<b>{prefix}{info['rank']}위. {info['name']}</b>\n"
            msg += f"  • 점수: {info['score']:.3f}\n"

    else:
        next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
        msg += f"☕ <b>관망 모드</b> (다음 리밸런싱: {next_rebalance_date.strftime('%m월 %d일')})\n\n"
        msg += f"📋 <b>현재 순위 (상위 10개)</b>\n\n"
        
        for info in all_rankings[:10]:
            score = info['score']
            emoji = "🔥🔥" if score >= 0.15 else "🔥" if score >= 0.08 else "⭐" if score > 0 else "💤"
            msg += f"<b>{info['rank']}위. {info['name']}</b> {emoji}\n"
            msg += f"  • 점수: {score:.3f}\n"
            msg += f"  • 가격: {info['price']:,.0f}원\n\n"

    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"<i>투자 원금: {cfg.ETF_ASSETS:,}원</i>"
    return msg

if __name__ == "__main__":
    get_todays_signal()
#코드 분리 요망