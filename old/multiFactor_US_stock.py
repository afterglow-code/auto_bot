import FinanceDataReader as fdr
import pandas as pd
import numpy as np
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
MY_TOTAL_ASSETS = 10000  # $10,000

# 멀티팩터 가중치 (미국 최적화 버전)
MOMENTUM_WEIGHT = 0.5   # 모멘텀 50%
VALUE_WEIGHT = 0.2      # 밸류 20%
QUALITY_WEIGHT = 0.3    # 퀄리티 30%
NUM_STOCKS = 5          # 보유 종목 수

# 리밸런싱 기간 (매월 1일 ~ 7일 사이)
REBALANCE_PERIOD_START = 1
REBALANCE_PERIOD_END = 7
# =========================================================


def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 텔레그램 설정이 없습니다. 메시지를 보내지 않습니다.")
        print(f"[메시지 미리보기]\n{msg}")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {
        'chat_id': CHAT_ID,
        'text': msg,
        'parse_mode': 'HTML'  # HTML 포맷 지원
    }
    try: 
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("✅ 텔레그램 전송 완료")
        else:
            print(f"⚠️ 텔레그램 전송 실패: {response.status_code}")
    except Exception as e: 
        print(f"❌ 전송 오류: {e}")


def calculate_multifactor_score(data, fundamental_data):
    """멀티팩터 점수 계산"""
    
    # 1. 모멘텀 팩터 (1M/3M/6M 가중)
    try:
        mom_1m = data.pct_change(20).iloc[-1]
        mom_3m = data.pct_change(60).iloc[-1]
        mom_6m = data.pct_change(120).iloc[-1]
        
        momentum_score = (
            mom_1m.fillna(0) * 0.2 +
            mom_3m.fillna(0) * 0.3 +
            mom_6m.fillna(0) * 0.5
        )
    except:
        momentum_score = pd.Series(index=data.columns, data=0)
    
    # 2. 밸류 팩터 (시가총액 역수)
    value_scores = {}
    for ticker in data.columns:
        if ticker == 'BIL':
            value_scores[ticker] = 0
            continue
        
        fund = fundamental_data.get(ticker, {})
        mcap = fund.get('marketcap', 0)
        
        if mcap > 0:
            value_scores[ticker] = 1 / np.log10(mcap + 1)
        else:
            value_scores[ticker] = 0
    
    value_score = pd.Series(value_scores)
    
    # 3. 퀄리티 팩터 (변동성 조정 일관성)
    quality_scores = {}
    for ticker in data.columns:
        if ticker == 'BIL':
            quality_scores[ticker] = 0
            continue
        
        try:
            recent_returns = data[ticker].pct_change().tail(120)
            volatility = recent_returns.std()
            positive_ratio = (recent_returns > 0).sum() / len(recent_returns)
            
            quality_scores[ticker] = positive_ratio / (volatility + 1e-6)
        except:
            quality_scores[ticker] = 0
    
    quality_score = pd.Series(quality_scores)
    
    # 4. 정규화
    def normalize(series):
        if series.std() == 0:
            return series
        return (series - series.min()) / (series.max() - series.min())
    
    mom_norm = normalize(momentum_score.fillna(0))
    val_norm = normalize(value_score.fillna(0))
    qual_norm = normalize(quality_score.fillna(0))
    
    # 5. 종합 점수
    total_score = (
        mom_norm * MOMENTUM_WEIGHT +
        val_norm * VALUE_WEIGHT +
        qual_norm * QUALITY_WEIGHT
    )
    
    # 개별 점수도 반환 (메시지 표시용)
    return total_score, mom_norm, val_norm, qual_norm


def get_todays_signal():
    print("="*70)
    print("🇺🇸 미국 멀티팩터 전략 신호 생성기")
    print("="*70)
    print(f"⏳ 데이터 수집 중... (약 2~3분 소요)")
    
    # 1. 종목 리스트 구성
    target_tickers = {}
    fundamental_data = {}
    
    try:
        df_sp500 = fdr.StockListing('S&P500')
        top_200 = df_sp500.head(200)
        
        for _, row in top_200.iterrows():
            ticker = row['Symbol']
            target_tickers[ticker] = ticker
            fundamental_data[ticker] = {
                'sector': row.get('Sector', 'Unknown'),
                'marketcap': row.get('Market Cap', 0)
            }
        
        target_tickers['BIL'] = 'BIL'
        fundamental_data['BIL'] = {'sector': 'Cash', 'marketcap': 0}
        
        print(f"✅ 분석 대상: {len(target_tickers)}개 종목 (S&P500 Top200 + BIL)")
    
    except Exception as e:
        send_telegram(f"❌ 종목 리스트 확보 실패: {e}")
        return
    
    # 2. 데이터 다운로드
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    
    market_index = None
    raw_data = pd.DataFrame()
    
    try:
        spy_df = fdr.DataReader('SPY', start=start_date, end=end_date)
        market_index = spy_df['Close'].ffill()
        
        df_list = []
        total_count = len(target_tickers)
        
        for i, (name, code) in enumerate(target_tickers.items()):
            if i % 20 == 0:
                print(f"   진행: {i}/{total_count} ({i/total_count*100:.0f}%)")
            
            try:
                df = fdr.DataReader(code, start=start_date, end=end_date)
                if df.empty or len(df) < 150:
                    continue
                
                series = df['Close'].rename(name)
                df_list.append(series)
            except:
                continue
            
            time.sleep(0.1)
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).fillna(method='ffill', limit=5)
            missing_ratio = raw_data.isnull().sum() / len(raw_data)
            valid_cols = missing_ratio[missing_ratio < 0.1].index
            raw_data = raw_data[valid_cols]
            print(f"✅ {len(raw_data.columns)}개 종목 데이터 준비 완료")
        else:
            raise Exception("유효한 데이터를 가져오지 못했습니다.")
    
    except Exception as e:
        send_telegram(f"❌ 데이터 다운로드 실패: {e}")
        return
    
    # 3. 멀티팩터 점수 계산
    try:
        total_score, mom_score, val_score, qual_score = calculate_multifactor_score(
            raw_data, fundamental_data)
        
        # 시장 타이밍
        spy_ma120 = market_index.rolling(window=120).mean().iloc[-1]
        current_spy = market_index.iloc[-1]
        
        if hasattr(current_spy, 'item'):
            current_spy = current_spy.item()
        if hasattr(spy_ma120, 'item'):
            spy_ma120 = spy_ma120.item()
        
        is_bull_market = current_spy > spy_ma120
        
        print(f"✅ 시장 판단: {'🔴 상승장' if is_bull_market else '🔵 하락장'}")
    
    except Exception as e:
        send_telegram(f"❌ 지표 계산 중 오류: {e}")
        return
    
    # 4. 종목 선정
    final_targets = []
    reason = ""
    top_10_info = []  # TOP 10 정보 저장
    
    if is_bull_market:
        scores = total_score.drop('BIL', errors='ignore').dropna()
        sorted_scores = scores.sort_values(ascending=False)
        
        # TOP 10 정보 저장 (메시지용)
        for rank, (ticker, score) in enumerate(sorted_scores.head(10).items(), 1):
            top_10_info.append({
                'rank': rank,
                'ticker': ticker,
                'total_score': score,
                'mom_score': mom_score.get(ticker, 0),
                'val_score': val_score.get(ticker, 0),
                'qual_score': qual_score.get(ticker, 0),
                'price': raw_data[ticker].iloc[-1] if ticker in raw_data.columns else 0
            })
        
        if sorted_scores.empty or sorted_scores.iloc[0] <= 0:
            final_targets = [('BIL', 1.0)]
            reason = "주도주 부재 → BIL 방어"
        else:
            selected = []
            for name, score in sorted_scores.items():
                if score > 0:
                    selected.append(name)
                if len(selected) >= NUM_STOCKS:
                    break
            
            if selected:
                weight = 1.0 / len(selected)
                for s in selected:
                    final_targets.append((s, weight))
                reason = f"멀티팩터 TOP {len(selected)}"
            else:
                final_targets = [('BIL', 1.0)]
                reason = "대상 종목 없음 → BIL 방어"
    else:
        final_targets = [('BIL', 1.0)]
        reason = "하락장 방어 (SPY < MA120)"
        
        # 하락장에도 TOP 10은 보여주기
        scores = total_score.drop('BIL', errors='ignore').dropna()
        sorted_scores = scores.sort_values(ascending=False)
        
        for rank, (ticker, score) in enumerate(sorted_scores.head(10).items(), 1):
            top_10_info.append({
                'rank': rank,
                'ticker': ticker,
                'total_score': score,
                'mom_score': mom_score.get(ticker, 0),
                'val_score': val_score.get(ticker, 0),
                'qual_score': qual_score.get(ticker, 0),
                'price': raw_data[ticker].iloc[-1] if ticker in raw_data.columns else 0
            })
    
    # 5. 메시지 생성
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    # HTML 포맷으로 메시지 작성
    msg = f"<b>🇺🇸 미국 멀티팩터 전략 [{today_dt.strftime('%Y-%m-%d')}]</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 전략 정보
    msg += f"📊 <b>전략 구성</b>\n"
    msg += f"  • 모멘텀: {MOMENTUM_WEIGHT*100:.0f}%\n"
    msg += f"  • 밸류: {VALUE_WEIGHT*100:.0f}%\n"
    msg += f"  • 퀄리티: {QUALITY_WEIGHT*100:.0f}%\n"
    msg += f"  • 보유: {NUM_STOCKS}종목\n\n"
    
    # 시장 상태
    spy_change = ((current_spy - spy_ma120) / spy_ma120) * 100
    msg += f"📈 <b>시장 상태</b>\n"
    msg += f"  • S&P 500: ${current_spy:.2f}\n"
    msg += f"  • MA120: ${spy_ma120:.2f}\n"
    msg += f"  • 시장: {'🔴 상승장' if is_bull_market else '🔵 하락장'} ({spy_change:+.1f}%)\n\n"
    
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 리밸런싱 여부에 따라 메시지 구분
    if is_rebalance_period:
        msg += f"🔔 <b>리밸런싱 주간</b>\n"
        msg += f"사유: {reason}\n\n"
        
        msg += f"💼 <b>매수 종목 ({len(final_targets)}개)</b>\n\n"
        
        for info in top_10_info[:NUM_STOCKS]:
            rank = info['rank']
            ticker = info['ticker']
            total = info['total_score']
            mom = info['mom_score']
            val = info['val_score']
            qual = info['qual_score']
            price = info['price']
            
            # 점수에 따른 이모지
            if total >= 0.8:
                emoji = "🔥🔥"
            elif total >= 0.6:
                emoji = "🔥"
            elif total >= 0.4:
                emoji = "⭐"
            else:
                emoji = "💡"
            
            # 종목이 선택되었는지 확인
            is_selected = any(t[0] == ticker for t in final_targets)
            
            if is_selected:
                weight = next(t[1] for t in final_targets if t[0] == ticker)
                buy_budget = MY_TOTAL_ASSETS * weight
                buy_qty = int(buy_budget // price) if price > 0 else 0
                
                msg += f"<b>{rank}위. {ticker}</b> {emoji}\n"
                msg += f"  • 가격: ${price:.2f} | 수량: {buy_qty}주\n"
                msg += f"  • 비중: {weight*100:.0f}% (${int(buy_budget):,})\n"
                msg += f"  • 점수: {total:.3f} (M:{mom:.2f} V:{val:.2f} Q:{qual:.2f})\n\n"
        
        # BIL인 경우
        if final_targets[0][0] == 'BIL':
            msg += f"<b>🛡️ BIL (초단기 국채)</b>\n"
            msg += f"  • 비중: 100%\n"
            msg += f"  • 사유: {reason}\n\n"
    
    else:
        msg += f"☕ <b>관망 모드</b>\n"
        msg += f"다음 리밸런싱: {next_rebalance_date.strftime('%m월 %d일')}\n\n"
        
        msg += f"📋 <b>현재 TOP 10 순위</b>\n\n"
        
        for info in top_10_info:
            rank = info['rank']
            ticker = info['ticker']
            total = info['total_score']
            mom = info['mom_score']
            price = info['price']
            
            # 상위 5개만 상세 정보
            if rank <= 5:
                if total >= 0.8:
                    emoji = "🔥🔥"
                elif total >= 0.6:
                    emoji = "🔥"
                else:
                    emoji = "⭐"
                
                msg += f"<b>{rank}위. {ticker}</b> {emoji}\n"
                msg += f"  • 점수: {total:.3f} (모멘텀:{mom:.2f})\n"
                msg += f"  • 가격: ${price:.2f}\n\n"
            else:
                # 6~10위는 간략하게
                msg += f"{rank}위. {ticker} ({total:.3f})\n"
    
    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"<i>투자 원금: ${MY_TOTAL_ASSETS:,}</i>"
    
    print("\n" + "="*70)
    print("메시지 미리보기:")
    print("="*70)
    # HTML 태그 제거한 버전으로 출력
    import re
    clean_msg = re.sub('<.*?>', '', msg)
    print(clean_msg)
    print("="*70)
    
    send_telegram(msg)


if __name__ == "__main__":
    get_todays_signal()
