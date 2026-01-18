import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import time
import numpy as np


# =========================================================
# [사용자 설정 영역]
# =========================================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

MY_TOTAL_ASSETS = 10000000 

REBALANCE_PERIOD_START = 1
REBALANCE_PERIOD_END = 7

# 멀티팩터 전략 파라미터
MOMENTUM_WEIGHT = 0.4
VALUE_WEIGHT = 0.3
QUALITY_WEIGHT = 0.3
VOLATILITY_WEIGHT = 0.0
NUM_STOCKS = 5
# =========================================================


def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 텔레그램 설정이 없습니다. 메시지를 보내지 않습니다.")
        print(f"[메시지 미리보기]\n{msg}")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {
        'chat_id': CHAT_ID,
        'text': msg
    }
    
    try: 
        requests.get(url, params=params)
        print("✅ 텔레그램 전송 완료")
    except Exception as e: 
        print(f"❌ 텔레그램 전송 실패: {e}")


def get_todays_signal():
    print("="*70)
    print("🎯 [멀티팩터 퀀트 전략] 데이터 분석 시작")
    print(f"   모멘텀 {MOMENTUM_WEIGHT*100:.0f}% | 밸류 {VALUE_WEIGHT*100:.0f}% | 퀄리티 {QUALITY_WEIGHT*100:.0f}%")
    print("="*70)
    
    # 1. 대상 종목 리스트 구성
    target_tickers = {}
    financial_data = {}
    
    try:
        print("\n📊 STEP 1: 종목 리스트 + 재무데이터 확보 중...")
        
        # KOSPI
        df_kospi = fdr.StockListing('KOSPI')
        top_kospi = df_kospi.sort_values('Marcap', ascending=False).head(100)
        
        # KOSDAQ
        df_kosdaq = fdr.StockListing('KOSDAQ')
        top_kosdaq = df_kosdaq.sort_values('Marcap', ascending=False).head(100)
        
        # 재무 데이터 저장
        for _, row in pd.concat([top_kospi, top_kosdaq]).iterrows():
            name = row['Name']
            target_tickers[name] = row['Code']
            financial_data[name] = {
                'PER': row.get('PER', np.nan),
                'PBR': row.get('PBR', np.nan),
                'ROE': row.get('ROE', np.nan),
                'DIV': row.get('DivYield', 0),
                'Marcap': row.get('Marcap', 0)
            }
        
        # 달러 선물
        target_tickers['KODEX 미국달러선물'] = '261240'
        financial_data['KODEX 미국달러선물'] = {
            'PER': np.nan, 'PBR': np.nan, 'ROE': 0, 'DIV': 0, 'Marcap': 0
        }
        
        print(f"   ✅ 총 {len(target_tickers)}개 종목 확보 (재무데이터 포함)")

    except Exception as e:
        send_telegram(f"❌ 종목 리스트 확보 실패: {e}")
        return

    # 2. 데이터 다운로드
    print("\n📈 STEP 2: 가격 데이터 다운로드 중...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    kospi = None
    raw_data = pd.DataFrame()
    
    try:
        # KOSPI 지수
        kospi_df = fdr.DataReader('KS11', start=start_date, end=end_date)
        kospi = kospi_df['Close'].ffill()

        # 개별 종목 데이터 수집
        df_list = []
        total_count = len(target_tickers)
        
        for i, (name, code) in enumerate(target_tickers.items()):
            if i % 20 == 0: 
                print(f"   진행: {i}/{total_count} ({i/total_count*100:.1f}%)")
            
            try:
                df = fdr.DataReader(code, start=start_date, end=end_date)
                if df.empty or len(df) < 120: 
                    continue
                
                series = df['Close'].rename(name)
                df_list.append(series)
            except:
                continue
            
            time.sleep(0.05)
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).fillna(method='ffill', limit=5)
            print(f"   ✅ {len(raw_data.columns)}개 종목 데이터 준비 완료")
        else:
            raise Exception("유효한 데이터를 하나도 가져오지 못했습니다.")

    except Exception as e:
        send_telegram(f"❌ 데이터 다운로드 실패: {e}")
        return

    # 3. 멀티팩터 점수 계산
    print("\n🧮 STEP 3: 멀티팩터 점수 계산 중...")
    
    try:
        # 3-1. 모멘텀 팩터
        daily_rets = raw_data.pct_change()
        ret_6m = raw_data.pct_change(120).iloc[-1]
        vol_6m = daily_rets.rolling(120).std().iloc[-1]
        
        epsilon = 1e-6
        momentum_score = ret_6m / (vol_6m + epsilon)
        
        # 3-2. 밸류 팩터
        value_scores = {}
        for name in raw_data.columns:
            if name == 'KODEX 미국달러선물':
                value_scores[name] = 0
                continue
            
            fin = financial_data.get(name, {})
            per = fin.get('PER', np.nan)
            pbr = fin.get('PBR', np.nan)
            
            score = 0
            if pd.notna(per) and 0 < per < 30:
                score += 1 / per
            if pd.notna(pbr) and 0 < pbr < 3:
                score += 1 / pbr
            
            value_scores[name] = score
        
        value_score = pd.Series(value_scores)
        
        # 3-3. 퀄리티 팩터
        quality_scores = {}
        for name in raw_data.columns:
            if name == 'KODEX 미국달러선물':
                quality_scores[name] = 0
                continue
            
            fin = financial_data.get(name, {})
            roe = fin.get('ROE', 0)
            per = fin.get('PER', np.nan)
            
            score = 0
            if roe > 15:
                score += 2
            elif roe > 10:
                score += 1
            
            if pd.notna(per) and 5 < per < 20:
                score += 1
            
            quality_scores[name] = score
        
        quality_score = pd.Series(quality_scores)
        
        # 3-4. 저변동성 팩터
        if VOLATILITY_WEIGHT > 0:
            vol_score = 1 / (vol_6m + epsilon)
        else:
            vol_score = pd.Series(index=raw_data.columns, data=0)
        
        # 3-5. 정규화
        def normalize(series):
            if series.std() == 0:
                return series
            return (series - series.min()) / (series.max() - series.min())
        
        mom_norm = normalize(momentum_score.reindex(raw_data.columns).fillna(0))
        val_norm = normalize(value_score.reindex(raw_data.columns).fillna(0))
        qual_norm = normalize(quality_score.reindex(raw_data.columns).fillna(0))
        vol_norm = normalize(vol_score.reindex(raw_data.columns).fillna(0))
        
        # 3-6. 종합 점수
        total_score = (
            mom_norm * MOMENTUM_WEIGHT +
            val_norm * VALUE_WEIGHT +
            qual_norm * QUALITY_WEIGHT +
            vol_norm * VOLATILITY_WEIGHT
        )
        
        print(f"   ✅ 멀티팩터 점수 계산 완료")
        
        # 시장 판단
        kospi_ma60 = kospi.rolling(window=60).mean().iloc[-1]
        current_kospi = kospi.iloc[-1]
        
        if hasattr(current_kospi, 'item'): 
            current_kospi = current_kospi.item()
        if hasattr(kospi_ma60, 'item'): 
            kospi_ma60 = kospi_ma60.item()
        
        is_bull_market = current_kospi > kospi_ma60

    except Exception as e:
        send_telegram(f"❌ 지표 계산 중 오류: {e}")
        return

    # 4. 종목 선정
    print("\n🎯 STEP 4: 종목 선정 중...")
    
    final_targets = []
    reason = ""
    top_10_info = []  # 상위 10개 종목 정보

    if is_bull_market:
        scores = total_score.drop('KODEX 미국달러선물', errors='ignore')
        sorted_scores = scores.sort_values(ascending=False)
        
        # 상위 10개 정보 저장 (메시지용)
        for rank, (name, score) in enumerate(sorted_scores.head(10).items(), 1):
            top_10_info.append({
                'rank': rank,
                'name': name,
                'total_score': score,
                'momentum': mom_norm.get(name, 0),
                'value': val_norm.get(name, 0),
                'quality': qual_norm.get(name, 0),
                'volatility': vol_norm.get(name, 0)
            })
        
        if sorted_scores.empty or sorted_scores.iloc[0] <= 0:
            final_targets = [('KODEX 미국달러선물', 1.0)]
            reason = "주도주 부재 → 달러 방어"
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
                final_targets = [('KODEX 미국달러선물', 1.0)]
                reason = "대상 종목 없음 → 달러 방어"
    else:
        final_targets = [('KODEX 미국달러선물', 1.0)]
        reason = "하락장 방어 (코스피 < 60일선)"
        
        # 하락장에서도 참고용으로 상위 10개 표시
        scores = total_score.drop('KODEX 미국달러선물', errors='ignore')
        sorted_scores = scores.sort_values(ascending=False)
        
        for rank, (name, score) in enumerate(sorted_scores.head(10).items(), 1):
            top_10_info.append({
                'rank': rank,
                'name': name,
                'total_score': score,
                'momentum': mom_norm.get(name, 0),
                'value': val_norm.get(name, 0),
                'quality': qual_norm.get(name, 0),
                'volatility': vol_norm.get(name, 0)
            })

    print(f"   ✅ 종목 선정 완료: {len(final_targets)}개")

    # 5. 메시지 생성
    print("\n📱 STEP 5: 메시지 생성 중...")
    
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    # 메시지 헤더
    msg = "━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🎯 멀티팩터 퀀트 전략\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📅 {today_dt.strftime('%Y-%m-%d %H:%M')}\n"
    msg += f"🔧 팩터구성: M{MOMENTUM_WEIGHT*100:.0f}% V{VALUE_WEIGHT*100:.0f}% Q{QUALITY_WEIGHT*100:.0f}%\n"
    msg += f"📊 시장: {'🔴 상승장 (매수)' if is_bull_market else '🔵 하락장 (방어)'}\n"
    msg += f"💡 전략: {reason}\n\n"
    
    # 리밸런싱 상태
    if is_rebalance_period:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "🔔 [리밸런싱 주간]\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    else:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "☕ [관망 모드 - 참고용]\n"
        msg += f"⏰ 다음 리밸런싱: {next_rebalance_date.strftime('%m월 %d일')}\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 선정 종목 (매수 대상)
    if final_targets[0][0] != 'KODEX 미국달러선물':
        msg += f"✅ 선정 종목 ({len(final_targets)}개)\n"
        msg += "─────────────────────\n"
        
        for idx, (name, weight) in enumerate(final_targets, 1):
            # 종합 점수
            total_s = total_score.get(name, 0)
            
            # 세부 점수
            mom_s = mom_norm.get(name, 0)
            val_s = val_norm.get(name, 0)
            qual_s = qual_norm.get(name, 0)
            
            # 순위 찾기
            rank = next((item['rank'] for item in top_10_info if item['name'] == name), '-')
            
            # 점수에 따른 이모지
            if total_s >= 0.8:
                emoji = "🔥🔥"
            elif total_s >= 0.6:
                emoji = "🔥"
            elif total_s >= 0.4:
                emoji = "⭐"
            else:
                emoji = "✓"
            
            if name in raw_data.columns:
                current_price = raw_data[name].iloc[-1]
                buy_budget = MY_TOTAL_ASSETS * weight
                buy_qty = int(buy_budget // current_price)
                
                msg += f"{idx}. {name} {emoji}\n"
                msg += f"   순위: {rank}위 | 점수: {total_s:.3f}\n"
                msg += f"   M{mom_s:.2f} V{val_s:.2f} Q{qual_s:.2f}\n"
                msg += f"   비중: {int(weight*100)}% | {buy_qty}주\n"
                msg += f"   가격: {int(current_price):,}원\n\n"
    else:
        msg += "🛡️ 방어 자산\n"
        msg += "─────────────────────\n"
        msg += "• KODEX 미국달러선물 (100%)\n\n"
    
    # 참고: 상위 10개 종목 순위
    if top_10_info:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "📊 종합 순위 TOP 10\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for info in top_10_info[:10]:
            selected_mark = "👉" if any(t[0] == info['name'] for t in final_targets) else "  "
            msg += f"{selected_mark}{info['rank']:2d}위 {info['name'][:8]}\n"
            msg += f"     점수 {info['total_score']:.3f} "
            msg += f"(M{info['momentum']:.2f} V{info['value']:.2f} Q{info['quality']:.2f})\n"
    
    msg += "\n━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 총 투자금: {MY_TOTAL_ASSETS:,}원\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━"

    print("\n" + msg)
    send_telegram(msg)
    
    print("\n" + "="*70)
    print("✅ 분석 완료!")
    print("="*70)


if __name__ == "__main__":
    get_todays_signal()
