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


def get_todays_signal():
    print("="*70)
    print("📊 한국 ETF 가중모멘텀 전략 신호 생성기")
    print("="*70)
    print("⏳ 데이터 분석 중...")
    
    # 1. 데이터 준비
    etf_tickers = {
        'KODEX 200': '069500',
        'KODEX 미국나스닥100TR': '379810',
        'ACE 미국S&P500': '360200',
        'KODEX 반도체': '091160',
        'KODEX 헬스케어': '266420',
        'KODEX 미국달러선물': '261240',
        'KODEX AI전력핵심설비': '487240',
        'ACE 구글벨류체인액티브': '483340',
        'PLUS K방산': '449170',
        'KODEX 미국30년국채액티브(H)': '484790',
        'KODEX 코스닥150': '229200',
    }
    
    end_date = datetime.now().strftime("%Y-%m-%d")
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
            time.sleep(0.1)
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).ffill().dropna()
            print(f"✅ {len(raw_data.columns)}개 ETF 데이터 수집 완료")
        else:
            raise Exception("데이터 수집 실패")

    except Exception as e:
        send_telegram(f"❌ 오류 발생: {e}")
        print(f"분석 실패: {e}")
        return

    # 2. 가중 평균 모멘텀 계산
    mom_1m = raw_data.pct_change(20).iloc[-1]
    mom_3m = raw_data.pct_change(60).iloc[-1]
    mom_6m = raw_data.pct_change(120).iloc[-1]

    weighted_score = ((mom_1m.fillna(0) * 0.3) + (mom_3m.fillna(0) * 0.3) + (mom_6m.fillna(0) * 0.4))

    # 시장 타이밍 (코스피 120일선)
    kospi_ma120 = kospi.rolling(window=120).mean().iloc[-1]
    current_kospi = kospi.iloc[-1]
    
    if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
    if hasattr(kospi_ma120, 'item'): kospi_ma120 = kospi_ma120.item()

    is_bull_market = current_kospi > kospi_ma120
    
    print(f"✅ 시장 판단: {'🔴 상승장' if is_bull_market else '🔵 하락장'}")

    # 3. 목표 종목 선정 (TOP 2 분산)
    final_targets = []
    reason = ""
    all_rankings = []  # 전체 순위 저장

    if is_bull_market:
        scores = weighted_score.drop('KODEX 미국달러선물', errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        # 전체 순위 저장 (메시지용)
        for rank, (name, score) in enumerate(top_assets.items(), 1):
            all_rankings.append({
                'rank': rank,
                'name': name,
                'score': score,
                'price': raw_data[name].iloc[-1]
            })
        
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [('KODEX 미국달러선물', 1.0)]
            reason = "주도주 부재 → 달러 방어"
        else:
            selected = []
            for name, score in top_assets.items():
                if score > 0:
                    selected.append(name)
                if len(selected) >= 2: break
            
            if len(selected) == 1:
                final_targets = [(selected[0], 1.0)]
                reason = f"단독 주도주"
            else:
                final_targets = [(selected[0], 0.5), (selected[1], 0.5)]
                reason = f"TOP 2 분산"
    else:
        # 하락장에도 순위는 보여주기
        scores = weighted_score.drop('KODEX 미국달러선물', errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        for rank, (name, score) in enumerate(top_assets.items(), 1):
            all_rankings.append({
                'rank': rank,
                'name': name,
                'score': score,
                'price': raw_data[name].iloc[-1]
            })
        
        final_targets = [('KODEX 미국달러선물', 1.0)]
        reason = "하락장 방어 (KOSPI < MA120)"

    # 4. 메시지 생성 (HTML 포맷)
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    # HTML 포맷으로 메시지 작성
    msg = f"<b>🇰🇷 한국 ETF 가중모멘텀 전략 [{today_dt.strftime('%Y-%m-%d')}]</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 전략 정보
    msg += f"📊 <b>전략 구성</b>\n"
    msg += f"  • 1개월: 30%\n"
    msg += f"  • 3개월: 30%\n"
    msg += f"  • 6개월: 40%\n"
    msg += f"  • 보유: TOP 2 분산\n\n"
    
    # 시장 상태
    kospi_change = ((current_kospi - kospi_ma120) / kospi_ma120) * 100
    msg += f"📈 <b>시장 상태</b>\n"
    msg += f"  • KOSPI: {current_kospi:,.2f}\n"
    msg += f"  • MA120: {kospi_ma120:,.2f}\n"
    msg += f"  • 시장: {'🔴 상승장' if is_bull_market else '🔵 하락장'} ({kospi_change:+.1f}%)\n\n"
    
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 리밸런싱 여부에 따라 메시지 구분
    if is_rebalance_period:
        msg += f"🔔 <b>리밸런싱 주간</b>\n"
        msg += f"사유: {reason}\n\n"
        
        msg += f"💼 <b>매수 종목 ({len(final_targets)}개)</b>\n\n"
        
        for name, weight in final_targets:
            # 점수 가져오기
            try:
                current_score = weighted_score[name]
            except:
                current_score = 0.0
            
            # 점수에 따른 이모지
            if current_score >= 0.15:
                emoji = "🔥🔥"
            elif current_score >= 0.08:
                emoji = "🔥"
            elif current_score > 0:
                emoji = "⭐"
            else:
                emoji = "🛡️"
            
            # 순위 찾기
            rank = next((r['rank'] for r in all_rankings if r['name'] == name), '-')
            
            current_price = raw_data[name].iloc[-1] if name in raw_data.columns else 0
            buy_budget = MY_TOTAL_ASSETS * weight
            buy_qty = int(buy_budget // current_price) if current_price > 0 else 0
            
            if name == 'KODEX 미국달러선물':
                msg += f"<b>🛡️ {name}</b>\n"
                msg += f"  • 비중: {weight*100:.0f}%\n"
                msg += f"  • 사유: {reason}\n\n"
            else:
                msg += f"<b>{rank}위. {name}</b> {emoji}\n"
                msg += f"  • 가격: {current_price:,.0f}원 | 수량: {buy_qty}주\n"
                msg += f"  • 비중: {weight*100:.0f}% ({int(buy_budget):,}원)\n"
                msg += f"  • 점수: {current_score:.3f}\n\n"
    
    else:
        msg += f"☕ <b>관망 모드</b>\n"
        msg += f"다음 리밸런싱: {next_rebalance_date.strftime('%m월 %d일')}\n\n"
        
        msg += f"📋 <b>현재 순위 (달러 제외)</b>\n\n"
        
        # 상위 5개는 상세, 나머지는 간략
        for info in all_rankings:
            rank = info['rank']
            name = info['name']
            score = info['score']
            price = info['price']
            
            if rank <= 5:
                # 점수에 따른 이모지
                if score >= 0.15:
                    emoji = "🔥🔥"
                elif score >= 0.08:
                    emoji = "🔥"
                elif score > 0:
                    emoji = "⭐"
                else:
                    emoji = "💤"
                
                msg += f"<b>{rank}위. {name}</b> {emoji}\n"
                msg += f"  • 점수: {score:.3f}\n"
                msg += f"  • 가격: {price:,.0f}원\n\n"
            else:
                # 6위 이하는 간략하게
                msg += f"{rank}위. {name} ({score:.3f})\n"
    
    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"<i>투자 원금: {MY_TOTAL_ASSETS:,}원</i>"
    
    # 콘솔 출력 (HTML 태그 제거 버전)
    print("\n" + "="*70)
    print("메시지 미리보기:")
    print("="*70)
    import re
    clean_msg = re.sub('<.*?>', '', msg)
    print(clean_msg)
    print("="*70)
    
    send_telegram(msg)


if __name__ == "__main__":
    get_todays_signal()
