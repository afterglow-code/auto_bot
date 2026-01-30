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
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {'chat_id': CHAT_ID, 'text': msg}
    try: requests.get(url, params=params)
    except: pass

# =========================================================
# [핵심] KRX 펀더멘털 데이터 수집 (전종목 한번에)
# =========================================================
def get_krx_fundamental_df():
    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    }
    
    # 최근 영업일 데이터 확보를 위한 루프
    for i in range(5):
        check_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        
        # BLD: MDCSTAT03501 (PER, PBR, 배당 등)
        data = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT03501',
            'locale': 'ko_KR',
            'searchType': '1',
            'mktId': 'ALL',
            'trdDd': check_date,
            'share': '1', 'money': '1', 'csvxls_isNo': 'false',
        }
        
        try:
            r = requests.post(url, data=data, headers=headers)
            res_json = r.json()
            if 'output' in res_json and len(res_json['output']) > 0:
                print(f"   ✅ KRX 펀더멘털 데이터 확보 (기준일: {check_date})")
                df = pd.DataFrame(res_json['output'])
                
                # 필요한 컬럼만 정리 및 인덱스 설정
                # ISU_SRT_CD: 종목코드, PER, PBR, EPS, BPS, DVD_YLD
                df = df.rename(columns={
                    'ISU_SRT_CD': 'Code', 
                    'PER': 'PER', 'PBR': 'PBR', 
                    'EPS': 'EPS', 'BPS': 'BPS', 'DVD_YLD': 'DivYield'
                })
                
                # 숫자형 변환 (콤마 제거 및 - 처리)
                cols = ['PER', 'PBR', 'EPS', 'BPS', 'DivYield']
                for col in cols:
                    df[col] = df[col].astype(str).str.replace(',', '').replace('-', np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Code를 인덱스로 설정하여 검색 속도 향상
                return df.set_index('Code')
                
        except Exception as e:
            print(f"   ⚠️ 데이터 요청 실패 ({check_date}): {e}")
            continue
            
    raise Exception("KRX 데이터를 가져오지 못했습니다.")

def get_todays_signal():
    print("="*70)
    print("🎯 [멀티팩터 퀀트 전략] 데이터 분석 시작")
    print(f"   모멘텀 {MOMENTUM_WEIGHT*100:.0f}% | 밸류 {VALUE_WEIGHT*100:.0f}% | 퀄리티 {QUALITY_WEIGHT*100:.0f}%")
    print("="*70)
    
    target_tickers = {} # {종목명: 코드}
    financial_data = {} # {종목명: {PER:..., PBR:...}}
    
    try:
        print("\n📊 STEP 1: 종목 리스트 구성 (fdr + KRX)")
        
        # 1. fdr로 전체 종목 리스트 가져오기 (시가총액 포함됨)
        df_master = fdr.StockListing('KRX')
        
        # 2. 시가총액(Marcap) 상위 200개 자르기
        # Marcap 컬럼이 문자열일 수도 있으므로 숫자로 변환
        if df_master['Marcap'].dtype == object:
             df_master['Marcap'] = pd.to_numeric(df_master['Marcap'], errors='coerce')
             
        top_200 = df_master.sort_values(by='Marcap', ascending=False).head(200)
        print(f"   ✅ fdr 시총 상위 200개 확보 완료")

        # 3. KRX 펀더멘털 데이터 가져오기 (전 종목)
        df_fund = get_krx_fundamental_df()

        # 4. 데이터 매핑 (fdr 리스트에 KRX 데이터 붙이기)
        for _, row in top_200.iterrows():
            code = row['Code']
            name = row['Name']
            
            # KRX 데이터에서 해당 코드 정보 찾기
            try:
                fund_info = df_fund.loc[code] # Index가 Code임
                
                per = fund_info['PER']
                pbr = fund_info['PBR']
                eps = fund_info['EPS']
                bps = fund_info['BPS']
                div = fund_info['DivYield']
                
                # ROE 직접 계산
                roe = 0
                if pd.notna(eps) and pd.notna(bps) and bps > 0:
                    roe = (eps / bps) * 100
                
            except KeyError:
                # KRX 데이터에 없는 종목 (스팩, 신규상장 등)은 NaN 처리
                per = np.nan
                pbr = np.nan
                roe = 0
                div = 0
            
            # 최종 데이터 저장
            target_tickers[name] = code
            financial_data[name] = {
                'PER': per,
                'PBR': pbr,
                'ROE': roe,
                'DIV': div,
                'Marcap': row['Marcap']
            }

        # 달러 선물 (ETF) 추가
        target_tickers['KODEX 미국달러선물'] = '261240'
        financial_data['KODEX 미국달러선물'] = {
            'PER': np.nan, 'PBR': np.nan, 'ROE': 0, 'DIV': 0, 'Marcap': 0
        }
        
        print(f"   ✅ 최종 분석 대상: {len(target_tickers)}개 종목 준비 완료")
        
        # 데이터 검증 출력
        first_stock = list(target_tickers.keys())[0]
        print(f"   🔎 데이터 샘플 ({first_stock}): {financial_data[first_stock]}")

    except Exception as e:
        send_telegram(f"❌ 종목 리스트 확보 실패: {e}")
        print(f"Error detail: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 데이터 다운로드 (가격)
    print("\n📈 STEP 2: 가격 데이터 다운로드 중 (FinanceDataReader)...")
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
            if i % 50 == 0: 
                print(f"   진행: {i}/{total_count} ({i/total_count*100:.1f}%)")
            
            try:
                df = fdr.DataReader(code, start=start_date, end=end_date)
                if df.empty or len(df) < 120: continue
                series = df['Close'].rename(name)
                df_list.append(series)
            except: continue
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
        # 3-1. 모멘텀
        daily_rets = raw_data.pct_change()
        ret_6m = raw_data.pct_change(120).iloc[-1]
        vol_6m = daily_rets.rolling(120).std().iloc[-1]
        epsilon = 1e-6
        momentum_score = ret_6m / (vol_6m + epsilon)
        
        # 3-2. 밸류
        value_scores = {}
        for name in raw_data.columns:
            if name == 'KODEX 미국달러선물':
                value_scores[name] = 0
                continue
            fin = financial_data.get(name, {})
            per = fin.get('PER', np.nan)
            pbr = fin.get('PBR', np.nan)
            score = 0
            if pd.notna(per) and 0 < per < 50: score += 1 / per
            if pd.notna(pbr) and 0 < pbr < 3: score += 1 / pbr
            value_scores[name] = score
        value_score = pd.Series(value_scores)
        
        # 3-3. 퀄리티
        quality_scores = {}
        for name in raw_data.columns:
            if name == 'KODEX 미국달러선물':
                quality_scores[name] = 0
                continue
            fin = financial_data.get(name, {})
            roe = fin.get('ROE', 0)
            score = 0
            if roe > 15: score += 2
            elif roe > 10: score += 1
            quality_scores[name] = score
        quality_score = pd.Series(quality_scores)
        
        # 3-4. 저변동성
        if VOLATILITY_WEIGHT > 0:
            vol_score = 1 / (vol_6m + epsilon)
        else:
            vol_score = pd.Series(index=raw_data.columns, data=0)
        
        # 3-5. 정규화 및 합산
        def normalize(series):
            if series.std() == 0: return series
            return (series - series.min()) / (series.max() - series.min())
        
        total_score = (
            normalize(momentum_score.reindex(raw_data.columns).fillna(0)) * MOMENTUM_WEIGHT +
            normalize(value_score.reindex(raw_data.columns).fillna(0)) * VALUE_WEIGHT +
            normalize(quality_score.reindex(raw_data.columns).fillna(0)) * QUALITY_WEIGHT +
            normalize(vol_score.reindex(raw_data.columns).fillna(0)) * VOLATILITY_WEIGHT
        )
        
        print(f"   ✅ 멀티팩터 점수 계산 완료")
        
        # 시장 판단
        kospi_ma60 = kospi.rolling(window=60).mean().iloc[-1]
        current_kospi = kospi.iloc[-1]
        if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
        if hasattr(kospi_ma60, 'item'): kospi_ma60 = kospi_ma60.item()
        is_bull_market = current_kospi > kospi_ma60

    except Exception as e:
        send_telegram(f"❌ 지표 계산 중 오류: {e}")
        return

    # 4. 종목 선정
    print("\n🎯 STEP 4: 종목 선정 중...")
    
    final_targets = []
    reason = ""
    top_10_info = []

    if is_bull_market:
        scores = total_score.drop('KODEX 미국달러선물', errors='ignore')
        sorted_scores = scores.sort_values(ascending=False)
        
        for rank, (name, score) in enumerate(sorted_scores.head(10).items(), 1):
            top_10_info.append({
                'rank': rank, 'name': name, 'total_score': score, 
                'm': momentum_score.get(name), 'v': value_score.get(name), 'q': quality_score.get(name)
            })
        
        if sorted_scores.empty or sorted_scores.iloc[0] <= 0:
            final_targets = [('KODEX 미국달러선물', 1.0)]
            reason = "주도주 부재 → 달러 방어"
        else:
            selected = []
            for name, score in sorted_scores.items():
                if score > 0: selected.append(name)
                if len(selected) >= NUM_STOCKS: break
            
            if selected:
                weight = 1.0 / len(selected)
                for s in selected: final_targets.append((s, weight))
                reason = f"멀티팩터 TOP {len(selected)}"
            else:
                final_targets = [('KODEX 미국달러선물', 1.0)]
                reason = "대상 종목 없음 → 달러 방어"
    else:
        final_targets = [('KODEX 미국달러선물', 1.0)]
        reason = "하락장 방어 (코스피 < 60일선)"
        scores = total_score.drop('KODEX 미국달러선물', errors='ignore')
        sorted_scores = scores.sort_values(ascending=False)
        for rank, (name, score) in enumerate(sorted_scores.head(10).items(), 1):
            top_10_info.append({'rank': rank, 'name': name, 'total_score': score})

    print(f"   ✅ 종목 선정 완료: {len(final_targets)}개")

    # 5. 메시지 생성 및 전송
    msg = "━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🎯 멀티팩터 퀀트 전략\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    msg += f"📊 시장: {'🔴 상승장 (매수)' if is_bull_market else '🔵 하락장 (방어)'}\n"
    msg += f"💡 전략: {reason}\n\n"
    
    if final_targets[0][0] != 'KODEX 미국달러선물':
        msg += f"✅ 선정 종목 ({len(final_targets)}개)\n"
        msg += "─────────────────────\n"
        for idx, (name, weight) in enumerate(final_targets, 1):
            price = raw_data[name].iloc[-1]
            buy_qty = int((MY_TOTAL_ASSETS * weight) // price)
            msg += f"{idx}. {name} ({int(price):,}원)\n"
            msg += f"   비중: {int(weight*100)}% | {buy_qty}주\n\n"
    else:
        msg += "🛡️ 방어 자산: KODEX 미국달러선물 (100%)\n\n"
        
    if top_10_info:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "📊 종합 순위 TOP 5\n"
        for info in top_10_info[:5]:
             msg += f"{info['rank']}위 {info['name']} ({info['total_score']:.2f})\n"

    print("\n" + msg)
    send_telegram(msg)
    print("="*70 + "\n✅ 분석 완료!\n" + "="*70)

if __name__ == "__main__":
    get_todays_signal()