import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import itertools

# =========================================================
# [1. 설정 영역]
# =========================================================
# 백테스트 기간 설정 (최근 1년 예시)
# 주의: 기간이 길수록 KRX 크롤링 시간이 오래 걸립니다.
START_DATE = (datetime.now() - relativedelta(months=36)).strftime("%Y%m%d")
END_DATE = datetime.now().strftime("%Y%m%d")

# 비교할 파라미터 조합 (이 범위를 조합해서 테스트함)
# 예: PER가 0~10인 경우, 0~15인 경우 등을 다 테스트
PER_RANGES = [(0, 10), (0, 20), (5, 30)] 
PBR_RANGES = [(0, 1.0), (0, 1.5), (0, 3.0)]
ROE_MINS = [0, 5, 10] # ROE n% 이상

TOP_N = 5 # 포트폴리오 종목 수
# =========================================================


# =========================================================
# [2. 데이터 수집 엔진] (KRX 크롤링 + FDR 주가)
# =========================================================
def get_krx_fundamental_snapshot(target_date):
    """특정 일자의 KRX 펀더멘털 전체 스냅샷"""
    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd',
    }
    
    # 휴일일 경우 평일 찾기
    dt = datetime.strptime(target_date, "%Y%m%d")
    for i in range(5):
        check_date = (dt - timedelta(days=i)).strftime("%Y%m%d")
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
            res = r.json()
            if 'output' in res and len(res['output']) > 0:
                df = pd.DataFrame(res['output'])
                df = df.rename(columns={
                    'ISU_SRT_CD': 'Code', 'ISU_ABBRV': 'Name',
                    'PER': 'PER', 'PBR': 'PBR', 'EPS': 'EPS', 'BPS': 'BPS', 'DVD_YLD': 'DivYield'
                })
                # 숫자 변환
                cols = ['PER', 'PBR', 'EPS', 'BPS', 'DivYield']
                for c in cols:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').replace('-', np.nan), errors='coerce')
                
                # ROE 계산
                df['ROE'] = df.apply(lambda x: (x['EPS']/x['BPS']*100) if (pd.notnull(x['BPS']) and x['BPS']>0) else 0, axis=1)
                
                print(f"   ✅ [데이터 확보] {check_date} 펀더멘털 데이터 수집 완료")
                return df.set_index('Code'), check_date
        except:
            continue
    return pd.DataFrame(), target_date

def collect_historical_data():
    """백테스트용 과거 데이터셋 생성 (가장 오래 걸리는 작업)"""
    print("="*60)
    print("🚀 [Phase 1] 과거 데이터 수집 시작 (한 번만 실행됩니다)")
    print("="*60)
    
    # 1. 리밸런싱 날짜 생성 (매월 말일)
    rebalance_dates = []
    curr = datetime.strptime(START_DATE, "%Y%m%d")
    end = datetime.strptime(END_DATE, "%Y%m%d")
    
    while curr <= end:
        rebalance_dates.append(curr.strftime("%Y%m%d"))
        curr += relativedelta(months=1)
    
    # 2. 월별 펀더멘털 데이터 수집
    fundamental_cache = {} # { '20230131': df, ... }
    all_codes = set()
    
    for date in rebalance_dates:
        df, valid_date = get_krx_fundamental_snapshot(date)
        if not df.empty:
            fundamental_cache[valid_date] = df
            all_codes.update(df.index.tolist())
            time.sleep(0.5) # 차단 방지
            
    # 3. 주가 데이터 수집 (한 번에)
    # 전체 종목을 다 받으면 너무 느리므로, 각 월별 시총 상위 500개 합집합만 받음
    print(f"\n📊 [가격 데이터 다운로드] 대상 종목 수 계산 중...")
    target_universe = set()
    for date, df in fundamental_cache.items():
        # PBR 등이 있는 종목 중 일부만 샘플링 (속도 최적화 위해)
        # 실제로는 전체를 다 받아야 정확하지만, 여기선 데모용으로 각 월별 상위 300개만 추적
        if 'BPS' in df.columns:
             # 시가총액 대용으로 BPS*상장주식수 대신 간단히 PBR, PER 있는것 위주
             valid_df = df.dropna(subset=['PER', 'PBR'])
             target_universe.update(valid_df.index.tolist())
    
    print(f"   👉 총 {len(target_universe)}개 종목의 가격 데이터를 수집합니다 (시간 소요됨)")
    
    price_cache = {} # { 'Code': Series(Close Price) }
    
    # fdr은 다수 종목 동시 다운로드가 안되므로 루프 (속도 개선을 위해 상위 일부만 하는게 좋음)
    # 여기서는 진행률을 보여줌
    count = 0
    for code in list(target_universe):
        try:
            # 전체 기간 한번에 다운로드
            df_p = fdr.DataReader(code, START_DATE, END_DATE)
            if not df_p.empty:
                price_cache[code] = df_p['Close']
        except:
            pass
        
        count += 1
        if count % 100 == 0:
            print(f"   ... {count}/{len(target_universe)} 완료")
            
    print("\n✅ 데이터 수집 완료! 이제 시뮬레이션을 반복할 수 있습니다.")
    return fundamental_cache, price_cache

# =========================================================
# [3. 백테스트 시뮬레이터] (고속 반복용)
# =========================================================
def run_simulation(fund_cache, price_cache, per_rng, pbr_rng, min_roe):
    """파라미터를 받아 수익률을 계산하는 함수"""
    
    dates = sorted(fund_cache.keys())
    total_capital = 1.0 # 수익률 계산용 (1.0 = 100%)
    
    # 월별 수익률 기록
    log_returns = []
    
    for i in range(len(dates) - 1):
        buy_date = dates[i]
        sell_date = dates[i+1]
        
        # 1. 종목 선정
        df = fund_cache[buy_date]
        
        # 조건 필터링
        mask = (df['PER'] >= per_rng[0]) & (df['PER'] <= per_rng[1]) & \
               (df['PBR'] >= pbr_rng[0]) & (df['PBR'] <= pbr_rng[1]) & \
               (df['ROE'] >= min_roe)
               
        candidates = df[mask]
        
        # 순위 매기기 (여기서는 단순하게 PBR+PER 낮은 순 합산 등 전략 적용 가능)
        # 예시: 밸류 점수 (1/PER + 1/PBR) 높은 순
        if not candidates.empty:
            candidates['Score'] = (1/candidates['PER'].replace(0, np.inf)) + (1/candidates['PBR'].replace(0, np.inf))
            # Quality 가중 (ROE)
            candidates['Score'] += (candidates['ROE'] / 100)
            
            portfolio = candidates.sort_values(by='Score', ascending=False).head(TOP_N).index.tolist()
        else:
            portfolio = []
            
        # 2. 수익률 계산
        period_return = 0
        if not portfolio:
            period_return = 0.0 # 보유 종목 없음 (현금 보유)
        else:
            sum_ret = 0
            count_valid = 0
            for code in portfolio:
                if code in price_cache:
                    prices = price_cache[code]
                    try:
                        # 매수일 종가
                        buy_price = prices.asof(buy_date)
                        # 매도일 종가
                        sell_price = prices.asof(sell_date)
                        
                        if not np.isnan(buy_price) and not np.isnan(sell_price) and buy_price > 0:
                            ret = (sell_price - buy_price) / buy_price
                            sum_ret += ret
                            count_valid += 1
                    except:
                        pass
            
            if count_valid > 0:
                period_return = sum_ret / count_valid
            else:
                period_return = 0
                
        total_capital = total_capital * (1 + period_return)
        log_returns.append(period_return)
        
    final_return = (total_capital - 1) * 100
    return final_return

# =========================================================
# [4. 메인 실행부]
# =========================================================
if __name__ == "__main__":
    # 1. 데이터 수집 (최초 1회 수행 - 시간 소요)
    # ⚠️ 이미 데이터를 받았다면 이 줄을 주석처리 하고 변수만 재사용 가능
    global_fund_cache, global_price_cache = collect_historical_data()
    
    print("\n" + "="*60)
    print("🧪 [Phase 2] 파라미터 최적화 (Grid Search) 시작")
    print("="*60)
    
    results = []
    
    # 모든 파라미터 조합 생성
    combinations = list(itertools.product(PER_RANGES, PBR_RANGES, ROE_MINS))
    
    print(f"총 {len(combinations)}개 시나리오 테스트 중...\n")
    print(f"{'PER 범위':<15} | {'PBR 범위':<15} | {'ROE 이상':<10} | {'누적수익률':<10}")
    print("-" * 65)

    for per_r, pbr_r, roe_min in combinations:
        try:
            # 백테스트 실행 (메모리에 있는 데이터 사용하므로 매우 빠름)
            ret = run_simulation(global_fund_cache, global_price_cache, per_r, pbr_r, roe_min)
            
            results.append({
                'PER': f"{per_r[0]}~{per_r[1]}",
                'PBR': f"{pbr_r[0]}~{pbr_r[1]}",
                'ROE': f"{roe_min}%↑",
                'Return': ret
            })
            
            print(f"{str(per_r):<15} | {str(pbr_r):<15} | {roe_min:<9} | {ret:>.2f}%")
        except Exception as e:
            print(f"Error: {e}")

    # 결과 정리
    print("-" * 65)
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        best = df_res.sort_values(by='Return', ascending=False).iloc[0]
        print(f"\n🏆 [최고 수익률 전략]")
        print(f"   조건: PER {best['PER']}, PBR {best['PBR']}, ROE {best['ROE']}")
        print(f"   수익률: {best['Return']:.2f}%")