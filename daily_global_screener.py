# daily_global_screener.py
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import concurrent.futures
from tqdm import tqdm
import os
import pickle
import time
import random

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)

# =========================================================
# 1. 백테스트 config 설정 이식 (내부 통합)
# =========================================================
PARAMS = {
    'STOCK_KR': {
        'NAME': '한국 개별주 가속 모멘텀',
        'LISTING': 'KRX',
        'PASSIVE': {
            'MOMENTUM_SHORT': 60,
            'MOMENTUM_LONG': 120,
            'VOLATILITY_WINDOW': 60,
        },
    },
    'STOCK_US': {
        'NAME': '미국 주식 가속 모멘텀',
        'LISTING': 'S&P500',
        'PASSIVE': {
            'MOMENTUM_WEIGHTS': (0.3, 0.3, 0.4), # 20일, 60일, 120일 가중치
        },
    }
}

# =========================================================
# 2. 백테스트 스코어링 로직 이식 (signals._compute_scores)
# =========================================================
def compute_scores(price_data, strategy_name):
    """
    config.py에 정의된 파라미터를 기반으로 시장별 맞춤형 모멘텀 스코어를 계산합니다.
    - US: 지정된 기간별 가중치 합산 방식
    - KR: 변동성으로 나누어 노이즈를 제거한 위험 조정 모멘텀 방식
    """
    strategy_cfg = PARAMS[strategy_name]
    params = strategy_cfg.get('PASSIVE', {})

    if 'MOMENTUM_WEIGHTS' in params: 
        # 가중 모멘텀 방식 (미국 시장)
        w1, w2, w3 = params['MOMENTUM_WEIGHTS']
        scores = (price_data.pct_change(20).fillna(0) * w1) + \
                 (price_data.pct_change(60).fillna(0) * w2) + \
                 (price_data.pct_change(120).fillna(0) * w3)
    else: 
        # 변동성 조절 방식 (한국 시장)
        daily_rets = price_data.pct_change()
        ret_3m = price_data.pct_change(params['MOMENTUM_SHORT'])
        ret_6m = price_data.pct_change(params['MOMENTUM_LONG'])
        vol_3m = daily_rets.rolling(params['VOLATILITY_WINDOW']).std()
        
        epsilon = 1e-6
        score_3m = ret_3m / (vol_3m + epsilon)
        score_6m = ret_6m / (vol_3m + epsilon)
        scores = (score_3m.fillna(0) * 0.5) + (score_6m.fillna(0) * 0.5)
        
    return scores

def get_last_month_first_day(today):
    first_day_of_current_month = today.replace(day=1)
    last_day_of_last_month = first_day_of_current_month - timedelta(days=1)
    return last_day_of_last_month.replace(day=1)

def fetch_price(args):
    """API 차단 방지용 지연 및 지수 백오프가 적용된 가격 수집기"""
    name, code, start_date, retries = args
    time.sleep(random.uniform(0.01, 0.5)) 
    
    for attempt in range(retries):
        try:
            df = fdr.DataReader(code, start=start_date)
            if not df.empty and len(df) > 120: # 120일(가장 긴 모멘텀 기간) 이상의 데이터 필요
                return df['Close'].rename(name)
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                pass 
    return None

# =========================================================
# 3. 메인 분석 엔진
# =========================================================
def analyze_market(strategy_name):
    cfg = PARAMS[strategy_name]
    print("\n" + "="*80)
    print(f"🚀 [{cfg['NAME']}] 시장 분석 시작")
    print("="*80)

    os.makedirs('data', exist_ok=True)
    screener_cache_path = f"data/screener_cache_{strategy_name}.pkl"
    listing_cache_path = f"data/listing_cache_{strategy_name}.pkl"
    
    price_data = None
    universe, sector_map, marcap_map = {}, {}, {}

    # 1. 상장 종목 및 섹터 정보 로딩
    if os.path.exists(listing_cache_path):
        with open(listing_cache_path, 'rb') as f:
            listing_cache = pickle.load(f)
            if listing_cache.get('date') == datetime.now().date():
                print("📦 당일 캐시된 상장 종목 정보 로드 중...")
                universe = listing_cache['universe']
                sector_map = listing_cache['sector_map']
                marcap_map = listing_cache.get('marcap_map', {})
    
    if not universe:
        print("⏳ 상장 종목 및 데이터 스크래핑 중...")
        if strategy_name == 'STOCK_KR':
            listing = fdr.StockListing('KRX')
            desc = fdr.StockListing('KRX-DESC')
            listing = pd.merge(listing, desc[['Code', 'Sector']], on='Code', how='left')
            listing = listing.dropna(subset=['Sector'])
            listing = listing[
                (~listing['Name'].str.contains('관리|환기|스팩|우$|우B$|우C$')) & 
                (listing['Marcap'] >= 100000000000) # 1000억 이상 우량주
            ]
            universe = {row['Name']: row['Code'] for _, row in listing.iterrows()}
            sector_map = dict(zip(listing['Name'], listing['Sector']))
            marcap_map = dict(zip(listing['Name'], listing['Marcap']))
            
        elif strategy_name == 'STOCK_US':
            listing = fdr.StockListing('S&P500')
            listing = listing.dropna(subset=['Sector'])
            universe = {row['Symbol']: row['Symbol'] for _, row in listing.iterrows()}
            sector_map = dict(zip(listing['Symbol'], listing['Sector']))
            marcap_map = {symbol: 1 for symbol in listing['Symbol']}
        
        with open(listing_cache_path, 'wb') as f:
            pickle.dump({'date': datetime.now().date(), 'universe': universe, 'sector_map': sector_map, 'marcap_map': marcap_map}, f)
        print("✅ 종목/섹터 정보 로딩 완료!")

    # 2. 가격 데이터 병렬 수집 (Max Workers: 4 제한으로 API 차단 회피)
    if os.path.exists(screener_cache_path):
        with open(screener_cache_path, 'rb') as f:
            screener_cache = pickle.load(f)
            if screener_cache.get('date') == datetime.now().date():
                print("📦 당일 캐시된 가격 데이터 로드 중...")
                price_data = screener_cache['price_data']
    
    if price_data is None:
        print("⏳ 종가 데이터 수집 중 (API 차단 방지 딜레이 적용)...")
        # 계산에 필요한 최대 기간(120일)에 여유를 더해 약 200일 전부터 수집
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        
        all_price_data = []
        fetch_args = [(name, code, start_date, 3) for name, code in universe.items()]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_price, arg): arg for arg in fetch_args}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(fetch_args), desc="데이터 다운로드", unit="종목"):
                result = future.result()
                if result is not None:
                    all_price_data.append(result)
        
        if not all_price_data:
            print("❌ 데이터 수집 실패. 네트워크 상태를 확인하세요.")
            return

        price_data = pd.concat(all_price_data, axis=1).ffill()
        with open(screener_cache_path, 'wb') as f:
            pickle.dump({'date': datetime.now().date(), 'price_data': price_data}, f)

    # 3. 전략별 스코어 계산 적용
    print("⏳ 맞춤형 모멘텀 스코어 연산 중...")
    scores = compute_scores(price_data, strategy_name)
    
    sector_df = pd.DataFrame.from_dict(sector_map, orient='index', columns=['Sector'])
    marcap_df = pd.DataFrame.from_dict(marcap_map, orient='index', columns=['Marcap'])
    
    today = price_data.index[-1]
    last_month_start = get_last_month_first_day(today)
    
    today_scores_date = scores.index[scores.index <= today][-1]
    last_month_scores_date = scores.index[scores.index <= last_month_start][-1]

# 시가총액 가중 평균 적용 함수
    def calc_weighted_score(df):
        if df['Marcap'].sum() == 0: 
            return df['score'].mean()
        return np.average(df['score'], weights=df['Marcap'])

    # --- 여기서부터 아래 부분으로 교체 ---

    # 당월 랭킹 계산
    merged_today = pd.concat([scores.loc[today_scores_date].rename('score'), sector_df, marcap_df], axis=1).dropna()
    sector_scores_today = merged_today.groupby('Sector').apply(calc_weighted_score)
    
    # [핵심 수정 사항] DataFrame으로 반환될 경우 1차원 Series로 강제 압축
    if isinstance(sector_scores_today, pd.DataFrame):
        sector_scores_today = sector_scores_today.squeeze()
        
    current_ranks = sector_scores_today.rank(ascending=False, method='first')
    
    # 전월 랭킹 계산
    merged_last_month = pd.concat([scores.loc[last_month_scores_date].rename('score'), sector_df, marcap_df], axis=1).dropna()
    sector_scores_last_month = merged_last_month.groupby('Sector').apply(calc_weighted_score)
    
    # [핵심 수정 사항] DataFrame으로 반환될 경우 1차원 Series로 강제 압축
    if isinstance(sector_scores_last_month, pd.DataFrame):
        sector_scores_last_month = sector_scores_last_month.squeeze()
        
    last_ranks = sector_scores_last_month.rank(ascending=False, method='first')
    
    print("\n--- [당월 전체 섹터 순위 Top 10] ---")
    top10 = sector_scores_today.nlargest(10).reset_index()
    top10.columns = ['Sector', 'Weighted Score']
    print(top10.to_string(index=False))

    # 4. 가속 섹터 로직 판별
    accelerating_sectors = []
    candidate_sectors = current_ranks[(current_ranks >= 3) & (current_ranks <= 5)].index
    
    for sector in candidate_sectors:
        if sector in last_ranks.index:
            rank_change = last_ranks[sector] - current_ranks[sector]
            if rank_change >= 2:
                accelerating_sectors.append({
                    'sector': sector, 
                    'rank': int(current_ranks[sector]), 
                    'prev_rank': int(last_ranks[sector])
                })

    print("\n--- [가속 모멘텀 분석 결과] ---")
    if accelerating_sectors:
        best_accelerating_sector = sorted(accelerating_sectors, key=lambda x: x['rank'])[0]
        sector_name = best_accelerating_sector['sector']
        prev_rank = best_accelerating_sector['prev_rank']
        current_rank = best_accelerating_sector['rank']
        
        stocks_in_sector = merged_today[merged_today['Sector'] == sector_name]
        top_stock = stocks_in_sector['score'].nlargest(1).index[0]
        
        print(f"> 📈 상승 가속 섹터: [{sector_name}] (전월 {prev_rank}위 -> 당월 {current_rank}위)")
        print(f"> 🥇 해당 섹터 대장주: [{top_stock}]")
    else:
        print("> 💤 현재 가속 모멘텀 조건에 부합하는 섹터가 없습니다.")

if __name__ == "__main__":
    markets_to_analyze = ['STOCK_KR', 'STOCK_US']
    for market in markets_to_analyze:
        analyze_market(market)