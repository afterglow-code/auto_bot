# backtest_v2/data_loader.py

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import config

def fetch_price_data(tickers, start_date, end_date):
    """(기존 전략용) 종가 데이터만 수집"""
    def _fetch_one(name, code):
        # [API 차단 방지] 랜덤 딜레이
        time.sleep(random.uniform(0.1, 1.0))
        try:
            df = fdr.DataReader(code, start=start_date, end=end_date)
            if df.empty: return None, f"{name}({code}) 데이터 없음"
            
            # [중요] Date가 컬럼으로 들어온 경우 인덱스로 설정
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # 인덱스를 날짜형으로 강제 변환 (안전장치)
            df.index = pd.to_datetime(df.index)
            
            return df['Close'].rename(name), None
        except Exception as e:
            return None, f"{name}({code}) 수집 실패: {e}"

    df_list = []
    total_count = len(tickers)
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(_fetch_one, name, code): name for name, code in tickers.items()}
        
        for i, future in enumerate(as_completed(future_to_ticker)):
            name = future_to_ticker[future]
            # 진행률 표시
            print(f"\r   [Price] 수집 진행: {i+1}/{total_count}", end='', flush=True)

            series, error_msg = future.result()
            if series is not None and not series.empty:
                df_list.append(series)

    print("\n✅ 병렬 데이터 수집 완료!")
    if not df_list: return pd.DataFrame()
    return pd.concat(df_list, axis=1)

def fetch_ohlcv_data(tickers, start_date, end_date):
    """(하이브리드 전략용) OHLCV 데이터 수집"""
    def _fetch_one(name, code):
        # [API 차단 방지] 랜덤 딜레이
        time.sleep(random.uniform(0.1, 1.0))
        try:
            df = fdr.DataReader(code, start=start_date, end=end_date)
            if df.empty: return None, None
            
            # [중요] Date가 컬럼이면 인덱스로 변환
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # 인덱스 날짜형 변환 보장
            df.index = pd.to_datetime(df.index)
            
            # 필요한 컬럼만 추출하여 리턴 (종목명으로 관리하기 위해 튜플 리턴)
            return name, df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            return None, None

    # 데이터 담을 그릇 초기화
    data_frames = {col: [] for col in ['Open', 'High', 'Low', 'Close', 'Volume']}
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(_fetch_one, name, code): name for name, code in tickers.items()}
        
        total = len(tickers)
        for i, future in enumerate(as_completed(future_to_ticker)):
            print(f"\r   [OHLCV] 수집 진행: {i+1}/{total}", end='', flush=True)
            res = future.result()
            if res[0] is not None: # name이 None이 아니면 성공
                name, df = res
                for col in data_frames.keys():
                    series = df[col].rename(name)
                    data_frames[col].append(series)

    print("\n✅ 병렬 데이터 수집 완료!")
    
    # 리스트 병합 (인덱스 자동 정렬됨)
    final_data = {}
    for col, series_list in data_frames.items():
        if series_list:
            final_data[col] = pd.concat(series_list, axis=1).ffill()
        else:
            final_data[col] = pd.DataFrame()
            
    return final_data

def load_data_for_strategy(strategy_name):
    """(기존) 전략별 데이터 로드"""
    print("\n" + "="*50)
    print(f"📊 데이터 로딩: [{strategy_name}]")
    print("="*50)

    cfg = config.PARAMS[strategy_name]
    fetch_start_dt = datetime.strptime(config.START_DATE, "%Y-%m-%d") - timedelta(days=365)
    fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d")

    # 유니버스 구성
    universe = {}
    if strategy_name == 'ETF_KR':
        print("   - 한국 ETF 전종목 리스트 조회...")
        etf_listing = fdr.StockListing('ETF/KR')
        
        # 필터링
        if 'MIN_MARCAP' in cfg['UNIVERSE'] and cfg['UNIVERSE']['MIN_MARCAP'] > 0:
            etf_listing = etf_listing[etf_listing['MarCap'] >= cfg['UNIVERSE']['MIN_MARCAP']]
            print(f"     ✓ 시총 {cfg['UNIVERSE']['MIN_MARCAP']}억 이상 필터")
        
        # 패턴 제외
        if 'EXCLUDE_PATTERNS' in cfg['UNIVERSE']:
            for pattern in cfg['UNIVERSE']['EXCLUDE_PATTERNS']:
                before = len(etf_listing)
                etf_listing = etf_listing[~etf_listing['Name'].str.contains(pattern, case=False, na=False)]
                if before > len(etf_listing):
                    print(f"     ✓ '{pattern}' 제외: {before - len(etf_listing)}개")
        
        # 상위 N개 선택
        if 'TOP_N_ETFS' in cfg['UNIVERSE'] and cfg['UNIVERSE']['TOP_N_ETFS'] > 0:
            etf_listing = etf_listing.nlargest(cfg['UNIVERSE']['TOP_N_ETFS'], 'MarCap')
            print(f"     ✓ 시총 상위 {cfg['UNIVERSE']['TOP_N_ETFS']}개 선택")
        
        for _, row in etf_listing.iterrows():
            universe[row['Name']] = row['Symbol']
        
        # 방어자산 추가
        universe[cfg['DEFENSE_ASSET']] = '261240'
        
    elif strategy_name == 'STOCK_KR':
        print("   - KOSPI/KOSDAQ 시총 상위 수집...")
        kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg['UNIVERSE']['KOSPI_TOP_N'])
        kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg['UNIVERSE']['KOSDAQ_TOP_N'])
        for _, row in pd.concat([kospi, kosdaq]).iterrows():
            universe[row['Name']] = row['Code']
        universe[cfg['DEFENSE_ASSET']] = '261240' # 달러선물
        
    elif strategy_name == 'STOCK_US':
        print("   - S&P500/NASDAQ 수집...")
        # (샘플링) 속도를 위해 50개만 테스트하려면 아래 주석 해제
        # sp500 = fdr.StockListing('S&P500').head(50) 
        sp500 = fdr.StockListing('S&P500')
        for _, row in sp500.iterrows(): universe[row['Symbol']] = row['Symbol']
        universe[cfg['DEFENSE_ASSET']] = 'BIL'

    # 벤치마크
    print(f"   - 벤치마크({cfg['MARKET_INDEX']}) 수집...")
    benchmark = fdr.DataReader(cfg['MARKET_INDEX'], fetch_start_str, config.END_DATE)['Close']
    if 'Date' in pd.DataFrame(benchmark).columns: # 벤치마크도 안전장치
         benchmark.index = pd.to_datetime(benchmark.index)

    # 가격 데이터
    print(f"   - 종목 데이터 수집 ({len(universe)}개)...")
    price_data = fetch_price_data(universe, fetch_start_str, config.END_DATE)
    
    return price_data, benchmark

def load_data_for_hybrid(strategy_name):
    """(신규) 하이브리드 전략용 데이터 로드"""
    print("\n" + "="*50)
    print(f"📊 하이브리드 데이터 로딩: [{strategy_name}]")
    print("="*50)

    # 1. 유니버스 구성 (기존 함수 로직 재사용 가능하지만 명시적으로 작성)
    cfg = config.PARAMS[strategy_name]
    fetch_start_dt = datetime.strptime(config.START_DATE, "%Y-%m-%d") - timedelta(days=365)
    fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d")

    universe = {}
    if strategy_name == 'ETF_KR':
        etf_listing = fdr.StockListing('ETF/KR')
        if 'MIN_MARCAP' in cfg['UNIVERSE'] and cfg['UNIVERSE']['MIN_MARCAP'] > 0:
            etf_listing = etf_listing[etf_listing['MarCap'] >= cfg['UNIVERSE']['MIN_MARCAP']]
        if 'EXCLUDE_PATTERNS' in cfg['UNIVERSE']:
            for pattern in cfg['UNIVERSE']['EXCLUDE_PATTERNS']:
                etf_listing = etf_listing[~etf_listing['Name'].str.contains(pattern, case=False, na=False)]
        if 'TOP_N_ETFS' in cfg['UNIVERSE'] and cfg['UNIVERSE']['TOP_N_ETFS'] > 0:
            etf_listing = etf_listing.nlargest(cfg['UNIVERSE']['TOP_N_ETFS'], 'MarCap')
        for _, row in etf_listing.iterrows():
            universe[row['Name']] = row['Symbol']
    elif strategy_name == 'STOCK_KR':
        kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg['UNIVERSE']['KOSPI_TOP_N'])
        kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg['UNIVERSE']['KOSDAQ_TOP_N'])
        for _, row in pd.concat([kospi, kosdaq]).iterrows():
            universe[row['Name']] = row['Code']
    elif strategy_name == 'STOCK_US':
        sp500 = fdr.StockListing('S&P500') # 전체 대상
        for _, row in sp500.iterrows(): universe[row['Symbol']] = row['Symbol']

    # 2. OHLCV 데이터 로드
    ohlcv_data = fetch_ohlcv_data(universe, fetch_start_str, config.END_DATE)
    
    # 3. 벤치마크
    benchmark = fdr.DataReader(cfg['MARKET_INDEX'], fetch_start_str, config.END_DATE)['Close']
    
    return ohlcv_data, benchmark

if __name__ == '__main__':
    # 모듈 단독 테스트
    # config.py에 STRATEGY_TO_RUN 설정이 되어 있어야 함
    try:
        strategy = config.STRATEGY_TO_RUN
        prices, benchmark = load_data_for_strategy(strategy)
        
        print("\n--- 가격 데이터 샘플 ---")
        print(prices.tail())
        
        print("\n--- 벤치마크 데이터 샘플 ---")
        print(benchmark.tail())
    except Exception as e:
        print(f"테스트 실패: {e}")