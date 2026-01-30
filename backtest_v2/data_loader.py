# backtest_v2/data_loader.py

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

def fetch_price_data(tickers, start_date, end_date):
    """
    여러 종목의 시세 데이터를 병렬로 수집합니다.
    :param tickers: {'종목명': '종목코드'} 형태의 딕셔너리
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :return: pd.DataFrame, 각 종목의 종가가 컬럼으로 구성됨
    """
    
    def _fetch_one(name, code):
        try:
            # 병렬 처리 시 너무 빠른 요청은 차단될 수 있으므로 필요시 sleep 추가 가능
            # import time; time.sleep(0.1) 
            df = fdr.DataReader(code, start=start_date, end=end_date)
            if df.empty:
                return None, f"{name}({code}) 데이터 없음"
            return df['Close'].rename(name), None
        except Exception as e:
            return None, f"{name}({code}) 수집 실패: {e}"

    df_list = []
    total_count = len(tickers)
    
    # config.MAX_WORKERS를 사용하여 스레드 개수 조절 (보통 4 권장)
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(_fetch_one, name, code): name for name, code in tickers.items()}
        
        for i, future in enumerate(as_completed(future_to_ticker)):
            name = future_to_ticker[future]
            # 진행 상황을 한 줄로 표시
            print(f"\r   데이터 수집 진행: {i+1}/{total_count}", end='', flush=True)

            series, error_msg = future.result()
            if error_msg:
                # 에러 로그는 너무 많으면 지저분하므로 주석 처리 (필요시 해제)
                # print(f"\n⚠️  {error_msg}") 
                continue
            
            if series is not None and not series.empty:
                df_list.append(series)

    print("\n✅ 병렬 데이터 수집 완료!")
    
    if not df_list:
        return pd.DataFrame()
        
    raw_data = pd.concat(df_list, axis=1)
    return raw_data

def load_data_for_strategy(strategy_name):
    """선택된 전략에 필요한 모든 데이터를 로드합니다."""
    print("\n" + "="*50)
    print(f"📊 데이터 로딩 시작: [{strategy_name}]")
    print("="*50)

    cfg = config.PARAMS[strategy_name]
    
    # 백테스트 기간에 지표 계산을 위한 1년(365일)을 더해서 다운로드
    fetch_start_dt = datetime.strptime(config.START_DATE, "%Y-%m-%d") - timedelta(days=365)
    fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d")

    # 1. 유니버스 구성
    universe = {}
    if strategy_name == 'ETF_KR':
        universe = cfg['UNIVERSE']
        
    elif strategy_name == 'STOCK_KR':
        print("   - KOSPI/KOSDAQ 시가총액 상위 종목 목록 수집 중...")
        kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(cfg['UNIVERSE']['KOSPI_TOP_N'])
        kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(cfg['UNIVERSE']['KOSDAQ_TOP_N'])
        for _, row in pd.concat([kospi, kosdaq]).iterrows():
            universe[row['Name']] = row['Code']
        # 방어 자산 추가
        universe[cfg['DEFENSE_ASSET']] = '261240' # KODEX 미국달러선물
        
    elif strategy_name == 'STOCK_US':
        print("   - S&P500 전종목 + NASDAQ 상위 100 종목 수집 중...")
        
        # [수정된 부분] -------------------------------------------------------
        # 1) S&P 500 전종목 가져오기 (알파벳 순서 문제 해결을 위해 전수 조사)
        sp500_df = fdr.StockListing('S&P500')
        sp500_tickers = set(sp500_df['Symbol'].tolist())
        
        # 2) NASDAQ 상위 100개 (QQQ 스타일, 성장주 보강)
        nasdaq_df = fdr.StockListing('NASDAQ')
        # 시총순 정렬되어 있다고 가정하고 상위 100개 추출
        nasdaq100_tickers = set(nasdaq_df.head(100)['Symbol'].tolist())
        
        # 3) 합집합으로 병합 (중복 제거)
        combined_tickers = sp500_tickers.union(nasdaq100_tickers)
        
        # 4) 유니버스 딕셔너리에 추가
        for ticker in combined_tickers:
            universe[ticker] = ticker
        # -------------------------------------------------------------------
            
        # 방어 자산 추가
        universe[cfg['DEFENSE_ASSET']] = 'BIL'

    # 2. 벤치마크(시장 지수) 데이터 로드
    print(f"   - 벤치마크({cfg['MARKET_INDEX']}) 데이터 수집 중...")
    benchmark_data = fdr.DataReader(cfg['MARKET_INDEX'], fetch_start_str, config.END_DATE)['Close'].rename('benchmark')
    
    # 3. 유니버스 가격 데이터 로드
    print(f"   - 유니버스({len(universe)}개 종목) 가격 데이터 수집 중...")
    price_data = fetch_price_data(universe, fetch_start_str, config.END_DATE)
    
    # 데이터 정제 (결측치가 너무 많은 종목 제거)
    # thresh=0.9 -> 데이터가 90% 이상 존재하는 종목만 남김 (상장폐지나 최근 상장주 필터링 효과)
    price_data = price_data.ffill().dropna(axis=1, thresh=int(len(price_data) * 0.9))
    
    print(f"✅ 데이터 로딩 완료! (최종 분석 대상 {len(price_data.columns)}개 종목)")
    
    return price_data, benchmark_data

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