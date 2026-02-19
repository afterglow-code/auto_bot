# backtest_v2/signals.py

import pandas as pd
import numpy as np
import config

def get_rebalance_dates(dates, start_date):
    """백테스트 기간 중 리밸런싱 날짜(매월 첫 거래일) 목록 반환"""
    df = pd.DataFrame(index=dates)
    df = df[df.index >= start_date]
    df['year_month'] = df.index.strftime('%Y-%m')
    rebalance_dates = df.reset_index().rename(columns={'index': 'Date'}).groupby('year_month')['Date'].first().tolist()
    return rebalance_dates

def generate_signals(price_or_ohlcv_data, benchmark_data, strategy_name, use_vol_filter=False):
    """
    전략에 맞는 투자 신호를 생성합니다. 
    use_vol_filter=True일 경우 리밸런싱 날짜 당일 거래량이 전일 대비 2배인 종목만 필터링합니다.
    
    Args:
        price_or_ohlcv_data: DataFrame (Close만) 또는 dict (OHLCV 딕셔너리)
    """
    # OHLCV 딕셔너리인지 DataFrame인지 체크
    if isinstance(price_or_ohlcv_data, dict):
        # OHLCV 데이터인 경우
        price_data = price_or_ohlcv_data['Close']
        volume_data = price_or_ohlcv_data['Volume']
    else:
        # Close만 있는 DataFrame인 경우
        price_data = price_or_ohlcv_data
        volume_data = None
    
    print(f"\n📈 투자 신호 생성: [{config.PARAMS[strategy_name]['NAME']}] (Vol Filter: {use_vol_filter})")

    rebalance_dates = get_rebalance_dates(price_data.index, config.START_DATE)
    signals = pd.DataFrame(index=rebalance_dates, columns=price_data.columns).fillna(0.0)
    cfg = config.PARAMS[strategy_name]
    ma_series = benchmark_data.rolling(window=cfg['MARKET_TIMING_MA']).mean()

    for date in rebalance_dates:
        # 1. 시장 타이밍 확인
        market_index_price = benchmark_data.loc[date]
        current_ma = ma_series.loc[date]
        if pd.isna(current_ma) or market_index_price < current_ma:
            signals.loc[date, cfg['DEFENSE_ASSET']] = 1.0
            continue

        # 2. 종목별 스코어 계산
        if strategy_name == 'STOCK_KR':
            daily_rets = price_data.pct_change()
            ret_3m = price_data.pct_change(60).loc[date]
            vol_3m = daily_rets.rolling(60).std().loc[date]
            scores = ret_3m / (vol_3m + 1e-6)
        else: # US or ETF
            w1, w2, w3 = cfg['MOMENTUM_WEIGHTS']
            scores = (price_data.pct_change(20).loc[date].fillna(0) * w1) + \
                     (price_data.pct_change(60).loc[date].fillna(0) * w2) + \
                     (price_data.pct_change(120).loc[date].fillna(0) * w3)
        
        scores = scores.drop(cfg['DEFENSE_ASSET'], errors='ignore')

        # 3. [추가] 거래량 필터 적용 (2배 돌파 여부)
        if use_vol_filter and volume_data is not None:
            try:
                # 당일 거래량 / 전일 거래량
                vol_ratio = volume_data.loc[date] / volume_data.shift(1).loc[date]
                vol_mask = vol_ratio >= 2.0
                scores = scores[vol_mask] 
            except Exception:
                pass
        elif use_vol_filter and volume_data is None:
            print("⚠️ 거래량 필터를 사용하려면 OHLCV 데이터가 필요합니다. 필터 무시됨.")

        # 4. 상위 종목 비중 할당
        positive_scores = scores[scores > 0].sort_values(ascending=False)
        if positive_scores.empty:
            signals.loc[date, cfg['DEFENSE_ASSET']] = 1.0
        else:
            top_n_assets = positive_scores.head(cfg['TOP_N'])
            weight = 1.0 / len(top_n_assets)
            for asset_name in top_n_assets.index:
                signals.loc[date, asset_name] = weight
    
    return signals
if __name__ == '__main__':
    # 모듈 단독 테스트
    from data_loader import load_data_for_strategy
    
    strategy = config.STRATEGY_TO_RUN
    price_data, benchmark_data = load_data_for_strategy(strategy)
    
    investment_signals = generate_signals(price_data, benchmark_data, strategy)
    
    print("\n--- 최종 신호 데이터 샘플 (0이 아닌 값만 표시) ---")
    print(investment_signals.apply(lambda x: x[x > 0], axis=1))
