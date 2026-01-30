# backtest_v2/signals.py

import pandas as pd
import numpy as np
import config

def get_rebalance_dates(dates, start_date):
    """백테스트 기간 중 리밸런싱이 필요한 날짜(매월 첫 거래일) 목록을 반환합니다."""
    df = pd.DataFrame(index=dates)
    df = df[df.index >= start_date]
    df['year_month'] = df.index.strftime('%Y-%m')
    
    # 각 월의 첫 번째 날짜를 리밸런싱 날짜로 선택
    rebalance_dates = df.reset_index().rename(columns={'index': 'Date'}).groupby('year_month')['Date'].first().tolist()
    return rebalance_dates

def generate_signals(price_data, benchmark_data, strategy_name):
    """전략에 맞는 투자 신호(종목별 비중)를 생성합니다."""
    print("\n" + "="*50)
    print(f"📈 투자 신호 생성 시작: [{config.PARAMS[strategy_name]['NAME']}]")
    print("="*50)

    # 리밸런싱 날짜 목록 생성
    rebalance_dates = get_rebalance_dates(price_data.index, config.START_DATE)
    
    # 신호를 저장할 데이터프레임 (인덱스: 날짜, 컬럼: 종목, 값: 비중)
    signals = pd.DataFrame(index=rebalance_dates, columns=price_data.columns).fillna(0.0)

    # 전략별 파라미터 로드
    cfg = config.PARAMS[strategy_name]
    
    # 전체 기간에 대한 이동평균선 미리 계산
    ma_series = benchmark_data.rolling(window=cfg['MARKET_TIMING_MA']).mean()

    for date in rebalance_dates:
        print(f"   - 신호 생성 중: {date.strftime('%Y-%m-%d')}")
        
        # 1. 시장 타이밍 확인
        market_index_price = benchmark_data.loc[date]
        current_ma = ma_series.loc[date]
        
        # 데이터가 부족하여 MA가 계산되지 않는 경우를 대비
        if pd.isna(current_ma):
            signals.loc[date, cfg['DEFENSE_ASSET']] = 1.0
            continue

        # 이평선 상승/하락 추세 확인 (간단하게 5일 전과 비교)
        try:
            # 5일 전 날짜에 데이터가 없을 수 있으므로, asof로 가장 가까운 과거 데이터를 찾음
            prev_ma_date = date - pd.Timedelta(days=5)
            prev_ma = ma_series.asof(prev_ma_date)
            ma_is_rising = current_ma > prev_ma
        except (KeyError, IndexError):
            ma_is_rising = False # 데이터가 부족할 경우 하락장으로 간주

        # 시장 국면 정의
        is_bull_market = market_index_price > current_ma
        is_neutral_market = not is_bull_market and ma_is_rising

        # 2. 자산 배분 비율 결정
        offensive_ratio = 0.0
        if is_bull_market:
            offensive_ratio = 1.0
        elif is_neutral_market:
            offensive_ratio = 0.5
        
        # 하락장 (offensive_ratio == 0.0)인 경우
        if offensive_ratio == 0.0:
            signals.loc[date, cfg['DEFENSE_ASSET']] = 1.0
            continue

        # 3. 공격 자산 선정 (기존 로직 활용)
        if strategy_name == 'ETF_KR' or strategy_name == 'STOCK_US':
            # 가중 모멘텀 스코어
            w1, w2, w3 = cfg['MOMENTUM_WEIGHTS']
            mom_1m = price_data.pct_change(20).loc[date]
            mom_3m = price_data.pct_change(60).loc[date]
            mom_6m = price_data.pct_change(120).loc[date]
            scores = (mom_1m.fillna(0) * w1) + (mom_3m.fillna(0) * w2) + (mom_6m.fillna(0) * w3)
        
        elif strategy_name == 'STOCK_KR':
            # 변동성 조절 모멘텀 스코어
            daily_rets = price_data.pct_change()
            ret_3m = price_data.pct_change(60).loc[date]
            ret_6m = price_data.pct_change(120).loc[date]
            vol_3m = daily_rets.rolling(60).std().loc[date]
            epsilon = 1e-6
            score_3m = ret_3m / (vol_3m + epsilon)
            score_6m = ret_6m / (vol_3m + epsilon)
            scores = (score_3m.fillna(0) * 0.5) + (score_6m.fillna(0) * 0.5)

        # 방어 자산은 투자 대상에서 제외
        scores = scores.drop(cfg['DEFENSE_ASSET'], errors='ignore')
        
        # 점수가 0 이상인 종목만 선택
        positive_scores = scores[scores > 0].sort_values(ascending=False)
        
        if positive_scores.empty:
            # 상승 모멘텀 종목이 없으면 전량 방어 자산으로
            signals.loc[date, cfg['DEFENSE_ASSET']] = 1.0
        else:
            # 4. 최종 자산 배분
            top_n_assets = positive_scores.head(cfg['TOP_N'])
            num_assets = len(top_n_assets)
            
            # 공격 자산 비중 적용
            weight = offensive_ratio / num_assets
            for asset_name in top_n_assets.index:
                signals.loc[date, asset_name] = weight
            
            # 방어 자산 비중 적용 (중립장에서만)
            if offensive_ratio < 1.0:
                signals.loc[date, cfg['DEFENSE_ASSET']] = 1.0 - offensive_ratio
    
    print("✅ 투자 신호 생성 완료!")
    return signals

if __name__ == '__main__':
    # 모듈 단독 테스트
    from data_loader import load_data_for_strategy
    
    strategy = config.STRATEGY_TO_RUN
    price_data, benchmark_data = load_data_for_strategy(strategy)
    
    investment_signals = generate_signals(price_data, benchmark_data, strategy)
    
    print("\n--- 최종 신호 데이터 샘플 (0이 아닌 값만 표시) ---")
    print(investment_signals.apply(lambda x: x[x > 0], axis=1))
