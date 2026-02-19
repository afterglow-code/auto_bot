# backtest_v2/compare_volume_filters.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from data_loader import load_data_for_hybrid
from hybrid_engine import HybridEngine
from reporting import analyze_performance
import platform

# 폰트 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 단순 거래량 엔진 (Case A) ---
# 기존 HybridEngine이 이미 이 로직을 쓰고 있으므로 그대로 사용
class HybridEngineSimple(HybridEngine):
    def run(self):
        print("\n🚀 [Case A] Simple Hybrid 실행 (전일 대비 2배)")
        return super().run()

# --- 2. 동적 거래량 엔진 (Case B) ---
class HybridEngineDynamic(HybridEngine):
    def calculate_indicators(self):
        super().calculate_indicators() # 기존 지표(ATR, Momentum) 계산
        
        # [추가] 20일 이동평균 거래량 계산 (어제 기준)
        # shift(1)을 하여 '어제까지의 20일 평균'을 만듦
        self.vol_ma20 = self.volume.rolling(window=20).mean().shift(1)

    def run(self):
        self.calculate_indicators()
        print(f"\n🚀 [Case B] Dynamic Hybrid 실행 (20일 평균 대비 2배)")
        
        sim_dates = self.close.index[self.close.index >= config.START_DATE]

        for date in sim_dates:
            # --- 0. 당일 주도주 순위 ---
            daily_mom = self.momentum.loc[date]
            current_top_n = daily_mom[daily_mom > 0].sort_values(ascending=False).head(self.cfg['TOP_N']).index.tolist()

            # --- 1. 매도 로직 (기존과 동일) ---
            # (super().run()의 로직을 그대로 가져와야 하지만, 
            #  오버라이딩의 한계로 여기서는 핵심 매수 로직만 변경하기 위해 전체 루프를 다시 씁니다)
            #  * 편의상 매도 로직은 HybridEngine과 완전히 동일하다고 가정하고 복사합니다 *
            
            # [매도 로직 생략 없이 구현]
            for ticker in list(self.holdings.keys()):
                info = self.holdings[ticker]
                if ticker not in self.low.columns or pd.isna(self.low.loc[date, ticker]): continue
                
                curr_low = self.low.loc[date, ticker]
                curr_high = self.high.loc[date, ticker]
                
                # 손절 / 본전청산
                if curr_low <= info['stop_price']:
                    sell_price = info['stop_price'] * (1 - self.slippage)
                    revenue = info['qty'] * sell_price
                    self.capital += (revenue - (revenue * self.commission))
                    del self.holdings[ticker]
                    continue
                
                # 본전 트리거
                if not info['is_breakeven']:
                    if curr_high >= info['buy_price'] * (1 + self.hp['BREAKEVEN_TRIGGER']):
                        self.holdings[ticker]['stop_price'] = info['buy_price'] * 1.005
                        self.holdings[ticker]['is_breakeven'] = True

                # 목표가 달성 및 순위 이탈 체크
                if curr_high >= info['target_price']:
                    self.holdings[ticker]['target_reached'] = True
                
                if info.get('target_reached', False):
                    if ticker not in current_top_n:
                        sell_price = self.close.loc[date, ticker] * (1 - self.slippage)
                        revenue = info['qty'] * sell_price
                        self.capital += (revenue - (revenue * self.commission))
                        del self.holdings[ticker]

            # --- 2. 매수 로직 (여기가 핵심 변경!) ---
            if len(self.holdings) < self.hp['MAX_SLOTS']:
                candidates = []
                prev_mom = self.momentum.shift(1).loc[date]
                today_sig = self.signal.loc[date]
                today_vol = self.volume.loc[date]
                
                # [변경] 전일 거래량이 아니라 '20일 평균 거래량' 가져오기
                vol_baseline = self.vol_ma20.loc[date]
                
                for ticker in current_top_n:
                    if ticker in self.holdings: continue
                    
                    # 데이터 유효성 체크
                    if pd.isna(vol_baseline[ticker]) or pd.isna(prev_mom[ticker]): continue

                    # 진입 조건
                    is_breakout = (daily_mom[ticker] >= 100) and (prev_mom[ticker] < 100)
                    is_strong = (daily_mom[ticker] >= 100) and (daily_mom[ticker] > today_sig[ticker])
                    
                    # [CASE B 조건] 오늘 거래량 >= 20일 평균 * 2.0
                    # (평균 거래량이 0인 경우 방지)
                    if vol_baseline[ticker] > 0:
                        is_volume_spike = (today_vol[ticker] >= vol_baseline[ticker] * 2.0)
                    else:
                        is_volume_spike = False
                    
                    if (is_breakout or is_strong) and is_volume_spike:
                        candidates.append({
                            'ticker': ticker, 
                            'momentum': daily_mom[ticker], 
                            'close': self.close.loc[date, ticker], 
                            'atr': self.atr.loc[date, ticker]
                        })

                candidates.sort(key=lambda x: x['momentum'], reverse=True)
                for cand in candidates:
                    if len(self.holdings) >= self.hp['MAX_SLOTS']: break
                    budget = self.capital / (self.hp['MAX_SLOTS'] - len(self.holdings))
                    buy_price = cand['close'] * (1 + self.slippage)
                    qty = int(budget // buy_price)
                    if qty > 0:
                        self.capital -= (qty * buy_price) * (1 + self.commission)
                        self.holdings[cand['ticker']] = {
                            'qty': qty, 'buy_price': buy_price,
                            'target_price': buy_price + (cand['atr'] * self.hp['TARGET_ATR_MULT']),
                            'stop_price': buy_price * (1 - self.hp['STOP_LOSS_PCT']),
                            'is_breakeven': False, 'target_reached': False
                        }

            # 3. 평가
            curr_val = self.capital
            for t, info in self.holdings.items():
                curr_val += info['qty'] * self.close.loc[date, t]
            self.history.append({'Date': date, 'TotalValue': curr_val})

        return pd.DataFrame(self.history).set_index('Date'), pd.DataFrame(self.trade_log)

def run_experiment():
    strategy_name = 'STOCK_KR' # 혹은 config.STRATEGY_TO_RUN
    print(f"🔬 거래량 필터 비교 실험 시작: {strategy_name}")
    
    # 데이터 로드
    ohlcv_data, benchmark = load_data_for_hybrid(strategy_name)
    
    # 1. Simple 실행
    engine_simple = HybridEngineSimple(ohlcv_data, strategy_name)
    hist_simple, _ = engine_simple.run()
    met_simple = analyze_performance(hist_simple, benchmark)
    
    # 2. Dynamic 실행
    engine_dynamic = HybridEngineDynamic(ohlcv_data, strategy_name)
    hist_dynamic, _ = engine_dynamic.run()
    met_dynamic = analyze_performance(hist_dynamic, benchmark)
    
    # 결과 출력
    print("\n" + "="*80)
    print(f"{'지표':<15} | {'Case A (Simple)':>15} | {'Case B (Dynamic)':>15} | {'차이 (B-A)':>15}")
    print("-" * 80)
    
    metrics = ['total_return_pct', 'cagr_pct', 'mdd_pct', 'sharpe_ratio']
    labels = ['Total Return', 'CAGR', 'MDD', 'Sharpe Ratio']
    
    for key, label in zip(metrics, labels):
        val_a = met_simple[key]
        val_b = met_dynamic[key]
        diff = val_b - val_a
        unit = "%" if "pct" in key else ""
        print(f"{label:<15} | {val_a:>14.2f}{unit} | {val_b:>14.2f}{unit} | {diff:>14.2f}{unit}")
        
    print("="*80)
    
    # 그래프
    plt.figure(figsize=(14, 7))
    plt.plot(hist_simple.index, hist_simple['TotalValue'], label='Case A: Simple (Prev * 2)', alpha=0.7)
    plt.plot(hist_dynamic.index, hist_dynamic['TotalValue'], label='Case B: Dynamic (MA20 * 2)', linestyle='--', linewidth=2)
    plt.title("Hybrid 전략 거래량 필터 비교")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()