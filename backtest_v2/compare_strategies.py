# backtest_v2/compare_strategies.py

import pandas as pd
import matplotlib.pyplot as plt
import config
from data_loader import load_data_for_strategy, load_data_for_hybrid
from signals import generate_signals
from hybrid_engine import HybridEngine
from reporting import analyze_performance
import platform
from engine import BacktestEngine, RiskManagedMonthlyEngine
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def run_comparison():
    strategy_name = config.STRATEGY_TO_RUN
    print(f"⚔️ 전략 3중 비교: [Monthly Pure] vs [Monthly Vol] vs [Hybrid 3.0]")
    
    # 1. 데이터 로드 (OHLCV 전체 데이터 사용)
    ohlcv_data, benchmark = load_data_for_hybrid(strategy_name)
    
    # CASE A: 순수 월간 전략 (기존)
    print("\n[CASE A] 순수 월간 리밸런싱 실행...")
    sig_pure = generate_signals(ohlcv_data, benchmark, strategy_name, use_vol_filter=False)
    hist_pure, _ = BacktestEngine(ohlcv_data['Close'], sig_pure).run()
    met_pure = analyze_performance(hist_pure, benchmark)
    
    # CASE B: 월간 + 거래량 2배 필터 적용
    print("\n[CASE B] 월간 + 거래량(2배) 필터 실행...")
    sig_vol = generate_signals(ohlcv_data, benchmark, strategy_name, use_vol_filter=True)
    hist_vol, _ = BacktestEngine(ohlcv_data['Close'], sig_vol).run()
    met_vol = analyze_performance(hist_vol, benchmark)
    
    # CASE C: 하이브리드 3.0 (Active 일간 대응)
    print("\n[CASE C] 하이브리드 3.0 실행...")
    # 주의: 유저가 업로드한 hybrid_engine.py에서 거래량 1.0배를 다시 2.0배로 확인하세요!
    engine_hybrid = HybridEngine(ohlcv_data, strategy_name)
    hist_hybrid, _ = engine_hybrid.run()
    met_hybrid = analyze_performance(hist_hybrid, benchmark)

    # [CASE D 추가] 리스크 관리형 월간
    print("\n[CASE D] 리스크 관리형 월간(Monthly + Stop-loss) 실행...")
    hist_rm, _ = RiskManagedMonthlyEngine(ohlcv_data, sig_pure).run()
    met_rm = analyze_performance(hist_rm, benchmark)
    
    # --- 결과 비교표 출력 ---
    print("\n" + "="*75)
    print(f"{'Performance 지표':<18} | {'Monthly(Pure)':>14} | {'Monthly(Vol)':>14} | {'Hybrid 3.0':>14} | {'Risk Managed':>14}")
    print("-" * 75)
    metrics = [
        ('total_return_pct', 'Total Return'),
        ('cagr_pct', 'CAGR'),
        ('mdd_pct', 'MDD'),
        ('sharpe_ratio', 'Sharpe Ratio')
    ]
    for key, label in metrics:
        unit = "%" if "pct" in key else ""
        print(f"{label:<18} | {met_pure[key]:>13.2f}{unit} | {met_vol[key]:>13.2f}{unit} | {met_hybrid[key]:>13.2f}{unit} | {met_rm[key]:>13.2f}{unit}")
    print("="*75)

    # 누적 수익률 그래프 시각화
    plt.figure(figsize=(15, 8))
    plt.plot(hist_pure.index, hist_pure['TotalValue'], label='Monthly (Pure)', alpha=0.5)
    plt.plot(hist_vol.index, hist_vol['TotalValue'], label='Monthly (Vol Filter)', linestyle='--')
    plt.plot(hist_hybrid.index, hist_hybrid['TotalValue'], label='Hybrid 3.0 (Active)', linewidth=2.5)
    plt.plot(hist_rm.index, hist_rm['TotalValue'], label='Risk Managed', linestyle='-.')
    plt.title(f"전략 비교 분석: {strategy_name}", fontsize=16)
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_comparison()