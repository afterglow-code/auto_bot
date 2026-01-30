# backtest_v2/reporting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os

import config


def analyze_performance(portfolio_history, benchmark_data):
    """백테스트 성과를 분석하고 주요 지표를 계산합니다."""
    
    # 1. 최종 수익률
    initial_capital = config.INITIAL_CAPITAL
    final_value = portfolio_history['TotalValue'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    # 2. CAGR (연평균 복리 수익률)
    days = (portfolio_history.index[-1] - portfolio_history.index[0]).days
    years = days / 365.25
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # 3. MDD (최대 낙폭)
    rolling_max = portfolio_history['TotalValue'].cummax()
    daily_drawdown = portfolio_history['TotalValue'] / rolling_max - 1.0
    mdd = daily_drawdown.min() * 100
    
    # 4. Sharpe Ratio (샤프 지수)
    daily_returns = portfolio_history['TotalValue'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # 5. 벤치마크 성과
    benchmark_period = benchmark_data.loc[portfolio_history.index]
    benchmark_return = (benchmark_period.iloc[-1] / benchmark_period.iloc[0] - 1) * 100
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'cagr_pct': cagr,
        'mdd_pct': mdd,
        'sharpe_ratio': sharpe_ratio,
        'benchmark_return_pct': benchmark_return,
        'num_years': years
    }
    return metrics

def print_summary(metrics):
    """분석 결과를 콘솔에 출력합니다."""
    strategy_name = config.PARAMS[config.STRATEGY_TO_RUN]['NAME']
    
    print("\n" + "="*60)
    print(f"📜 최종 성과 보고서: [{strategy_name}]")
    print("="*60)
    print(f"  - 백테스트 기간: {metrics['num_years']:.2f} 년")
    print(f"  - 초기 자본: {metrics['initial_capital']:,.0f} 원")
    print(f"  - 최종 자산: {metrics['final_value']:,.0f} 원")
    print("-" * 60)
    print(f"  - 총 수익률: {metrics['total_return_pct']:.2f} %")
    print(f"  - 연평균 복리 수익률 (CAGR): {metrics['cagr_pct']:.2f} %")
    print(f"  - 벤치마크 수익률: {metrics['benchmark_return_pct']:.2f} %")
    print("-" * 60)
    print(f"  - 최대 낙폭 (MDD): {metrics['mdd_pct']:.2f} %")
    print(f"  - 샤프 지수 (Sharpe Ratio): {metrics['sharpe_ratio']:.2f}")
    print("="*60)

def plot_results(portfolio_history, benchmark_data, metrics):
    """백테스트 결과를 시각화합니다."""
    strategy_name = config.PARAMS[config.STRATEGY_TO_RUN]['NAME']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    # 폰트 설정
    if platform.system() == 'Darwin': 
        plt.rc('font', family='AppleGothic')
    else: 
        plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 누적 수익률 그래프
    ax1 = fig.add_subplot(2, 1, 1)
    benchmark_period = benchmark_data.loc[portfolio_history.index]
    benchmark_norm = benchmark_period / benchmark_period.iloc[0] * config.INITIAL_CAPITAL
    
    ax1.plot(portfolio_history.index, portfolio_history['TotalValue'], label=strategy_name, linewidth=2)
    ax1.plot(benchmark_norm.index, benchmark_norm, label='Benchmark', linestyle='--', color='gray')
    
    ax1.set_title(f'누적 수익률 (CAGR: {metrics["cagr_pct"]:.2f}%)', fontsize=16)
    ax1.set_ylabel('자산 가치')
    ax1.legend()

    # 2. Drawdown 그래프
    ax2 = fig.add_subplot(2, 1, 2)
    rolling_max = portfolio_history['TotalValue'].cummax()
    daily_drawdown = portfolio_history['TotalValue'] / rolling_max - 1.0
    
    ax2.fill_between(daily_drawdown.index, daily_drawdown * 100, 0, color='red', alpha=0.3)
    ax2.set_title(f'Drawdown (MDD: {metrics["mdd_pct"]:.2f}%)', fontsize=16)
    ax2.set_ylabel('Drawdown (%)')
    
    plt.tight_layout()
    
    # 그래프 저장
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f"results/{config.STRATEGY_TO_RUN}_backtest_result.png"
    plt.savefig(filename, dpi=150)
    print(f"\n📈 그래프 저장 완료: {filename}")
    
    plt.show()

def save_to_excel(portfolio_history, trade_log, metrics):
    """결과를 엑셀 파일로 저장합니다."""
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f"results/{config.STRATEGY_TO_RUN}_backtest_log.xlsx"

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 요약 시트
            summary_df = pd.DataFrame([metrics])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 일별 자산 시트
            portfolio_history.to_excel(writer, sheet_name='Daily_Portfolio')
            
            # 거래 로그 시트
            trade_log.to_excel(writer, sheet_name='Trade_Log', index=False)
        
        print(f"💾 엑셀 로그 저장 완료: {filename}")
    except Exception as e:
        print(f"❌ 엑셀 저장 실패: {e}")

if __name__ == '__main__':
    # 모듈 단독 테스트
    from data_loader import load_data_for_strategy
    from signals import generate_signals
    from engine import BacktestEngine
    
    strategy = config.STRATEGY_TO_RUN
    price_data, benchmark_data = load_data_for_strategy(strategy)
    investment_signals = generate_signals(price_data, strategy)
    
    engine = BacktestEngine(price_data, investment_signals)
    portfolio_history, trade_log = engine.run()
    
    metrics = analyze_performance(portfolio_history, benchmark_data)
    print_summary(metrics)
    plot_results(portfolio_history, benchmark_data, metrics)
    save_to_excel(portfolio_history, trade_log, metrics)
