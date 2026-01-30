# backtest_v2/main.py

import config
from data_loader import load_data_for_strategy
from signals import generate_signals
from engine import BacktestEngine
from reporting import analyze_performance, print_summary, plot_results, save_to_excel

def main():
    """통합 백테스트 프레임워크의 메인 실행 함수"""
    
    # 1. 설정에서 테스트할 전략 이름 가져오기
    strategy_name = config.STRATEGY_TO_RUN
    
    # 2. 데이터 로딩
    # 선택된 전략에 필요한 가격 데이터와 벤치마크 데이터를 가져옴
    price_data, benchmark_data = load_data_for_strategy(strategy_name)
    
    if price_data.empty:
        print("❌ 데이터 로딩 실패. 백테스트를 중단합니다.")
        return
        
    # 3. 투자 신호 생성
    # 가격 데이터를 기반으로 리밸런싱 날짜와 종목별 비중을 계산
    investment_signals = generate_signals(price_data, benchmark_data, strategy_name)
    
    # 4. 백테스트 엔진 실행
    # 생성된 신호에 따라 매매를 시뮬레이션하고 결과(자산 변화, 거래 로그)를 기록
    engine = BacktestEngine(price_data, investment_signals)
    portfolio_history, trade_log = engine.run()
    
    if portfolio_history.empty:
        print("❌ 백테스트 실행 중 오류가 발생했습니다.")
        return
        
    # 5. 성과 분석 및 리포팅
    # 백테스트 결과를 분석하고, 요약 출력, 그래프 생성, 엑셀 저장
    metrics = analyze_performance(portfolio_history, benchmark_data)
    print_summary(metrics)
    plot_results(portfolio_history, benchmark_data, metrics)
    save_to_excel(portfolio_history, trade_log, metrics)
    
    print("\n🎉 모든 백테스트 과정이 성공적으로 완료되었습니다.")

if __name__ == '__main__':
    main()
