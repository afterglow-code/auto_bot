# backtest_v2/engine.py

import pandas as pd
import numpy as np
import config

class BacktestEngine:
    def __init__(self, price_data, signals):
        self.price_data = price_data
        self.signals = signals
        self.capital = config.INITIAL_CAPITAL
        self.commission = config.COMMISSION
        self.slippage = config.SLIPPAGE
        
        self.portfolio_history = []
        self.trade_log = []
        self.holdings = {}

    def run(self):
        """백테스트 시뮬레이션을 실행합니다."""
        print("\n" + "="*50)
        print(f"🚀 백테스트 엔진 실행 시작")
        print("="*50)

        # 시뮬레이션할 날짜 목록 (신호가 있는 첫날부터)
        sim_dates = self.price_data[self.price_data.index >= self.signals.index[0]].index

        for date in sim_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # --- 1. 리밸런싱 실행 (신호가 있는 날에만) ---
            if date in self.signals.index:
                print(f"   - 리밸런싱 실행: {date_str}")
                
                # a. 기존 보유 종목 전량 매도
                if self.holdings:
                    for ticker, qty in list(self.holdings.items()):
                        if ticker in self.price_data.columns and pd.notna(self.price_data.loc[date, ticker]):
                            price = self.price_data.loc[date, ticker]
                            # 슬리피지 적용 (매도 시에는 불리하게)
                            actual_price = price * (1 - self.slippage)
                            sell_value = qty * actual_price
                            fee = sell_value * self.commission
                            self.capital += (sell_value - fee)
                            
                            self.trade_log.append({
                                'Date': date, 'Ticker': ticker, 'Type': 'Sell',
                                'Price': actual_price, 'Qty': qty, 'Value': sell_value
                            })
                    self.holdings = {}

                # b. 새로운 포트폴리오 매수
                target_portfolio = self.signals.loc[date]
                target_assets = target_portfolio[target_portfolio > 0]
                
                # 매수에 사용할 총 자본 (리밸런싱 시점의 총 자산)
                total_asset_before_buy = self.capital 
                
                for ticker, weight in target_assets.items():
                    if ticker in self.price_data.columns and pd.notna(self.price_data.loc[date, ticker]):
                        price = self.price_data.loc[date, ticker]
                        # 슬리피지 적용 (매수 시에는 불리하게)
                        actual_price = price * (1 + self.slippage)
                        budget = total_asset_before_buy * weight
                        
                        if actual_price > 0:
                            qty = int(budget // actual_price)
                            if qty > 0:
                                buy_value = qty * actual_price
                                fee = buy_value * self.commission
                                self.capital -= (buy_value + fee)
                                self.holdings[ticker] = qty
                                
                                self.trade_log.append({
                                    'Date': date, 'Ticker': ticker, 'Type': 'Buy',
                                    'Price': actual_price, 'Qty': qty, 'Value': buy_value
                                })

            # --- 2. 일별 포트폴리오 가치 평가 ---
            current_stock_value = 0
            if self.holdings:
                for ticker, qty in self.holdings.items():
                    if ticker in self.price_data.columns and pd.notna(self.price_data.loc[date, ticker]):
                        current_stock_value += qty * self.price_data.loc[date, ticker]
            
            total_value = self.capital + current_stock_value
            self.portfolio_history.append({'Date': date, 'TotalValue': total_value})

        print("✅ 백테스트 엔진 실행 완료!")
        
        # 결과를 데이터프레임으로 변환
        history_df = pd.DataFrame(self.portfolio_history).set_index('Date')
        log_df = pd.DataFrame(self.trade_log)
        
        return history_df, log_df

if __name__ == '__main__':
    # 모듈 단독 테스트
    from data_loader import load_data_for_strategy
    from signals import generate_signals
    
    strategy = config.STRATEGY_TO_RUN
    price_data, _ = load_data_for_strategy(strategy)
    investment_signals = generate_signals(price_data, strategy)
    
    engine = BacktestEngine(price_data, investment_signals)
    portfolio_history, trade_log = engine.run()
    
    print("\n--- 포트폴리오 자산 변화 샘플 ---")
    print(portfolio_history.head())
    
    print("\n--- 거래 로그 샘플 ---")
    print(trade_log.head())
