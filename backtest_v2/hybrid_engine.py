# backtest_v2/hybrid_engine.py

import pandas as pd
import numpy as np
import config

class HybridEngine:
    def __init__(self, ohlcv_data, strategy_name):
        self.ohlcv = ohlcv_data
        self.close = ohlcv_data['Close']
        self.high = ohlcv_data['High']
        self.low = ohlcv_data['Low']
        self.volume = ohlcv_data['Volume']
        
        self.strategy_name = strategy_name
        self.cfg = config.PARAMS[strategy_name]
        self.hp = config.HYBRID_PARAMS
        
        self.capital = config.INITIAL_CAPITAL
        self.commission = config.COMMISSION
        self.slippage = config.SLIPPAGE
        
        self.holdings = {} 
        self.history = []
        self.trade_log = []

    def calculate_indicators(self):
        print("⚙️ 지표 계산 중...")
        # 1. 모멘텀 및 시그널
        self.momentum = (self.close / self.close.shift(10)) * 100
        self.signal = self.momentum.rolling(window=9).mean()
        
        # 2. ATR 계산
        prev_close = self.close.shift(1)
        tr = np.maximum(self.high - self.low, 
                        np.maximum(np.abs(self.high - prev_close), 
                                   np.abs(self.low - prev_close)))
        self.atr = tr.rolling(window=self.hp['ATR_WINDOW']).mean()
        self.vol_prev = self.volume.shift(1)

    def run(self):
        self.calculate_indicators()
        
        print(f"\n🚀 Hybrid 3.0 실행 (ATR 목표 + 본전설정 + 주도주 홀딩)")
        sim_dates = self.close.index[self.close.index >= config.START_DATE]

        for date in sim_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # --- 0. 당일의 주도주 순위 계산 (매일 업데이트) ---
            daily_mom = self.momentum.loc[date]
            # 모멘텀이 0보다 크고 결측치 없는 종목들 중 상위 TOP_N 추출
            current_top_n = daily_mom[daily_mom > 0].sort_values(ascending=False).head(self.cfg['TOP_N']).index.tolist()

            # --- 1. 매도 로직 ---
            for ticker in list(self.holdings.keys()):
                info = self.holdings[ticker]
                if ticker not in self.low.columns or pd.isna(self.low.loc[date, ticker]):
                    continue
                    
                curr_low = self.low.loc[date, ticker]
                curr_high = self.high.loc[date, ticker]
                
                # A. 하드 스탑 (-5%) 또는 본전 스탑 체크
                if curr_low <= info['stop_price']:
                    sell_price = info['stop_price'] * (1 - self.slippage)
                    revenue = info['qty'] * sell_price
                    self.capital += (revenue - (revenue * self.commission))
                    
                    sell_type = 'StopLoss' if not info['is_breakeven'] else 'BreakevenStop'
                    self.trade_log.append({'Date': date, 'Ticker': ticker, 'Type': sell_type, 'Price': sell_price, 'Qty': info['qty'], 'Value': revenue})
                    del self.holdings[ticker]
                    continue

                # B. 본전 설정 트리거 (+5% 도달 시)
                if not info['is_breakeven']:
                    if curr_high >= info['buy_price'] * (1 + self.hp['BREAKEVEN_TRIGGER']):
                        self.holdings[ticker]['stop_price'] = info['buy_price'] * 1.005 # 본전+수수료로 스탑 상향
                        self.holdings[ticker]['is_breakeven'] = True

                # C. [핵심] ATR 목표 달성 및 주도주 이탈 체크
                # 이미 목표가에 도달한 적이 있는지 확인
                if curr_high >= info['target_price']:
                    self.holdings[ticker]['target_reached'] = True
                
                if info.get('target_reached', False):
                    # 목표가는 넘었으나, 여전히 TOP N 이라면? -> 홀딩 (팔지 않음)
                    if ticker in current_top_n:
                        pass 
                    else:
                        # 목표가도 넘었고, 순위에서도 밀려났다면? -> 익절
                        sell_price = self.close.loc[date, ticker] * (1 - self.slippage)
                        revenue = info['qty'] * sell_price
                        self.capital += (revenue - (revenue * self.commission))
                        
                        self.trade_log.append({'Date': date, 'Ticker': ticker, 'Type': 'TakeProfit(ExitRank)', 'Price': sell_price, 'Qty': info['qty'], 'Value': revenue})
                        del self.holdings[ticker]

            # --- 2. 매수 로직 (거래량 2배 원칙) ---
            if len(self.holdings) < self.hp['MAX_SLOTS']:
                # 오늘 신호가 뜬 후보군
                candidates = []
                prev_mom = self.momentum.shift(1).loc[date]
                today_sig = self.signal.loc[date]
                today_vol = self.volume.loc[date]
                prev_vol = self.vol_prev.loc[date]
                
                for ticker in current_top_n: # 주도주 순위 안에 있는 종목만 검토
                    if ticker in self.holdings: continue
                    
                    # 진입 조건: 모멘텀 돌파 + 거래량 2.0배
                    is_breakout = (daily_mom[ticker] >= 100) and (prev_mom[ticker] < 100)
                    is_strong = (daily_mom[ticker] >= 100) and (daily_mom[ticker] > today_sig[ticker])
                    is_volume_spike = (today_vol[ticker] >= prev_vol[ticker] * 2.0)
                    
                    if (is_breakout or is_strong) and is_volume_spike:
                        candidates.append({'ticker': ticker, 'momentum': daily_mom[ticker], 'close': self.close.loc[date, ticker], 'atr': self.atr.loc[date, ticker]})

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
                        self.trade_log.append({'Date': date, 'Ticker': cand['ticker'], 'Type': 'Buy', 'Price': buy_price, 'Qty': qty, 'Value': qty * buy_price})

            # 3. 평가
            curr_val = self.capital
            for t, info in self.holdings.items():
                curr_val += info['qty'] * self.close.loc[date, t]
            self.history.append({'Date': date, 'TotalValue': curr_val})

        return pd.DataFrame(self.history).set_index('Date'), pd.DataFrame(self.trade_log)