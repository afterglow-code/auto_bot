import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import platform

# 폰트 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 설정값 수정 ---
MOSIG_TOP_N_KOSPI = 50   
MOSIG_TOP_N_KOSDAQ = 50  
MAX_SLOTS = 3            

# [수정된 파라미터]
STOP_LOSS_PCT = 0.05     # 고정 손절 -5% (유지)
ATR_TARGET_MULT = 3.0    # [변경] 5.0 -> 3.0 (욕심을 줄여서 승률을 확보)
BREAKEVEN_TRIGGER = 0.05 # 5% 수익 시 본전 설정 (유지)
VOLUME_MULTIPLIER = 2.0  # 거래량 2배 (유지)

class MosigHybridBacktester:
    def __init__(self, start_date, end_date, initial_capital=10000000, commission=0.00015, slippage=0.0001):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.history = []   
        self.trade_log = [] 
        self.target_stocks = {} 
        self.stock_db = {} 
        self.kospi_index = None
        self.holdings = {} 

    def fetch_target_stocks(self):
        print("📊 [1/3] 종목 리스트 확보 중...")
        try:
            df_kospi = fdr.StockListing('KOSPI').sort_values('Marcap', ascending=False).head(MOSIG_TOP_N_KOSPI)
            df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('Marcap', ascending=False).head(MOSIG_TOP_N_KOSDAQ)
            for _, row in df_kospi.iterrows(): self.target_stocks[row['Name']] = row['Code']
            for _, row in df_kosdaq.iterrows(): self.target_stocks[row['Name']] = row['Code']
        except Exception:
            pass

    def calculate_atr(self, df, window=20):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()

    def download_data(self):
        fetch_start_date = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=90)
        fetch_start_str = fetch_start_date.strftime("%Y-%m-%d")
        
        print(f"📊 [2/3] 데이터 다운로드 및 ATR 계산 ({fetch_start_str} ~ )...")
        self.kospi_index = fdr.DataReader('KS11', start=fetch_start_str, end=self.end_date)['Close']

        total_count = len(self.target_stocks)
        for i, (name, code) in enumerate(self.target_stocks.items()):
            try:
                if i % 10 == 0: print(f"   진행률: {int((i/total_count)*100)}%", end='\r')
                
                df = fdr.DataReader(code, start=fetch_start_str, end=self.end_date)
                if df.empty or len(df) < 60: continue

                df['Momentum'] = (df['Close'] / df['Close'].shift(10)) * 100
                df['Signal'] = df['Momentum'].rolling(window=9).mean()
                df['Vol_Prev'] = df['Volume'].shift(1)
                
                # ATR 계산 (익절 목표가용)
                df['ATR'] = self.calculate_atr(df)
                
                self.stock_db[name] = df
            except Exception:
                pass 
        print(f"\n-> {len(self.stock_db)}개 종목 데이터 준비 완료")

    def run(self):
        print(f"📊 [3/3] 백테스팅 시작 (ATR 익절 + 고정 손절 -5%)")
        
        full_dates = self.kospi_index.index
        sim_dates = full_dates[(full_dates >= self.start_date) & (full_dates <= self.end_date)]

        for current_date in sim_dates:
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # 1. 매도 (Sell)
            stocks_to_sell = []
            
            for name, info in self.holdings.items():
                if name not in self.stock_db: continue
                df = self.stock_db[name]
                if current_date not in df.index: continue
                
                daily_data = df.loc[current_date]
                buy_price = info['buy_price']
                stop_price = info['stop_price']
                target_price = info['target_price'] # ATR로 계산된 목표가
                qty = info['qty']
                
                current_low = daily_data['Low']
                current_high = daily_data['High']
                
                sell_type = None
                sell_price = 0

                # A. 손절 (Fixed -5%)
                if current_low <= stop_price:
                    sell_price = stop_price * (1 - self.slippage)
                    sell_type = '손절'
                
                # B. 익절 (ATR Target)
                elif current_high >= target_price:
                    sell_price = target_price * (1 - self.slippage)
                    sell_type = '익절'
                
                # C. 본전 설정 (Trailing Stop)
                elif not info['is_breakeven']:
                    trigger_price = buy_price * (1 + BREAKEVEN_TRIGGER)
                    if current_high >= trigger_price:
                        # 손절가를 매수가(본전)로 상향
                        self.holdings[name]['stop_price'] = buy_price * 1.005 
                        self.holdings[name]['is_breakeven'] = True
                        
                if sell_type:
                    sell_amt = qty * sell_price
                    fee = sell_amt * self.commission
                    self.capital += (sell_amt - fee)
                    
                    profit_rate = (sell_price - buy_price) / buy_price * 100
                    if sell_type == '손절' and info['is_breakeven']: sell_type = '본전컷'

                    self.trade_log.append({
                        '날짜': current_date_str, '구분': sell_type, '종목': name,
                        '가격': int(sell_price), '수량': qty, '수익률': f"{profit_rate:.2f}%",
                        '잔고': int(self.capital)
                    })
                    stocks_to_sell.append(name)
                    print(f"[{current_date_str}] {sell_type}: {name} ({profit_rate:.2f}%)")

            for name in stocks_to_sell: del self.holdings[name]

            # 2. 매수 (Buy)
            if len(self.holdings) < MAX_SLOTS:
                buy_candidates = []

                for name, df in self.stock_db.items():
                    if name in self.holdings: continue 
                    if current_date not in df.index: continue

                    try:
                        today_idx = df.index.get_loc(current_date)
                        if today_idx < 15: continue
                    except KeyError: continue

                    today = df.iloc[today_idx]
                    yesterday = df.iloc[today_idx - 1]

                    if pd.isna(today['Momentum']) or pd.isna(today['ATR']): continue

                    # [조건] 모멘텀 돌파 + 거래량 2배
                    is_breakout = (today['Momentum'] >= 100) and \
                                  (yesterday['Momentum'] < 100) and \
                                  (today['Momentum'] > today['Signal'])
                    
                    is_volume_spike = (today['Volume'] >= today['Vol_Prev'] * VOLUME_MULTIPLIER)

                    if is_breakout and is_volume_spike:
                        buy_candidates.append({
                            'name': name,
                            'momentum': today['Momentum'],
                            'price': today['Close'],
                            'atr': today['ATR']
                        })

                if buy_candidates:
                    buy_candidates.sort(key=lambda x: x['momentum'], reverse=True)
                    
                    for candidate in buy_candidates:
                        if len(self.holdings) >= MAX_SLOTS: break 
                        
                        slots_available = MAX_SLOTS - len(self.holdings)
                        invest_amt = self.capital / slots_available
                        if invest_amt < 100000: break

                        buy_price = candidate['price'] * (1 + self.slippage)
                        qty = int(invest_amt // buy_price)
                        atr = candidate['atr']
                        
                        if qty > 0:
                            cost = qty * buy_price
                            fee = cost * self.commission
                            
                            if self.capital >= (cost + fee):
                                self.capital -= (cost + fee)
                                
                                # [핵심] 익절은 ATR 기반 / 손절은 고정 비율
                                target_price = buy_price + (atr * ATR_TARGET_MULT)
                                stop_price = buy_price * (1 - STOP_LOSS_PCT) # 고정 -5%
                                
                                self.holdings[candidate['name']] = {
                                    'qty': qty, 
                                    'buy_price': buy_price,
                                    'target_price': target_price,
                                    'stop_price': stop_price,
                                    'is_breakeven': False
                                }
                                
                                self.trade_log.append({
                                    '날짜': current_date_str, '구분': '매수', '종목': candidate['name'],
                                    '가격': int(buy_price), '수량': qty, '수익률': '-',
                                    '잔고': int(self.capital)
                                })

            current_holdings_val = 0
            for name, info in self.holdings.items():
                if name in self.stock_db and current_date in self.stock_db[name].index:
                    current_holdings_val += info['qty'] * self.stock_db[name].loc[current_date]['Close']
                else:
                    current_holdings_val += info['qty'] * info['buy_price']

            total_val = self.capital + current_holdings_val
            self.history.append({'Date': current_date, 'TotalValue': total_val})

        self.result_df = pd.DataFrame(self.history).set_index('Date')
        return self.result_df

    def print_result(self):
        if self.result_df is None or self.result_df.empty:
            print("❌ 결과 없음")
            return

        final_val = self.result_df['TotalValue'].iloc[-1]
        profit_rate = (final_val - self.initial_capital) / self.initial_capital * 100
        peak = self.result_df['TotalValue'].cummax()
        mdd = ((self.result_df['TotalValue'] - peak) / peak).min() * 100

        win = len([x for x in self.trade_log if x['구분'] == '익절'])
        loss = len([x for x in self.trade_log if x['구분'] == '손절'])
        be = len([x for x in self.trade_log if x['구분'] == '본전컷'])
        total = win + loss + be
        win_rate = (win / total * 100) if total > 0 else 0

        print("\n" + "="*40)
        print("📊 [Hybrid 백테스트 리포트]")
        print("="*40)
        print(f"최종 자본 : {int(final_val):,}원 (수익률: {profit_rate:.2f}%)")
        print(f"M D D    : {mdd:.2f}%")
        print(f"거래 횟수 : {total}회 (익절 {win} / 본전 {be} / 손절 {loss})")
        print(f"승률     : {win_rate:.2f}% (본전 포함 방어율: {((win+be)/total*100):.2f}%)")
        print("="*40)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df.index, self.result_df['TotalValue'], label='Hybrid Strategy (ATR Target + Fixed Stop)')
        if self.kospi_index is not None:
            k_norm = self.kospi_index.reindex(self.result_df.index).ffill()
            k_norm = k_norm / k_norm.iloc[0] * self.initial_capital
            plt.plot(k_norm.index, k_norm, label='KOSPI', color='gray', linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # 2021~2023 3년치 테스트
    bt = MosigHybridBacktester(start_date='2021-01-01', end_date='2023-12-31')
    bt.fetch_target_stocks()
    bt.download_data()
    bt.run()
    bt.print_result()