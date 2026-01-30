import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import platform
import time

# 폰트 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

class USTopStocks_Backtester:
    def __init__(self, start_date, end_date, initial_capital=10000): # 자본금 $10,000 (달러 기준)
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.commission = 0.00015 # 미국 주식 수수료 (가정)
        
        self.history = []
        self.trade_log = []
        self.target_tickers = {} 
        self.market_index = None # SPY (S&P500)
        self.data = pd.DataFrame()

    def fetch_top_stocks(self):
        print("📊 미국 우량주(S&P 500) 리스트 확보 중...")
        
        # S&P 500 리스트 가져오기
        df_sp500 = fdr.StockListing('S&P500')
        top_stocks = df_sp500.head(200) 
        
        for _, row in top_stocks.iterrows():
            self.target_tickers[row['Symbol']] = row['Symbol']
            
        # [필수] 하락장 방어용: 초단기 국채 ETF (BIL)
        self.target_tickers['BIL'] = 'BIL'
        
        print(f"-> 총 {len(self.target_tickers)}개 종목 (S&P 500 Top 200 + BIL) 준비 완료")

    def download_data(self):
        target_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        fetch_start_date = target_date - timedelta(days=365)
        fetch_start_str = fetch_start_date.strftime("%Y-%m-%d")
        
        print(f"\n데이터 다운로드 시작 ({fetch_start_str} ~ )... 미국 서버라 느릴 수 있습니다!")

        # 1. 시장 지수 (SPY)
        spy_df = fdr.DataReader('SPY', start=fetch_start_str, end=self.end_date)
        self.market_index = spy_df['Close'].ffill()

        # 2. 개별 종목 데이터 수집
        df_list = []
        total_count = len(self.target_tickers)
        
        for i, (name, code) in enumerate(self.target_tickers.items()):
            try:
                print(f"\r[{i+1}/{total_count}] {code} 수집 중...", end='')
                
                df = fdr.DataReader(code, start=fetch_start_str, end=self.end_date)
                if df.empty: continue

                series = df['Close'].rename(code)
                df_list.append(series)
                time.sleep(0.1) 
                
            except Exception as e:
                pass 
        
        print("\n-> 데이터 병합 중...")
        if df_list:
            self.data = pd.concat(df_list, axis=1).ffill().dropna(how='all')
            print("-> 데이터 준비 완료!")
        else:
            print("⛔ 데이터 수집 실패")

    def run(self):
        print("\n=== 백테스팅 시작 (US TOP 200 Universe) ===")
        
        if self.data is None or self.data.empty: return

        # 가중 평균 모멘텀
        mom_1m = self.data.pct_change(20)
        mom_3m = self.data.pct_change(60)
        mom_6m = self.data.pct_change(120)
        
        weighted_score = ((mom_1m.fillna(0) * 0.2) + (mom_3m.fillna(0) * 0.3) + (mom_6m.fillna(0) * 0.5))

        # 시장 타이밍: SPY의 120일 이평선
        spy_ma120 = self.market_index.rolling(window=120).mean()
        dates = self.data.index
        
        start_idx = 0
        target_start = datetime.strptime(self.start_date, "%Y-%m-%d")
        for i, d in enumerate(dates):
            if d >= target_start:
                start_idx = i
                break
        sim_dates = dates[start_idx:]
        
        holdings = {} 
        prev_month = -1 
        
        for i, date in enumerate(sim_dates):
            current_prices = self.data.loc[date].dropna()
            
            is_trading_day = False
            if date.month != prev_month:
                is_trading_day = True
                prev_month = date.month
            
            if is_trading_day:
                date_str = date.strftime('%Y-%m-%d')
                
                # 1. 매도
                if holdings:
                    for name, qty in list(holdings.items()):
                        if name in current_prices:
                            price = current_prices[name]
                            sell_val = qty * price
                            fee = sell_val * self.commission
                            self.capital += (sell_val - fee)
                            
                            self.trade_log.append({
                                '날짜': date_str, '구분': '매도', '종목': name, 
                                '가격': float(price), '수량': qty, '잔고': float(self.capital)
                            })
                    holdings = {}

                # 2. 시장 판단 (S&P 500 기준)
                try:
                    m_val = self.market_index.asof(date)
                    m_ma = spy_ma120.asof(date)
                    
                    if hasattr(m_val, 'item'): m_val = m_val.item()
                    if hasattr(m_ma, 'item'): m_ma = m_ma.item()

                    if pd.isna(m_val) or pd.isna(m_ma): is_bull = False
                    else: is_bull = m_val > m_ma
                except: is_bull = False

                # 3. 종목 선정 (TOP 3 분산)
                targets = []
                
                if is_bull:
                    valid_tickers = current_prices.index
                    scores = weighted_score.loc[date].reindex(valid_tickers).drop('BIL', errors='ignore')
                    scores = scores.dropna().sort_values(ascending=False)
                    
                    if scores.empty or scores.iloc[0] <= 0:
                        targets = [('BIL', 1.0)]
                    else:
                        selected = []
                        for name, score in scores.items():
                            if score > 0: selected.append(name)
                            if len(selected) >= 3: break 
                        
                        count = len(selected)
                        if count > 0:
                            weight = 1.0 / count
                            for s in selected:
                                targets.append((s, weight))
                else:
                    targets = [('BIL', 1.0)]
                
                # 4. 매수
                current_cash = self.capital
                for target, weight in targets:
                    if target in current_prices:
                        price = current_prices[target]
                        budget = current_cash * weight
                        if price > 0:
                            qty = int(budget // price)
                            if qty > 0:
                                buy_val = qty * price
                                fee = buy_val * self.commission
                                self.capital -= (buy_val + fee)
                                holdings[target] = qty
                                
                                print(f"[매수] {date_str} : {target} {qty}주 (${price:.2f})")
                                self.trade_log.append({
                                    '날짜': date_str, '구분': '매수', '종목': target, 
                                    '가격': float(price), '수량': qty, '잔고': float(self.capital)
                                })

            # 평가
            stock_val = 0
            for name, qty in holdings.items():
                if name in current_prices:
                    stock_val += qty * current_prices[name]
            
            self.history.append({'Date': date, 'TotalValue': self.capital + stock_val})

        if not self.history: return pd.DataFrame()
        self.result_df = pd.DataFrame(self.history).set_index('Date')
        return self.result_df

    # [추가됨] 엑셀 저장 함수
    def save_log_to_excel(self, filename="US_Trade_Log.xlsx"):
        if not self.trade_log:
            print("⚠️ 저장할 매매 기록이 없습니다.")
            return

        print(f"\n💾 엑셀 파일로 저장 중... ({filename})")
        try:
            df_log = pd.DataFrame(self.trade_log)
            # 보기 좋게 컬럼 순서 지정
            cols = ['날짜', '구분', '종목', '가격', '수량', '잔고']
            # 데이터프레임에 해당 컬럼들이 다 있는지 확인 후 정렬
            if all(c in df_log.columns for c in cols):
                df_log = df_log[cols]
                
            df_log.to_excel(filename, index=False)
            print(f"✅ 저장 완료! 파일을 확인하세요: {filename}")
        except Exception as e:
            print(f"❌ 엑셀 저장 실패: {e}")
            print("팁: 'pip install openpyxl'을 설치했는지 확인해보세요.")

    def plot_result(self):
        if self.result_df is None or self.result_df.empty: return
        final_val = self.result_df['TotalValue'].iloc[-1]
        earning_rate = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        spy_series = self.market_index.loc[self.result_df.index]
        spy_norm = spy_series / spy_series.iloc[0] * self.initial_capital

        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df.index, self.result_df['TotalValue'], label='US Momentum Strategy', color='blue')
        plt.plot(spy_norm.index, spy_norm, label='S&P 500 (SPY)', color='gray', linestyle='--')
        
        plt.title(f"CAGR Result: {earning_rate:.2f}% (Capital ${self.initial_capital:,} -> ${int(final_val):,})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # 자본금 $10,000로 시작
    bt = USTopStocks_Backtester(start_date='2020-01-01', end_date='2026-01-01', initial_capital=10000)
    
    # S&P 500 리스트 확보
    bt.fetch_top_stocks()
    
    # 데이터 다운로드 (시간 꽤 걸림)
    bt.download_data()
    
    # 실행
    bt.run()
    
    # [추가됨] 엑셀로 저장
    bt.save_log_to_excel("US_Momentum_Trade_Log.xlsx")
    
    # 그래프 출력
    bt.plot_result()