import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import platform
import time
import os

# 폰트 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

class TopStocks_Backtester:
    def __init__(self, start_date, end_date, initial_capital=10000000): 
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.commission = 0.00015 
        
        self.history = []   # 일별 자산 흐름
        self.trade_log = [] # 매매 일지
        self.target_tickers = {} 
        self.kospi_index = None
        self.data = pd.DataFrame()

    def fetch_top_stocks(self):
        print("📊 시가총액 상위 종목 리스트 확보 중...")
        
        # 1. KOSPI 상위 100개
        df_kospi = fdr.StockListing('KOSPI')
        top_kospi = df_kospi.sort_values('Marcap', ascending=False).head(100)
        
        # 2. KOSDAQ 상위 100개
        df_kosdaq = fdr.StockListing('KOSDAQ')
        top_kosdaq = df_kosdaq.sort_values('Marcap', ascending=False).head(100)
        
        # 3. 딕셔너리로 변환
        for _, row in top_kospi.iterrows():
            self.target_tickers[row['Name']] = row['Code']
            
        for _, row in top_kosdaq.iterrows():
            self.target_tickers[row['Name']] = row['Code']
            
        # [필수] 하락장 방어용 달러 추가
        self.target_tickers['KODEX 미국달러선물'] = '261240'
        
        print(f"-> 총 {len(self.target_tickers)}개 종목 (KOSPI 100 + KOSDAQ 100 + 달러) 준비 완료")

    def download_data(self):
        target_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        fetch_start_date = target_date - timedelta(days=365)
        fetch_start_str = fetch_start_date.strftime("%Y-%m-%d")
        
        print(f"\n데이터 다운로드 시작 ({fetch_start_str} ~ )... 시간 좀 걸립니다!")

        # 1. KOSPI 지수
        kospi_df = fdr.DataReader('KS11', start=fetch_start_str, end=self.end_date)
        self.kospi_index = kospi_df['Close'].ffill()

        # 2. 개별 종목 데이터 수집
        df_list = []
        total_count = len(self.target_tickers)
        
        for i, (name, code) in enumerate(self.target_tickers.items()):
            try:
                if i % 10 == 0: print(f"\r[{i+1}/{total_count}] 데이터 수집 중...", end='')
                
                df = fdr.DataReader(code, start=fetch_start_str, end=self.end_date)
                if df.empty: continue

                series = df['Close'].rename(name)
                df_list.append(series)
                time.sleep(0.05) 
                
            except Exception as e:
                pass 
        
        print("\n-> 데이터 병합 중...")
        if df_list:
            self.data = pd.concat(df_list, axis=1).ffill().dropna(how='all')
            print("-> 데이터 준비 완료!")
        else:
            print("⛔ 데이터 수집 실패")

    def run(self):
        print("\n=== 백테스팅 시작 (60일선 기준 / 변동성 조절 모멘텀) ===")
        
        if self.data is None or self.data.empty: return

        # -------------------------------------------------------------
        # [변동성 조절 모멘텀 스코어 계산]
        # -------------------------------------------------------------
        daily_rets = self.data.pct_change()
        ret_3m = self.data.pct_change(60)
        ret_6m = self.data.pct_change(120)

        vol_1m = daily_rets.rolling(20).std()
        vol_3m = daily_rets.rolling(60).std()
        vol_6m = daily_rets.rolling(120).std()

        epsilon = 1e-6
        score_3m = ret_3m / (vol_3m + epsilon)
        score_6m = ret_6m / (vol_3m + epsilon)

        weighted_score = (score_3m.fillna(0) * 0.5) + (score_6m.fillna(0) * 0.5)
        # -------------------------------------------------------------

        # [수정 포인트] 시장 기준을 120일 -> 60일로 변경
        # 60일선은 '수급선'이라고 불리며 중기 추세를 판단하는 핵심 지표입니다.
        kospi_ma60 = self.kospi_index.rolling(window=60).mean()
        
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
                
                # -----------------------------------------------------
                # 1. 매도 로직 (Sell)
                # -----------------------------------------------------
                if holdings:
                    for name, qty in list(holdings.items()):
                        if name in current_prices:
                            price = current_prices[name]
                            sell_val = qty * price
                            fee = sell_val * self.commission
                            self.capital += (sell_val - fee)
                            
                            print(f"[매도] {date_str} : {name} {qty}주 (평가액 {int(sell_val):,}원)")
                            
                            self.trade_log.append({
                                '날짜': date_str, '구분': '매도', '종목': name, 
                                '가격': int(price), '수량': qty, '잔고': int(self.capital)
                            })
                    holdings = {} 

                # -----------------------------------------------------
                # 2. 시장 판단 및 종목 선정 (60일선 기준 적용)
                # -----------------------------------------------------
                try:
                    k_val = self.kospi_index.asof(date)
                    k_ma = kospi_ma60.asof(date) # 60일 이동평균값 사용
                    
                    if pd.isna(k_val) or pd.isna(k_ma): is_bull = False
                    else: is_bull = k_val > k_ma # 주가가 60일선 위에 있으면 상승장
                except: is_bull = False

                targets = []
                
                if is_bull:
                    valid_tickers = current_prices.index
                    scores = weighted_score.loc[date].reindex(valid_tickers).drop('KODEX 미국달러선물', errors='ignore')
                    scores = scores.dropna().sort_values(ascending=False)
                    
                    if scores.empty or scores.iloc[0] <= 0:
                        targets = [('KODEX 미국달러선물', 1.0)]
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
                    targets = [('KODEX 미국달러선물', 1.0)]
                
                # -----------------------------------------------------
                # 3. 매수 로직 (Buy)
                # -----------------------------------------------------
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
                                
                                print(f"[매수] {date_str} : {target} {qty}주")
                                self.trade_log.append({
                                    '날짜': date_str, '구분': '매수', '종목': target, 
                                    '가격': int(price), '수량': qty, '잔고': int(self.capital)
                                })

            # 일별 자산 평가
            stock_val = 0
            for name, qty in holdings.items():
                if name in current_prices:
                    stock_val += qty * current_prices[name]
            
            total_val = self.capital + stock_val
            self.history.append({'Date': date, 'TotalValue': total_val})

        if not self.history: return pd.DataFrame()
        self.result_df = pd.DataFrame(self.history).set_index('Date')
        return self.result_df

    # ------------------------------------------------------------------
    # [추가] 엑셀 저장 및 성과 분석 함수
    # ------------------------------------------------------------------
    def save_results_to_excel(self, filename="Korea_Stock_Backtest_Result.xlsx"):
        if self.result_df is None or self.result_df.empty:
            print("❌ 결과 데이터가 없습니다.")
            return

        print(f"\n💾 엑셀 저장 중... ({filename})")
        
        # 1. 성과 분석 (CAGR, MDD, 수익률)
        final_val = self.result_df['TotalValue'].iloc[-1]
        total_return = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        # CAGR 계산 (연평균 성장률)
        days = (self.result_df.index[-1] - self.result_df.index[0]).days
        years = days / 365.25
        cagr = ((final_val / self.initial_capital) ** (1/years) - 1) * 100
        
        # MDD 계산
        historical_max = self.result_df['TotalValue'].cummax()
        daily_drawdown = self.result_df['TotalValue'] / historical_max - 1.0
        mdd = daily_drawdown.min() * 100

        # 요약 데이터프레임 생성
        summary_data = {
            '항목': ['초기 자본금', '최종 자산', '총 수익률', '연평균 수익률(CAGR)', '최대 낙폭(MDD)', '시작일', '종료일'],
            '값': [
                f"{int(self.initial_capital):,}원",
                f"{int(final_val):,}원",
                f"{total_return:.2f}%",
                f"{cagr:.2f}%",
                f"{mdd:.2f}%",
                self.start_date,
                self.end_date
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        # 2. 엑셀 쓰기 (멀티 시트)
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 시트 1: 요약 정보
                df_summary.to_excel(writer, sheet_name='성과요약', index=False)
                
                # 시트 2: 매매 일지
                df_log = pd.DataFrame(self.trade_log)
                if not df_log.empty:
                    df_log = df_log[['날짜', '구분', '종목', '가격', '수량', '잔고']]
                    df_log.to_excel(writer, sheet_name='매매일지', index=False)
                
                # 시트 3: 일별 자산 추이
                self.result_df.to_excel(writer, sheet_name='일별자산추이')
                
            print(f"✅ 저장 완료! [수익률: {total_return:.2f}% / MDD: {mdd:.2f}%]")
            print(f"📂 파일 위치: {os.path.abspath(filename)}")
            
        except Exception as e:
            print(f"❌ 엑셀 저장 실패: {e}")
            print("👉 'pip install openpyxl' 명령어로 라이브러리를 설치해주세요.")

    def plot_result(self):
        if self.result_df is None or self.result_df.empty: return
        
        # 성과 지표 계산
        final_val = self.result_df['TotalValue'].iloc[-1]
        earning_rate = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        # 벤치마크 (KOSPI)
        k_series = self.kospi_index.loc[self.result_df.index]
        k_norm = k_series / k_series.iloc[0] * self.initial_capital

        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df.index, self.result_df['TotalValue'], label='Risk-Adjusted Momentum', color='blue', linewidth=2)
        plt.plot(k_norm.index, k_norm, label='KOSPI Index', color='gray', linestyle='--')
        
        plt.title(f"Backtest Result: Return {earning_rate:.2f}% (Cap {int(self.initial_capital/10000)}만 -> {int(final_val/10000)}만)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # 백테스팅 실행
    bt = TopStocks_Backtester(start_date='2023-01-01', end_date='2026-01-02')
    
    bt.fetch_top_stocks()
    bt.download_data()
    bt.run()
    
    # [중요] 엑셀로 저장
    bt.save_results_to_excel()
    
    # 그래프 출력
    bt.plot_result()