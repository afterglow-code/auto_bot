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

class Global_Macro_Backtester:
    def __init__(self, start_date, end_date, initial_capital=1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.commission = 0.00015 
        
        self.history = []
        self.trade_log = []
        
        # [수정] FinanceDataReader용 티커 (뒤에 .KS 제거)
        self.etf_tickers = {
            'KODEX 200': '069500',
            'KODEX 미국나스닥100TR': '379810',
            'ACE 미국S&P500': '360200',
            'KODEX 반도체': '091160',
            'KODEX 헬스케어': '266420',
            'KODEX 미국달러선물': '261240',
            'KODEX AI전력핵심설비' : '487240', #실전용에는 있음
            'ACE 구글벨류체인액티브' : '483340',
            'PLUS K방산': '449170',
            'TIGER 조선TOP10': '494670',
            'KODEX 미국30년국채액티브(H)': '484790',
            #'ACE 인버스' : '145670' 하락장시 달러 잡아서 의미 없음
            #'ACE KRX 금현물': '411060'
        }
        self.data = pd.DataFrame()
        self.kospi_index = None

    def download_data(self):
        # 지표 계산을 위해 365일 전 데이터부터 조회
        target_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        fetch_start_date = target_date - timedelta(days=365)
        fetch_start_str = fetch_start_date.strftime("%Y-%m-%d")
        
        print(f"데이터 다운로드 중 ({fetch_start_str} ~ {self.end_date})...")
        print("※ FDR을 사용하여 1년 치 데이터를 미리 가져옵니다.")

        # 1. KOSPI 지수 (FDR 사용)
        try:
            # KS11: 코스피 지수
            kospi_df = fdr.DataReader('KS11', start=fetch_start_str, end=self.end_date)
            self.kospi_index = kospi_df['Close'].ffill()
        except Exception as e:
            print(f"❌ KOSPI 데이터 실패: {e}")
            return

        # 2. ETF 데이터 (FDR은 반복문으로 수집해야 함)
        df_list = []
        
        for name, code in self.etf_tickers.items():
            try:
                # 데이터 가져오기
                df = fdr.DataReader(code, start=fetch_start_str, end=self.end_date)
                
                # 필요한 'Close' 컬럼만 뽑아서 이름 변경
                series = df['Close'].rename(name)
                df_list.append(series)
                
                # 너무 빠른 요청 방지 (0.1초 대기)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ {name}({code}) 수집 실패: {e}")

        # 3. 데이터 합치기 (가로로 병합)
        if df_list:
            self.data = pd.concat(df_list, axis=1).ffill().dropna()
            print("-> 데이터 준비 완료")
        else:
            print("⛔ 모든 ETF 데이터 수집 실패")

    def run(self):
        print("\n=== 백테스팅 시작 (가중 모멘텀 + TOP 2 분산) ===")
        
        if self.data is None or self.data.empty:
            print("⛔ 중단: 데이터가 없습니다.")
            return pd.DataFrame()

        # [업그레이드 1] 기간별 모멘텀 계산 (가중 평균용)
        # 20일(1달), 60일(3달), 120일(6달)
        mom_1m = self.data.pct_change(20)
        mom_3m = self.data.pct_change(60)
        mom_6m = self.data.pct_change(120)
        
        # [핵심] 종합 점수 = (1개월 + 3개월 + 6개월) / 3
        # 단기, 중기, 장기 추세가 모두 좋은 종목이 높은 점수를 받음
        weighted_score = (mom_1m * 0.3) + (mom_3m * 0.3) + (mom_6m * 0.4)

        # 시장 타이밍용 (기존 유지)
        kospi_ma120 = self.kospi_index.rolling(window=120).mean()

        dates = self.data.index
        
        # 시작일 찾기
        start_idx = 0
        target_start = datetime.strptime(self.start_date, "%Y-%m-%d")
        for i, d in enumerate(dates):
            if d >= target_start:
                start_idx = i
                break
        
        if start_idx == 0 and dates[0] < target_start:
             print("⚠️ 데이터 부족으로 시작일이 조정될 수 있습니다.")

        sim_dates = dates[start_idx:]
        
        holdings = {} 
        prev_month = -1 
        
        for i, date in enumerate(sim_dates):
            current_prices = self.data.loc[date]
            is_trading_day = False
            
            # 월 변경 감지 (리밸런싱)
            if date.month != prev_month:
                is_trading_day = True
                prev_month = date.month
            
            # --- 리밸런싱 실행 ---
            if is_trading_day:
                date_str = date.strftime('%Y-%m-%d')
                
                # 1. 기존 보유 종목 전량 매도
                if holdings:
                    for name, qty in list(holdings.items()):
                        if name in current_prices and not pd.isna(current_prices[name]):
                            price = current_prices[name]
                            sell_val = qty * price
                            fee = sell_val * self.commission
                            self.capital += (sell_val - fee)
                            
                            # 로그 기록
                            self.trade_log.append({
                                '날짜': date_str, '구분': '매도', '종목': name, 
                                '가격': int(price), '수량': qty, '잔고': int(self.capital)
                            })
                    holdings = {} # 잔고 초기화

                # 2. 시장 상황 판단 (상승장/하락장)
                try:
                    k_val = self.kospi_index.asof(date)
                    k_ma = kospi_ma120.asof(date)
                    
                    if hasattr(k_val, 'item'): k_val = k_val.item()
                    if hasattr(k_ma, 'item'): k_ma = k_ma.item()
                    
                    if pd.isna(k_val) or pd.isna(k_ma): is_bull = False
                    else: is_bull = k_val > k_ma
                except: is_bull = False

                # 3. 종목 선정 (TOP 2 전략)
                targets = [] # 매수할 종목 리스트
                
                if is_bull:
                    # 달러 제외하고 점수 산출
                    scores = weighted_score.loc[date].drop('KODEX 미국달러선물', errors='ignore')
                    scores = scores.dropna()
                    
                    if scores.empty:
                        # 살 게 없으면 달러
                        targets = [('KODEX 미국달러선물', 1.0)] # 종목명, 비중(100%)
                    else:
                        # 점수 높은 순서로 정렬
                        top_assets = scores.sort_values(ascending=False)
                        
                        # 1등이 마이너스 점수면 -> 다 하락세 -> 달러
                        if top_assets.iloc[0] < 0:
                            targets = [('KODEX 미국달러선물', 1.0)]
                        else:
                            # [업그레이드 2] 상위 2개 종목 선정
                            # 만약 2등도 점수가 플러스라면 같이 사고, 아니면 1등만 삼
                            selected = []
                            for asset_name, score in top_assets.items():
                                if score > 0:
                                    selected.append(asset_name)
                                if len(selected) >= 2: break
                            
                            if len(selected) == 1:
                                targets = [(selected[0], 1.0)] # 1개면 몰빵
                            elif len(selected) >= 2:
                                targets = [(selected[0], 0.5), (selected[1], 0.5)] # 2개면 반반
                else:
                    # 하락장 -> 달러 방어
                    targets = [('KODEX 미국달러선물', 1.0)]
                
                # 4. 매수 실행
                for target, weight in targets:
                    if target in current_prices and not pd.isna(current_prices[target]):
                        price = current_prices[target]
                        
                        # 할당된 자본금 (비중 * 현재 총자본)
                        alloc_capital = self.capital * weight
                        
                        # 이미 다른 종목 사서 돈이 줄었을 수 있으므로 체크 (마지막 종목용)
                        # 여기서는 간단히 루프 돌기전에 capital을 배분하지 않고, 
                        # 보유 현금 내에서 비중만큼 산다고 가정 (TOP2 동시 매수 위해 로직 필요)
                        
                        # [수정] 정확한 분산 투자를 위해 임시 변수 사용
                        if len(targets) > 1 and weight == 0.5:
                             # 2개 살 때는 현재 현금의 50%씩 사용
                             # 첫 번째 사고 남은 돈의 100%가 아니라, "원래 현금의 50%"여야 함.
                             # 편의상 루프 돌 때마다 현재 self.capital의 weight만큼 산다고 하면 오차가 생김.
                             # -> 총 자본금을 미리 기억해두고 나눔
                             pass 
                
                # [매수 로직 정밀화] 분산 투자를 위해 자본금 배분
                current_cash = self.capital
                buy_log_str = ""
                
                for target, weight in targets:
                    if target in current_prices and not pd.isna(current_prices[target]):
                        price = current_prices[target]
                        
                        # 살 수 있는 금액 책정
                        budget = current_cash * weight
                        
                        if price > 0:
                            qty = int(budget // price)
                            if qty > 0:
                                buy_val = qty * price
                                fee = buy_val * self.commission
                                
                                # 실제 현금 차감
                                self.capital -= (buy_val + fee)
                                holdings[target] = qty
                                
                                # 로그 기록
                                self.trade_log.append({
                                    '날짜': date_str, '구분': '매수', '종목': target, 
                                    '가격': int(price), '수량': qty, '잔고': int(self.capital)
                                })
                                print(f"[매수] {date_str} : {target} {qty}주 (비중 {int(weight*100)}%)")

            # 자산 평가
            stock_val = 0
            for name, qty in holdings.items():
                if name in current_prices and not pd.isna(current_prices[name]):
                    stock_val += qty * current_prices[name]
            
            self.history.append({'Date': date, 'TotalValue': self.capital + stock_val})

        if not self.history:
            return pd.DataFrame()

        self.result_df = pd.DataFrame(self.history).set_index('Date')
        return self.result_df

    def print_trade_log(self):
        if not self.trade_log:
            print("매매 기록이 없습니다.")
            return
        
        print("\n=== 📜 최종 매매 일지 ===")
        df_log = pd.DataFrame(self.trade_log)
        print(df_log[['날짜', '구분', '종목', '수량', '가격', '잔고']].to_string(index=False))

    def plot_result(self):
        if self.result_df is None or self.result_df.empty: return
        final_val = self.result_df['TotalValue'].iloc[-1]
        earning_rate = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        k_series = self.kospi_index.loc[self.result_df.index]
        k_norm = k_series / k_series.iloc[0] * self.initial_capital

        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df.index, self.result_df['TotalValue'], label='내 전략', color='red')
        plt.plot(k_norm.index, k_norm, label='KOSPI 지수', color='gray', linestyle='--')
        
        plt.title(f"수익률: {earning_rate:.2f}%")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # FinanceDataReader는 데이터 제한이 없으므로 과거~현재까지 테스트 가능

    bt = Global_Macro_Backtester(start_date='2023-01-01', end_date='2025-12-01')
    bt.download_data()
    bt.run()
    bt.print_trade_log()
    bt.plot_result()