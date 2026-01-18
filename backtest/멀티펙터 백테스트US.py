import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import platform
import time

if platform.system() == 'Darwin': 
    plt.rc('font', family='AppleGothic')
else: 
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


class USMultiFactorBacktest:
    """미국 주식 멀티팩터 전략 백테스트"""
    
    def __init__(self, start_date, end_date, initial_capital=10000,
                 momentum_weight=0.5, value_weight=0.25, 
                 quality_weight=0.25, num_stocks=5):
        
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.commission = 0.00015
        
        # 멀티팩터 파라미터 (미국 최적화)
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.num_stocks = num_stocks
        
        self.history = []
        self.trade_log = []
        self.target_tickers = {}
        self.fundamental_data = {}  # PER, PBR, ROE 등
        self.market_index = None
        self.data = pd.DataFrame()
        
        print("="*70)
        print("🇺🇸 미국 멀티팩터 전략 백테스터")
        print("="*70)
        print(f"  📅 기간: {start_date} ~ {end_date}")
        print(f"  💰 초기자본: ${initial_capital:,}")
        print(f"  🔧 팩터: M{momentum_weight*100:.0f}% V{value_weight*100:.0f}% Q{quality_weight*100:.0f}%")
        print(f"  📊 종목수: {num_stocks}개")
        print("="*70)


    def fetch_top_stocks(self):
        print("\n📊 STEP 1: S&P 500 종목 리스트 확보 중...")
        
        try:
            # S&P 500 리스트
            df_sp500 = fdr.StockListing('S&P500')
            top_stocks = df_sp500.head(200)
            
            for _, row in top_stocks.iterrows():
                ticker = row['Symbol']
                self.target_tickers[ticker] = ticker
                
                # 재무 데이터 (FinanceDataReader는 미국 주식 재무 데이터 제한적)
                # 실전에서는 yfinance 등 사용 권장
                self.fundamental_data[ticker] = {
                    'sector': row.get('Sector', 'Unknown'),
                    'marketcap': row.get('Market Cap', 0)
                }
            
            # 방어 자산
            self.target_tickers['BIL'] = 'BIL'
            self.fundamental_data['BIL'] = {'sector': 'Cash', 'marketcap': 0}
            
            print(f"   ✅ {len(self.target_tickers)}개 종목 확보")
            return True
            
        except Exception as e:
            print(f"   ❌ 종목 리스트 수집 실패: {e}")
            return False


    def download_data(self):
        print("\n📈 STEP 2: 가격 데이터 다운로드 중...")
        print("   (미국 서버라 느릴 수 있습니다...)")
        
        try:
            target_date = datetime.strptime(self.start_date, "%Y-%m-%d")
            fetch_start_date = target_date - timedelta(days=400)
            fetch_start_str = fetch_start_date.strftime("%Y-%m-%d")
            
            # SPY 지수
            spy_df = fdr.DataReader('SPY', start=fetch_start_str, end=self.end_date)
            self.market_index = spy_df['Close'].ffill()
            
            # 개별 종목
            df_list = []
            total_count = len(self.target_tickers)
            
            for i, (name, code) in enumerate(self.target_tickers.items()):
                if i % 20 == 0:
                    print(f"   진행: {i}/{total_count} ({i/total_count*100:.0f}%)")
                
                try:
                    df = fdr.DataReader(code, start=fetch_start_str, end=self.end_date)
                    if df.empty or len(df) < 150:
                        continue
                    
                    series = df['Close'].rename(code)
                    df_list.append(series)
                except:
                    continue
                
                time.sleep(0.1)
            
            if df_list:
                self.data = pd.concat(df_list, axis=1).fillna(method='ffill', limit=5)
                missing_ratio = self.data.isnull().sum() / len(self.data)
                valid_cols = missing_ratio[missing_ratio < 0.1].index
                self.data = self.data[valid_cols]
                
                print(f"   ✅ {len(self.data.columns)}개 종목 데이터 준비 완료")
                return True
            else:
                print("   ❌ 유효한 데이터 없음")
                return False
                
        except Exception as e:
            print(f"   ❌ 가격 데이터 수집 실패: {e}")
            return False


    def calculate_multifactor_score(self, date, current_prices):
        """멀티팩터 점수 계산 (미국 버전)"""
        
        # 1. 모멘텀 팩터 (1M/3M/6M 가중 평균)
        try:
            mom_1m = self.data.pct_change(20).loc[date]
            mom_3m = self.data.pct_change(60).loc[date]
            mom_6m = self.data.pct_change(120).loc[date]
            
            # 가중 평균 (최근일수록 가중치 높임)
            momentum_score = (
                mom_1m.fillna(0) * 0.2 +
                mom_3m.fillna(0) * 0.3 +
                mom_6m.fillna(0) * 0.5
            )
        except:
            momentum_score = pd.Series(index=current_prices.index, data=0)
        
        # 2. 밸류 팩터 (간이 버전 - 시가총액 역수)
        # 주의: 실전에서는 yfinance로 PER, PBR 가져와야 함
        value_scores = {}
        for ticker in current_prices.index:
            if ticker == 'BIL':
                value_scores[ticker] = 0
                continue
            
            fund = self.fundamental_data.get(ticker, {})
            mcap = fund.get('marketcap', 0)
            
            # 시가총액이 작을수록 저평가 가능성 (간이 지표)
            if mcap > 0:
                # 로그 스케일로 정규화
                value_scores[ticker] = 1 / np.log10(mcap + 1)
            else:
                value_scores[ticker] = 0
        
        value_score = pd.Series(value_scores)
        
        # 3. 퀄리티 팩터 (변동성 + 추세 안정성)
        quality_scores = {}
        for ticker in current_prices.index:
            if ticker == 'BIL':
                quality_scores[ticker] = 0
                continue
            
            try:
                # 최근 120일 수익률의 안정성
                recent_returns = self.data[ticker].pct_change().tail(120)
                volatility = recent_returns.std()
                
                # 변동성이 낮고 상승 추세가 일관되면 고퀄리티
                positive_ratio = (recent_returns > 0).sum() / len(recent_returns)
                
                # 점수 = 일관성 - 변동성
                quality_scores[ticker] = positive_ratio / (volatility + 1e-6)
            except:
                quality_scores[ticker] = 0
        
        quality_score = pd.Series(quality_scores)
        
        # 4. 정규화
        def normalize(series):
            if series.std() == 0:
                return series
            return (series - series.min()) / (series.max() - series.min())
        
        mom_norm = normalize(momentum_score.reindex(current_prices.index).fillna(0))
        val_norm = normalize(value_score.reindex(current_prices.index).fillna(0))
        qual_norm = normalize(quality_score.reindex(current_prices.index).fillna(0))
        
        # 5. 종합 점수
        total_score = (
            mom_norm * self.momentum_weight +
            val_norm * self.value_weight +
            qual_norm * self.quality_weight
        )
        
        return total_score, mom_norm, val_norm, qual_norm


    def run(self):
        print("\n🚀 STEP 3: 백테스트 실행 중...")
        
        if self.data.empty:
            return
        
        # SPY 이평선
        spy_ma120 = self.market_index.rolling(window=120).mean()
        
        # 시뮬레이션 날짜
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
            if i == 0:
                continue
            
            current_prices = self.data.loc[date].dropna()
            
            # 월간 리밸런싱
            is_trading_day = False
            if date.month != prev_month:
                is_trading_day = True
                prev_month = date.month
            
            if is_trading_day:
                date_str = date.strftime('%Y-%m-%d')
                
                # 매도
                if holdings:
                    for name, qty in list(holdings.items()):
                        if name in current_prices:
                            price = current_prices[name]
                            sell_val = qty * price
                            fee = sell_val * self.commission
                            self.capital += (sell_val - fee)
                            
                            self.trade_log.append({
                                '날짜': date_str, '구분': '매도', '종목': name,
                                '가격': float(price), '수량': qty
                            })
                    holdings = {}
                
                # 시장 판단
                try:
                    spy_val = self.market_index.asof(date)
                    spy_ma = spy_ma120.asof(date)
                    is_bull = spy_val > spy_ma if not (pd.isna(spy_val) or pd.isna(spy_ma)) else False
                except:
                    is_bull = False
                
                # 종목 선정
                if not is_bull:
                    targets = [('BIL', 1.0)]
                else:
                    total_score, mom, val, qual = self.calculate_multifactor_score(date, current_prices)
                    
                    scores = total_score.drop('BIL', errors='ignore')
                    sorted_scores = scores.dropna().sort_values(ascending=False)
                    
                    if sorted_scores.empty or sorted_scores.iloc[0] <= 0:
                        targets = [('BIL', 1.0)]
                    else:
                        selected = []
                        for name, score in sorted_scores.items():
                            if score > 0:
                                selected.append(name)
                            if len(selected) >= self.num_stocks:
                                break
                        
                        if selected:
                            weight = 1.0 / len(selected)
                            targets = [(name, weight) for name in selected]
                        else:
                            targets = [('BIL', 1.0)]
                
                # 매수
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
                                
                                self.trade_log.append({
                                    '날짜': date_str, '구분': '매수', '종목': target,
                                    '가격': float(price), '수량': qty
                                })
                
                # 진행 상황
                if i % 60 == 0:
                    stock_val = sum(qty * current_prices.get(name, 0) 
                                  for name, qty in holdings.items())
                    total_val = self.capital + stock_val
                    progress = i / len(sim_dates) * 100
                    print(f"   [{date_str}] {progress:.1f}% | ${int(total_val):,}")
            
            # 평가
            stock_val = sum(qty * current_prices.get(name, 0) 
                          for name, qty in holdings.items())
            total_val = self.capital + stock_val
            self.history.append({'Date': date, 'TotalValue': total_val})
        
        if self.history:
            self.result_df = pd.DataFrame(self.history).set_index('Date')
            print("   ✅ 백테스트 완료!")


    def analyze(self):
        if self.result_df is None or self.result_df.empty:
            return
        
        print("\n📊 STEP 4: 성과 분석")
        print("="*70)
        
        final_val = self.result_df['TotalValue'].iloc[-1]
        total_return = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        days = (self.result_df.index[-1] - self.result_df.index[0]).days
        years = days / 365.25
        cagr = ((final_val / self.initial_capital) ** (1/years) - 1) * 100
        
        historical_max = self.result_df['TotalValue'].cummax()
        daily_drawdown = self.result_df['TotalValue'] / historical_max - 1.0
        mdd = daily_drawdown.min() * 100
        
        daily_returns = self.result_df['TotalValue'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        monthly_returns = self.result_df['TotalValue'].resample('M').last().pct_change().dropna()
        win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100
        
        # SPY 비교
        spy_period = self.market_index.loc[self.result_df.index]
        spy_return = ((spy_period.iloc[-1] - spy_period.iloc[0]) / spy_period.iloc[0]) * 100
        
        print(f"  📅 기간: {days}일 ({years:.2f}년)")
        print(f"\n  💰 초기: ${self.initial_capital:,}")
        print(f"  💰 최종: ${int(final_val):,}")
        print(f"  📈 총 수익률: {total_return:.2f}%")
        print(f"  📈 CAGR: {cagr:.2f}%")
        print(f"  📉 MDD: {mdd:.2f}%")
        print(f"  ⚖️  Sharpe: {sharpe:.3f}")
        print(f"  🎯 승률: {win_rate:.1f}%")
        print(f"\n  📊 S&P 500 (SPY): {spy_return:.2f}%")
        print(f"  🔥 초과 수익: +{total_return - spy_return:.2f}%p")
        print("="*70)


    def plot(self):
        if self.result_df is None or self.result_df.empty:
            return
        
        print("\n📈 STEP 5: 시각화 생성 중...")
        
        final_val = self.result_df['TotalValue'].iloc[-1]
        total_return = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        spy_period = self.market_index.loc[self.result_df.index]
        spy_norm = spy_period / spy_period.iloc[0] * self.initial_capital
        
        plt.figure(figsize=(14, 6))
        plt.plot(self.result_df.index, self.result_df['TotalValue'],
                label='US Multi-Factor', color='#6C5CE7', linewidth=2.5)
        plt.plot(spy_norm.index, spy_norm,
                label='S&P 500 (SPY)', color='gray', linestyle='--', linewidth=2)
        
        plt.title(f'🇺🇸 US Multi-Factor Strategy | Return: {total_return:.2f}%', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)', fontsize=11)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('us_multifactor_result.png', dpi=150)
        print("   ✅ 그래프 저장: us_multifactor_result.png")
        plt.show()


    def save_excel(self, filename="US_MultiFactor_Result.xlsx"):
        if self.result_df is None or self.result_df.empty:
            return
        
        print(f"\n💾 STEP 6: 엑셀 저장 중... ({filename})")
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 매매일지
                if self.trade_log:
                    pd.DataFrame(self.trade_log).to_excel(
                        writer, sheet_name='매매일지', index=False)
                
                # 일별 자산
                self.result_df.to_excel(writer, sheet_name='일별자산')
            
            print("   ✅ 저장 완료!")
        except Exception as e:
            print(f"   ❌ 저장 실패: {e}")


if __name__ == "__main__":
    # 미국 멀티팩터 백테스트
    bt = USMultiFactorBacktest(
        start_date='2021-01-01',
        end_date='2026-01-17',
        initial_capital=10000,
        momentum_weight=0.5,    # 미국 최적: 모멘텀 50%
        value_weight=0.2,      # 밸류 25%
        quality_weight=0.3,    # 퀄리티 25%
        num_stocks=5

    )
    
    if bt.fetch_top_stocks():
        if bt.download_data():
            bt.run()
            bt.analyze()
            bt.plot()
            bt.save_excel()
            
            print("\n✅ 모든 작업 완료!")
