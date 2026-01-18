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


class MultiFactorBacktest:
    """멀티팩터 전략 백테스트"""
    
    def __init__(self, start_date, end_date, initial_capital=10000000, 
                 momentum_weight=0.4, value_weight=0.3, 
                 quality_weight=0.3, volatility_weight=0.0,
                 num_stocks=5, slippage=0.003):
        
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.initial_capital = initial_capital
        
        # 전략 파라미터
        self.momentum_weight = momentum_weight
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.volatility_weight = volatility_weight
        self.num_stocks = num_stocks
        
        # 거래 비용
        self.commission = 0.00015
        self.slippage = slippage
        
        # 데이터
        self.target_tickers = {}
        self.financial_data = {}
        self.kospi_index = None
        self.data = pd.DataFrame()
        
        # 결과
        self.history = []
        self.trade_log = []
        self.result_df = None
        
        print("="*70)
        print("🎯 멀티팩터 전략 백테스터 초기화")
        print("="*70)
        print(f"  📅 기간: {start_date} ~ {end_date}")
        print(f"  💰 초기자본: {initial_capital:,}원")
        print(f"  🔧 팩터: M{momentum_weight*100:.0f}% V{value_weight*100:.0f}% Q{quality_weight*100:.0f}%")
        print(f"  📊 종목수: {num_stocks}개")
        print("="*70)


    def fetch_data(self):
        """데이터 수집"""
        print("\n📊 STEP 1: 데이터 수집 중...")
        
        # 1. 종목 리스트
        try:
            df_kospi = fdr.StockListing('KOSPI')
            top_kospi = df_kospi.sort_values('Marcap', ascending=False).head(100)
            
            df_kosdaq = fdr.StockListing('KOSDAQ')
            top_kosdaq = df_kosdaq.sort_values('Marcap', ascending=False).head(100)
            
            for _, row in pd.concat([top_kospi, top_kosdaq]).iterrows():
                name = row['Name']
                self.target_tickers[name] = row['Code']
                self.financial_data[name] = {
                    'PER': row.get('PER', np.nan),
                    'PBR': row.get('PBR', np.nan),
                    'ROE': row.get('ROE', np.nan),
                    'DIV': row.get('DivYield', 0),
                    'Marcap': row.get('Marcap', 0)
                }
            
            self.target_tickers['KODEX 미국달러선물'] = '261240'
            self.financial_data['KODEX 미국달러선물'] = {
                'PER': np.nan, 'PBR': np.nan, 'ROE': 0, 'DIV': 0, 'Marcap': 0
            }
            
            print(f"   ✅ {len(self.target_tickers)}개 종목 확보")
            
        except Exception as e:
            print(f"   ❌ 종목 리스트 수집 실패: {e}")
            return False
        
        # 2. 가격 데이터
        try:
            target_date = datetime.strptime(self.start_date, "%Y-%m-%d")
            fetch_start_date = target_date - timedelta(days=400)
            fetch_start_str = fetch_start_date.strftime("%Y-%m-%d")
            
            # KOSPI 지수
            kospi_df = fdr.DataReader('KS11', start=fetch_start_str, end=self.end_date)
            self.kospi_index = kospi_df['Close'].ffill()
            
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
                    
                    series = df['Close'].rename(name)
                    df_list.append(series)
                except:
                    continue
                
                time.sleep(0.05)
            
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


    def calculate_scores(self, date, prev_date, current_prices):
        """멀티팩터 점수 계산"""
        
        # 1. 모멘텀
        try:
            ret_6m = self.data.pct_change(120).loc[prev_date]
            daily_rets = self.data.pct_change().loc[:prev_date].tail(120)
            vol_6m = daily_rets.std()
            
            epsilon = 1e-6
            momentum_score = ret_6m / (vol_6m + epsilon)
        except:
            momentum_score = pd.Series(index=current_prices.index, data=0)
        
        # 2. 밸류
        value_scores = {}
        for name in current_prices.index:
            if name == 'KODEX 미국달러선물':
                value_scores[name] = 0
                continue
            
            fin = self.financial_data.get(name, {})
            per = fin.get('PER', np.nan)
            pbr = fin.get('PBR', np.nan)
            
            score = 0
            if pd.notna(per) and 0 < per < 30:
                score += 1 / per
            if pd.notna(pbr) and 0 < pbr < 3:
                score += 1 / pbr
            
            value_scores[name] = score
        
        value_score = pd.Series(value_scores)
        
        # 3. 퀄리티
        quality_scores = {}
        for name in current_prices.index:
            if name == 'KODEX 미국달러선물':
                quality_scores[name] = 0
                continue
            
            fin = self.financial_data.get(name, {})
            roe = fin.get('ROE', 0)
            per = fin.get('PER', np.nan)
            
            score = 0
            if roe > 15:
                score += 2
            elif roe > 10:
                score += 1
            
            if pd.notna(per) and 5 < per < 20:
                score += 1
            
            quality_scores[name] = score
        
        quality_score = pd.Series(quality_scores)
        
        # 4. 저변동성
        if self.volatility_weight > 0:
            epsilon = 1e-6
            vol_score = 1 / (vol_6m + epsilon)
        else:
            vol_score = pd.Series(index=current_prices.index, data=0)
        
        # 5. 정규화
        def normalize(series):
            if series.std() == 0:
                return series
            return (series - series.min()) / (series.max() - series.min())
        
        mom_norm = normalize(momentum_score.reindex(current_prices.index).fillna(0))
        val_norm = normalize(value_score.reindex(current_prices.index).fillna(0))
        qual_norm = normalize(quality_score.reindex(current_prices.index).fillna(0))
        vol_norm = normalize(vol_score.reindex(current_prices.index).fillna(0))
        
        # 6. 종합 점수
        total_score = (
            mom_norm * self.momentum_weight +
            val_norm * self.value_weight +
            qual_norm * self.quality_weight +
            vol_norm * self.volatility_weight
        )
        
        return total_score


    def run(self):
        """백테스트 실행"""
        print("\n🚀 STEP 2: 백테스트 실행 중...")
        
        if self.data.empty:
            print("   ❌ 데이터가 없습니다.")
            return
        
        # KOSPI 이평선
        kospi_ma60 = self.kospi_index.rolling(window=60).mean()
        
        # 시뮬레이션 날짜
        dates = self.data.index
        start_idx = 0
        target_start = datetime.strptime(self.start_date, "%Y-%m-%d")
        for i, d in enumerate(dates):
            if d >= target_start:
                start_idx = i
                break
        sim_dates = dates[start_idx:]
        
        # 초기화
        holdings = {}
        prev_month = -1
        
        for i, date in enumerate(sim_dates):
            if i == 0:
                continue
            
            prev_date = sim_dates[i-1]
            current_prices = self.data.loc[date].dropna()
            
            # 월간 리밸런싱 (1~7일)
            is_trading_day = False
            if date.month != prev_month and 1 <= date.day <= 7:
                is_trading_day = True
                prev_month = date.month
            
            if is_trading_day:
                date_str = date.strftime('%Y-%m-%d')
                
                # 매도
                if holdings:
                    for name, qty in list(holdings.items()):
                        if name in current_prices:
                            base_price = current_prices[name]
                            actual_price = base_price * (1 - self.slippage)
                            sell_val = qty * actual_price
                            fee = sell_val * self.commission
                            self.capital += (sell_val - fee)
                            
                            self.trade_log.append({
                                '날짜': date_str, '구분': '매도', '종목': name,
                                '가격': int(actual_price), '수량': qty
                            })
                    holdings = {}
                
                # 시장 판단
                try:
                    k_val = self.kospi_index.asof(prev_date)
                    k_ma = kospi_ma60.asof(prev_date)
                    is_bull = k_val > k_ma if not (pd.isna(k_val) or pd.isna(k_ma)) else False
                except:
                    is_bull = False
                
                # 종목 선정
                if not is_bull:
                    targets = [('KODEX 미국달러선물', 1.0)]
                else:
                    total_score = self.calculate_scores(date, prev_date, current_prices)
                    scores = total_score.drop('KODEX 미국달러선물', errors='ignore')
                    sorted_scores = scores.dropna().sort_values(ascending=False)
                    
                    if sorted_scores.empty or sorted_scores.iloc[0] <= 0:
                        targets = [('KODEX 미국달러선물', 1.0)]
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
                            targets = [('KODEX 미국달러선물', 1.0)]
                
                # 매수
                current_cash = self.capital
                for target, weight in targets:
                    if target in current_prices:
                        base_price = current_prices[target]
                        actual_price = base_price * (1 + self.slippage)
                        
                        budget = current_cash * weight
                        if actual_price > 0:
                            qty = int(budget // actual_price)
                            if qty > 0:
                                buy_val = qty * actual_price
                                fee = buy_val * self.commission
                                self.capital -= (buy_val + fee)
                                holdings[target] = qty
                                
                                self.trade_log.append({
                                    '날짜': date_str, '구분': '매수', '종목': target,
                                    '가격': int(actual_price), '수량': qty
                                })
                
                # 진행 상황 출력
                if i % 60 == 0:
                    stock_val = sum(qty * current_prices.get(name, 0) 
                                  for name, qty in holdings.items())
                    total_val = self.capital + stock_val
                    progress = i / len(sim_dates) * 100
                    print(f"   [{date_str}] {progress:.1f}% | 자산: {int(total_val):,}원")
            
            # 일별 자산 평가
            stock_val = sum(qty * current_prices.get(name, 0) 
                          for name, qty in holdings.items())
            total_val = self.capital + stock_val
            self.history.append({'Date': date, 'TotalValue': total_val})
        
        # 결과 저장
        if self.history:
            self.result_df = pd.DataFrame(self.history).set_index('Date')
            print(f"   ✅ 백테스트 완료!")
        else:
            print(f"   ❌ 백테스트 실패")


    def analyze(self):
        """성과 분석"""
        if self.result_df is None or self.result_df.empty:
            print("   ❌ 결과가 없습니다.")
            return
        
        print("\n📊 STEP 3: 성과 분석")
        print("="*70)
        
        # 수익률
        final_val = self.result_df['TotalValue'].iloc[-1]
        total_return = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        # CAGR
        days = (self.result_df.index[-1] - self.result_df.index[0]).days
        years = days / 365.25
        cagr = ((final_val / self.initial_capital) ** (1/years) - 1) * 100
        
        # MDD
        historical_max = self.result_df['TotalValue'].cummax()
        daily_drawdown = self.result_df['TotalValue'] / historical_max - 1.0
        mdd = daily_drawdown.min() * 100
        
        # Sharpe
        daily_returns = self.result_df['TotalValue'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # 승률
        monthly_returns = self.result_df['TotalValue'].resample('M').last().pct_change().dropna()
        win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100
        
        # KOSPI 비교
        kospi_period = self.kospi_index.loc[self.result_df.index]
        kospi_return = ((kospi_period.iloc[-1] - kospi_period.iloc[0]) / kospi_period.iloc[0]) * 100
        
        print(f"  📅 기간: {self.result_df.index[0].strftime('%Y-%m-%d')} ~ {self.result_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  📆 일수: {days}일 ({years:.2f}년)")
        print(f"\n  💰 초기 자본: {self.initial_capital:,}원")
        print(f"  💰 최종 자산: {int(final_val):,}원")
        print(f"  📈 총 수익률: {total_return:.2f}%")
        print(f"  📈 연평균 수익률 (CAGR): {cagr:.2f}%")
        print(f"  📉 최대 낙폭 (MDD): {mdd:.2f}%")
        print(f"  ⚖️  샤프 지수: {sharpe:.3f}")
        print(f"  🎯 월간 승률: {win_rate:.1f}%")
        print(f"\n  📊 KOSPI 수익률: {kospi_return:.2f}%")
        print(f"  🔥 KOSPI 대비 초과 수익: {total_return - kospi_return:.2f}%p")
        print("="*70)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'kospi_return': kospi_return
        }


    def plot(self):
        """결과 시각화"""
        if self.result_df is None or self.result_df.empty:
            return
        
        print("\n📈 STEP 4: 시각화 생성 중...")
        
        # 메트릭 계산
        final_val = self.result_df['TotalValue'].iloc[-1]
        total_return = ((final_val - self.initial_capital) / self.initial_capital) * 100
        
        historical_max = self.result_df['TotalValue'].cummax()
        daily_drawdown = self.result_df['TotalValue'] / historical_max - 1.0
        mdd = daily_drawdown.min() * 100
        
        daily_returns = self.result_df['TotalValue'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # KOSPI 비교
        kospi_period = self.kospi_index.loc[self.result_df.index]
        kospi_norm = kospi_period / kospi_period.iloc[0] * self.initial_capital
        
        # 그래프
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 누적 수익
        plt.subplot(2, 2, 1)
        plt.plot(self.result_df.index, self.result_df['TotalValue'],
                label='멀티팩터 전략', color='#6C5CE7', linewidth=2.5)
        plt.plot(kospi_norm.index, kospi_norm,
                label='KOSPI', color='gray', linestyle='--', linewidth=2, alpha=0.7)
        plt.title(f'누적 자산 추이 | 수익률 {total_return:.2f}%', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('자산 (원)', fontsize=11)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 2. Drawdown
        plt.subplot(2, 2, 2)
        plt.fill_between(self.result_df.index, daily_drawdown * 100, 0,
                        color='#FF6B6B', alpha=0.5)
        plt.plot(self.result_df.index, daily_drawdown * 100,
                color='#FF6B6B', linewidth=1.5)
        plt.title(f'Drawdown | MDD {mdd:.2f}%', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=11)
        plt.axhline(y=mdd, color='red', linestyle='--', linewidth=2,
                   label=f'최대 {mdd:.2f}%')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. 월별 수익률
        plt.subplot(2, 2, 3)
        monthly_returns = self.result_df['TotalValue'].resample('M').last().pct_change() * 100
        colors = ['#95E1D3' if x > 0 else '#F38181' for x in monthly_returns]
        plt.bar(monthly_returns.index, monthly_returns, color=colors, alpha=0.7)
        plt.title('월별 수익률', fontsize=14, fontweight='bold')
        plt.ylabel('수익률 (%)', fontsize=11)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 매매 통계
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        kospi_final = kospi_norm.iloc[-1]
        kospi_return = ((kospi_final - self.initial_capital) / self.initial_capital) * 100
        
        days = (self.result_df.index[-1] - self.result_df.index[0]).days
        years = days / 365.25
        cagr = ((final_val / self.initial_capital) ** (1/years) - 1) * 100
        
        monthly_returns_data = self.result_df['TotalValue'].resample('M').last().pct_change().dropna()
        win_rate = (monthly_returns_data > 0).sum() / len(monthly_returns_data) * 100
        
        num_trades = len(self.trade_log)
        num_buy = len([t for t in self.trade_log if t['구분'] == '매수'])
        
        summary_text = f"""
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📊 멀티팩터 전략 성과 요약
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        📈 수익률
          • 총 수익률: {total_return:.2f}%
          • CAGR: {cagr:.2f}%
          • KOSPI: {kospi_return:.2f}%
          • 초과수익: +{total_return - kospi_return:.2f}%p
        
        📉 리스크
          • MDD: {mdd:.2f}%
          • Sharpe: {sharpe:.3f}
          • 월간 승률: {win_rate:.1f}%
        
        🔧 전략 설정
          • 모멘텀: {self.momentum_weight*100:.0f}%
          • 밸류: {self.value_weight*100:.0f}%
          • 퀄리티: {self.quality_weight*100:.0f}%
          • 종목수: {self.num_stocks}개
        
        💼 매매 통계
          • 총 거래: {num_trades}건
          • 매수: {num_buy}건
          • 리밸런싱: 월 1회
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('multifactor_backtest_result.png', dpi=150, bbox_inches='tight')
        print("   ✅ 그래프 저장: multifactor_backtest_result.png")
        plt.show()


    def save_excel(self, filename="MultiFactor_Backtest.xlsx"):
        """결과를 엑셀로 저장"""
        if self.result_df is None or self.result_df.empty:
            return
        
        print(f"\n💾 STEP 5: 엑셀 저장 중... ({filename})")
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 성과 요약
                metrics = self.analyze()
                summary_df = pd.DataFrame([{
                    '전략명': '멀티팩터',
                    '초기자본': self.initial_capital,
                    '최종자산': int(self.result_df['TotalValue'].iloc[-1]),
                    '총수익률(%)': metrics['total_return'],
                    'CAGR(%)': metrics['cagr'],
                    'MDD(%)': metrics['mdd'],
                    'Sharpe': metrics['sharpe'],
                    '승률(%)': metrics['win_rate'],
                    'KOSPI수익률(%)': metrics['kospi_return']
                }])
                summary_df.to_excel(writer, sheet_name='성과요약', index=False)
                
                # 매매일지
                if self.trade_log:
                    pd.DataFrame(self.trade_log).to_excel(
                        writer, sheet_name='매매일지', index=False)
                
                # 일별 자산
                self.result_df.to_excel(writer, sheet_name='일별자산')
            
            print(f"   ✅ 엑셀 저장 완료!")
            
        except Exception as e:
            print(f"   ❌ 엑셀 저장 실패: {e}")


if __name__ == "__main__":
    # 백테스트 실행
    bt = MultiFactorBacktest(
        start_date='2023-01-01',
        end_date='2026-01-17',
        initial_capital=10000000,
        momentum_weight=0.4,
        value_weight=0.3,
        quality_weight=0.3,
        volatility_weight=0.0,
        num_stocks=5,
        slippage=0.003
    )
    
    # 1. 데이터 수집
    if bt.fetch_data():
        # 2. 백테스트 실행
        bt.run()
        
        # 3. 성과 분석
        bt.analyze()
        
        # 4. 시각화
        bt.plot()
        
        # 5. 엑셀 저장
        bt.save_excel()
        
        print("\n" + "="*70)
        print("✅ 모든 작업이 완료되었습니다!")
        print("="*70)
    else:
        print("\n❌ 데이터 수집 실패로 백테스트를 중단합니다.")
