import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import re

# 폰트 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

class UniversalRiskRewardCalculator:
    def __init__(self):
        pass

    def calculate_atr(self, df, period):
        """특정 기간 ATR 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def get_market_info(self, ticker):
        clean_ticker = re.sub(r'\.[A-Z]+$', '', ticker)
        
        if clean_ticker.isdigit(): 
            return {
                'country': 'KR',
                'currency_symbol': '',
                'currency_suffix': '원',
                'format': "{:,.0f}", 
            }
        else:
            return {
                'country': 'US',
                'currency_symbol': '$',
                'currency_suffix': '',
                'format': "${:,.2f}", 
            }

    def analyze(self, ticker, entry_price):
        market_info = self.get_market_info(ticker)
        # 포맷 함수 정의
        fmt_func = lambda x: market_info['format'].format(x).replace('$', market_info['currency_symbol']) + market_info['currency_suffix']
        
        print(f"\n🌍 [{ticker}] 통합 분석 시작 ({market_info['country']} Market)...")
        
        df = fdr.DataReader(ticker)
        if df.empty:
            print("❌ 데이터를 찾을 수 없습니다.")
            return

        df = df.tail(250) 
        current_price = df['Close'].iloc[-1]
        
        if entry_price == 0:
            entry_price = current_price
            print(f"👉 매수단가 미입력 -> 현재가({fmt_func(current_price)})로 계산")

        # ATR 계산
        atr_14 = self.calculate_atr(df, 14).iloc[-1]
        atr_22 = self.calculate_atr(df, 22).iloc[-1]
        atr_60 = self.calculate_atr(df, 60).iloc[-1]

        print("-" * 55)
        print(f"📊 현재 주가: {fmt_func(current_price)}")
        print(f"🌊 변동성(ATR) 현황:")
        print(f"   - 단기(14일): ±{fmt_func(atr_14)}")
        print(f"   - 스윙(22일): ±{fmt_func(atr_22)}")
        print(f"   - 추세(60일): ±{fmt_func(atr_60)}")
        print("-" * 55)

        # 전략 설정
        strategies = [
            {
                "name": "⚡ 단기 (Scalping)",
                "period": 14,
                "atr_val": atr_14,
                "risk_mult": 1.5,
                "reward_ratio": 1.5,
                "style": ":", # 점선
                "alpha": 0.6
            },
            {
                "name": "📈 스윙 (Swing)",
                "period": 22,
                "atr_val": atr_22,
                "risk_mult": 2.5,
                "reward_ratio": 2.0,
                "style": "--", # 파선
                "alpha": 0.8
            },
            {
                "name": "🚀 추세 (Trend)",
                "period": 60,
                "atr_val": atr_60,
                "risk_mult": 3.5,
                "reward_ratio": 3.0,
                "style": "-", # 실선
                "alpha": 1.0
            }
        ]

        print(f"🎯 진입 가격: {fmt_func(entry_price)}\n")
        
        for strategy in strategies:
            risk_width = strategy['atr_val'] * strategy['risk_mult']
            stop_loss = entry_price - risk_width
            
            reward_width = risk_width * strategy['reward_ratio']
            take_profit = entry_price + reward_width

            loss_amount = entry_price - stop_loss
            profit_amount = take_profit - entry_price     

            loss_pct = ((stop_loss - entry_price) / entry_price) * 100
            profit_pct = ((take_profit - entry_price) / entry_price) * 100

            print(f"[{strategy['name']}]")
            print(f"  🟦 익절(TP): {fmt_func(take_profit)} (+{profit_pct:.2f}%)")
            print(f"  🟧 손절(SL): {fmt_func(stop_loss)} ({loss_pct:.2f}%)")
            print(f"  ⚖️ 손익비: 1 : {strategy['reward_ratio']}")
            print(f"  💡 1주당 예상: -{fmt_func(int(loss_amount))} 잃거나, +{fmt_func(int(profit_amount))} 범")
            print("-" * 35)

        # 차트 시각화 (모든 전략 전달)
        self.plot_all_strategies(df, ticker, entry_price, strategies, market_info, fmt_func)

    def plot_all_strategies(self, df, ticker, entry_price, strategies, market_info, fmt_func):
        plt.figure(figsize=(14, 8)) # 그래프 크기 키움
        
        # 최근 6개월 데이터
        plot_data = df.tail(120)
        plt.plot(plot_data.index, plot_data['Close'], label='Close Price', color='black', alpha=0.6, linewidth=1.5)
        
        # 진입가 (파란 실선)
        plt.axhline(y=entry_price, color='blue', linestyle='-', linewidth=2, label=f'Entry: {fmt_func(entry_price)}')
        
        # 각 전략별 TP/SL 그리기
        for strat in strategies:
            risk_width = strat['atr_val'] * strat['risk_mult']
            stop_loss = entry_price - risk_width
            take_profit = entry_price + (risk_width * strat['reward_ratio'])
            
            # 라인 스타일 및 투명도 적용
            line_style = strat['style']
            alpha_val = strat['alpha']
            strat_name = strat['name'].split(' ')[1] # "단기", "스윙" 등만 추출
            
            # 익절 라인 (초록)
            plt.axhline(y=take_profit, color='green', linestyle=line_style, alpha=alpha_val, 
                        label=f'{strat_name} TP: {fmt_func(take_profit)}')
            
            # 손절 라인 (빨강)
            plt.axhline(y=stop_loss, color='red', linestyle=line_style, alpha=alpha_val, 
                        label=f'{strat_name} SL: {fmt_func(stop_loss)}')

        # 가장 넓은 범위(추세 전략)에 배경색 칠하기 (가독성 위해 하나만)
        trend_strat = strategies[2]
        trend_risk = trend_strat['atr_val'] * trend_strat['risk_mult']
        trend_sl = entry_price - trend_risk
        trend_tp = entry_price + (trend_risk * trend_strat['reward_ratio'])
        
        plt.axhspan(entry_price, trend_tp, color='green', alpha=0.05) # 아주 옅은 초록
        plt.axhspan(trend_sl, entry_price, color='red', alpha=0.05)   # 아주 옅은 빨강

        plt.title(f"[{ticker}] Multi-Strategy Risk/Reward Analysis", fontsize=15)
        plt.legend(loc='best', fontsize=9, framealpha=0.8) # 범례 표시
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    calc = UniversalRiskRewardCalculator()
    
    print("=== 🌏 만능 손익비 계산기 (KR/US) ===")
    print(" 예시) 삼성전자: 005930, 애플: AAPL")
    
    user_ticker = input("종목코드를 입력하세요: ").strip().upper()
    try:
        price_input = input("매수단가 (0 입력시 현재가): ").replace(',', '')
        user_price = float(price_input)
    except:
        user_price = 0
        
    calc.analyze(user_ticker, user_price)