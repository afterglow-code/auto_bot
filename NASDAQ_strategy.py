import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_supertrend(df, period=10, multiplier=2.0):
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    hl2 = (high + low) / 2
    upper_band, lower_band = hl2 + (multiplier * atr), hl2 - (multiplier * atr)
    f_upper, f_lower = upper_band.copy(), lower_band.copy()
    for i in range(1, len(df)):
        if upper_band.iloc[i] < f_upper.iloc[i-1] or close.iloc[i-1] > f_upper.iloc[i-1]: f_upper.iloc[i] = upper_band.iloc[i]
        else: f_upper.iloc[i] = f_upper.iloc[i-1]
        if lower_band.iloc[i] > f_lower.iloc[i-1] or close.iloc[i-1] < f_lower.iloc[i-1]: f_lower.iloc[i] = lower_band.iloc[i]
        else: f_lower.iloc[i] = f_lower.iloc[i-1]
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if trend[i-1] == 1: trend[i] = -1 if close.iloc[i] < f_lower.iloc[i] else 1
        else: trend[i] = 1 if close.iloc[i] > f_upper.iloc[i] else -1
    df['Trend'] = trend
    return df

def calculate_macd(df):
    df = df.copy()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    return df

def calculate_rsi(df, period=14):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    df['RSI'] = 100 * gain / (gain + loss).replace(0, np.nan)
    df['RSI'] = df['RSI'].fillna(0)
    return df

def generate_signals(df):
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Leverage_Level'] = 1
    for i in range(len(df)):
        # 초정밀 튜닝 결과: RSI > 46.0 적용
        if (df['Trend'].iloc[i] == 1 and 
            df['Close'].iloc[i] > df['SMA200'].iloc[i] and 
            df['MACD'].iloc[i] > df['Signal'].iloc[i] and
            df['RSI'].iloc[i] > 46.0): 
            df.at[df.index[i], 'Leverage_Level'] = 3
        else:
            df.at[df.index[i], 'Leverage_Level'] = 1
    return df

if __name__ == "__main__":
    print("최신 나스닥 데이터를 분석 중입니다...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    try:
        # 1. 판단 지표(QQQ) 분석
        qqq = fdr.DataReader('QQQ', start_date)
        qqq = calculate_supertrend(qqq)
        qqq = calculate_macd(qqq)
        qqq = calculate_rsi(qqq)
        qqq = generate_signals(qqq)
        latest = qqq.iloc[-1]
        
        # 2. 추천 종목 결정 및 가격 로드
        target_symbol = 'TQQQ' if latest['Leverage_Level'] == 3 else 'QQQM'
        target_data = fdr.DataReader(target_symbol, (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        target_price = target_data['Close'].iloc[-1]
        
        # 3. 최종 리포트 출력
        print("\n" + "★"*25)
        print(f" [ 나스닥 퀀트 마스터: 최종 시스템 ]")
        print(f" 분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("★"*25)
        print(f" 오늘의 추천 자산: {target_symbol}")
        print(f" 추천 종목 현재가: ${target_price:,.2f}")
        print("-" * 50)
        print(f" [ 전략 핵심 상태 ]")
        print(f" - 추세(Supertrend): {'상승' if latest['Trend']==1 else '하락'}")
        print(f" - 모멘텀(MACD)   : {'살아있음' if latest['MACD'] > latest['Signal'] else '죽어있음'}")
        print(f" - 시장강도(RSI)  : {latest['RSI']:.2f} (기준: 46.0)")
        print("-" * 50)
        print(f" 결론: {target_symbol}을(를) ${target_price:,.2f}에 매수/보유하세요.")
        print("="*50)
    except Exception as e:
        print(f"에러 발생: {e}")
