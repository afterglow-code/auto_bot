import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_supertrend(df, period=10, multiplier=3.0, change_atr=True):
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    hl2 = (high + low) / 2

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)

    atr_sma = tr.rolling(window=period).mean()
    atr_rma = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr = atr_rma if change_atr else atr_sma

    up_raw = hl2 - (multiplier * atr)
    dn_raw = hl2 + (multiplier * atr)

    up = up_raw.copy()
    dn = dn_raw.copy()

    for i in range(1, len(df)):
        up1 = up.iloc[i - 1] if pd.notna(up.iloc[i - 1]) else up_raw.iloc[i]
        dn1 = dn.iloc[i - 1] if pd.notna(dn.iloc[i - 1]) else dn_raw.iloc[i]

        up.iloc[i] = max(up_raw.iloc[i], up1) if close.iloc[i - 1] > up1 else up_raw.iloc[i]
        dn.iloc[i] = min(dn_raw.iloc[i], dn1) if close.iloc[i - 1] < dn1 else dn_raw.iloc[i]

    trend = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        prev_trend = trend[i - 1]
        up1 = up.iloc[i - 1] if pd.notna(up.iloc[i - 1]) else up.iloc[i]
        dn1 = dn.iloc[i - 1] if pd.notna(dn.iloc[i - 1]) else dn.iloc[i]

        if prev_trend == -1 and close.iloc[i] > dn1:
            trend[i] = 1
        elif prev_trend == 1 and close.iloc[i] < up1:
            trend[i] = -1
        else:
            trend[i] = prev_trend

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
    # 튜닝 결과 반영: SMA 200 -> 180으로 최적화
    df['SMA180'] = df['Close'].rolling(window=180).mean()
    df['Leverage_Level'] = 1
    for i in range(len(df)):
        # 1. 하락/방어 조건 (Trend 하락 혹은 180일선 하회)
        if df['Trend'].iloc[i] == -1 or df['Close'].iloc[i] < df['SMA180'].iloc[i]:
            df.at[df.index[i], 'Leverage_Level'] = 1
        
        # 2. 강력 상승 조건 (모든 모멘텀 일치)
        elif (df['MACD'].iloc[i] > df['Signal'].iloc[i] and 
              df['RSI'].iloc[i] > 46.0):
            df.at[df.index[i], 'Leverage_Level'] = 3
            
        # 3. 중립 조건 (추세는 살아있으나 모멘텀 부족)
        else:
            df.at[df.index[i], 'Leverage_Level'] = 2
            
    return df

if __name__ == "__main__":
    print("🚀 나스닥 3단계 레버리지 시스템 가동 중...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    try:
        # 1. 판단 지표(QQQ) 분석
        qqq = fdr.DataReader('QQQ', start_date)
        qqq = calculate_supertrend(qqq)
        qqq = calculate_macd(qqq)
        qqq = calculate_rsi(qqq)
        qqq = generate_signals(qqq)
        latest = qqq.iloc[-1]
        
        # 2. 추천 종목 결정 (3: TQQQ, 2: QLD, 1: QQQM)
        level_map = {3: 'TQQQ', 2: 'QLD', 1: 'QQQM'}
        target_symbol = level_map[latest['Leverage_Level']]
        
        target_data = fdr.DataReader(target_symbol, (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        target_price = target_data['Close'].iloc[-1]
        
        # 3. 최종 리포트 출력
        print("\n" + "★"*25)
        print(f" [ 나스닥 퀀트 마스터: 3단계 기어 변속 ]")
        print(f" 분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("★"*25)
        print(f" 오늘의 시장 강도 : {int(latest['Leverage_Level'])}단계")
        print(f" 추천 타겟 자산   : {target_symbol}")
        print(f" 현재가(종가기준) : ${target_price:,.2f}")
        print("-" * 50)
        
        mode_desc = {3: "🔥 강력 상승 (Full Power)", 2: "⚖️ 중립 유지 (Middle Gear)", 1: "🛡️ 방어 모드 (Safety First)"}
        print(f" [ 전략 핵심 상태 ]")
        print(f" - 현재 모드      : {mode_desc[latest['Leverage_Level']]}")
        print(f" - 추세(Supertrend): {'상승' if latest['Trend']==1 else '하락'}")
        print(f" - 장기추세(SMA180): {'상회' if latest['Close'] > latest['SMA180'] else '하회'}")
        print(f" - 모멘텀(MACD)   : {'살아있음' if latest['MACD'] > latest['Signal'] else '죽어있음'}")
        print(f" - 시장강도(RSI)  : {latest['RSI']:.2f} (기준: 46.0)")
        print("-" * 50)
        print(f" 결론: {target_symbol}을(를) 통해 시장 {int(latest['Leverage_Level'])}배수 대응을 유지하세요.")
        print("="*50)
    except Exception as e:
        print(f"에러 발생: {e}")
