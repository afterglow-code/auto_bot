# technical_indicators.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import FinanceDataReader as fdr

def calculate_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ichimoku(df):
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    tenkan = (high9 + low9) / 2

    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    kijun = (high26 + low26) / 2

    span_a = ((tenkan + kijun) / 2).shift(26)

    high52 = df['High'].rolling(window=52).max()
    low52 = df['Low'].rolling(window=52).min()
    span_b = ((high52 + low52) / 2).shift(26)

    chikou = df['Close'].shift(-26)

    return tenkan, kijun, span_a, span_b, chikou

def resample_ohlc(df, rule):
    ohlc = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return ohlc.dropna()

class UniversalRiskRewardCalculator:
    def calculate_atr(self, df, period):
        tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def analyze(self, ticker, entry_price):
        df = fdr.DataReader(ticker, end=datetime.now().strftime('%Y-%m-%d'), start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        if df.empty: return None, None

        current_price = df['Close'].iloc[-1]
        if entry_price == 0: entry_price = current_price

        strategies = [
            {"name": "Scalping", "atr_period": 14, "risk_mult": 1.5, "reward_ratio": 1.5, "style": ":"},
            {"name": "Swing", "atr_period": 22, "risk_mult": 2.5, "reward_ratio": 2.0, "style": "--"},
            {"name": "Trend", "atr_period": 60, "risk_mult": 3.5, "reward_ratio": 3.0, "style": "-"}
        ]
        results = []
        targets = []
        stops = []
        for s in strategies:
            atr = self.calculate_atr(df, s['atr_period']).iloc[-1]
            risk = atr * s['risk_mult']
            stop, tp = entry_price - risk, entry_price + (risk * s['reward_ratio'])
            results.append({"Mode": s['name'], "Target": tp, "Stop": stop, "R/R": f"1:{s['reward_ratio']}", "Risk": f"-{(entry_price-stop)/entry_price*100:.1f}%"})
            targets.append(tp)
            stops.append(stop)

        rr_data = {
            'entry': entry_price,
            'targets': targets,
            'stops': stops
        }

        return pd.DataFrame(results), rr_data


class InstitutionalExecution:
    def __init__(self, account_balance, risk_per_trade_pct=2.0):
        """
        :param account_balance: 총 투자 원금
        :param risk_per_trade_pct: 1회 트레이딩 당 허용 손실률 (1~2%)
        """
        self.balance = account_balance
        self.risk_pct = risk_per_trade_pct
        self.risk_amount = self.balance * (self.risk_pct / 100.0)

    def calculate_position(self, entry_price, stop_loss_price):
        if entry_price <= 0 or stop_loss_price <= 0:
            return None

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return None

        qty = int(self.risk_amount / risk_per_share)
        total_invest = qty * entry_price

        if total_invest > self.balance:
            qty = int(self.balance / entry_price)
            total_invest = qty * entry_price

        return {
            "qty": qty,
            "total_amt": total_invest,
            "risk_amt": qty * risk_per_share,
            "portfolio_pct": (total_invest / self.balance) * 100 if self.balance > 0 else 0,
        }

    def get_pyramiding_plan(self, entry_price, total_qty, stop_price=None, target_price=None):
        plan = []
        if total_qty <= 0:
            return pd.DataFrame()

        q1 = int(total_qty * 0.3)
        plan.append({"단계": "1차 진입 (정찰병)", "가격": entry_price, "수량": q1, "조건": "현재가 진입"})

        p2 = entry_price * 1.02
        q2 = int(total_qty * 0.4)
        plan.append({"단계": "2차 진입 (추세확인)", "가격": p2, "수량": q2, "조건": "+2% 수익 발생 시"})

        p3 = p2 * 1.02
        q3 = max(0, total_qty - q1 - q2)
        plan.append({"단계": "3차 진입 (가속화)", "가격": p3, "수량": q3, "조건": "추가 +2% 상승 시"})

        if stop_price is not None:
            for row in plan:
                row["손절가"] = stop_price
        if target_price is not None:
            for row in plan:
                row["익절가"] = target_price

        return pd.DataFrame(plan)
