import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf  # 추가된 라이브러리
from datetime import datetime, timedelta
import requests
import os
import time

# =========================================================
# [사용자 설정 영역]
# =========================================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

MY_TOTAL_ASSETS = 10000000 

REBALANCE_PERIOD_START = 1
REBALANCE_PERIOD_END = 7

# 비상용 엑셀 파일명 (Name, Code 컬럼이 있어야 함)
BACKUP_EXCEL_FILE = 'target_tickers_backup.xlsx'
# =========================================================

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 텔레그램 설정이 없습니다. 메시지를 보내지 않습니다.")
        print(f"[메시지 미리보기]\n{msg}")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {'chat_id': CHAT_ID, 'text': msg}
    
    try: 
        requests.get(url, params=params)
        print("전송 완료")
    except Exception as e: 
        print(f"전송 실패: {e}")

def get_data_hybrid(code, start_date, end_date):
    """ 
    FDR 실패 시 YFinance로 전환하는 하이브리드 함수 
    (KOSPI .KS / KOSDAQ .KQ 자동 판별 시도)
    """
    # 1. FDR 시도
    try:
        df = fdr.DataReader(code, start=start_date, end=end_date)
        if not df.empty and len(df) > 10: # 데이터가 너무 적으면 실패로 간주
            return df['Close']
    except:
        pass

    # 2. YFinance 시도
    # 한국 주식은 .KS(코스피) 혹은 .KQ(코스닥) 접미사가 필요함
    suffixes = ['.KS', '.KQ']
    
    for suffix in suffixes:
        try:
            yf_code = f"{code}{suffix}"
            # progress=False로 지저분한 로그 제거
            df = yf.download(yf_code, start=start_date, end=end_date, progress=False)
            
            if not df.empty:
                # yfinance 최신 버전은 MultiIndex 컬럼일 수 있음 (Price, Ticker)
                if isinstance(df.columns, pd.MultiIndex):
                    # Close 컬럼의 해당 티커 데이터만 추출
                    if 'Close' in df.columns:
                        series = df['Close'][yf_code]
                    else:
                        continue # Close가 없으면 다음 시도
                else:
                    series = df['Close']
                
                # 데이터가 충분한지 확인
                if len(series.dropna()) > 10:
                    return series
        except:
            continue
            
    return None

def get_todays_signal():
    print("🚀 [TOP 200 변동성조절 전략 + Hybrid Data] 데이터 분석 시작...")
    
    # 1. 대상 종목 리스트 구성 (FDR 실패 시 엑셀 백업 사용)
    target_tickers = {}
    
    try:
        print("   [1단계] FDR 종목 리스트 확보 시도...")
        # KOSPI
        df_kospi = fdr.StockListing('KOSPI')
        top_kospi = df_kospi.sort_values('Marcap', ascending=False).head(100)
        for _, row in top_kospi.iterrows():
            target_tickers[row['Name']] = row['Code']

        # KOSDAQ
        df_kosdaq = fdr.StockListing('KOSDAQ')
        top_kosdaq = df_kosdaq.sort_values('Marcap', ascending=False).head(100)
        for _, row in top_kosdaq.iterrows():
            target_tickers[row['Name']] = row['Code']

        print(f"   -> FDR 성공: 총 {len(target_tickers)}개 종목 후보 확보")

    except Exception as e:
        print(f"⚠️ FDR 리스트 확보 실패 ({e}) -> 엑셀 백업 파일 로드 시도")
        
        try:
            # 엑셀 파일 읽기 (Code 컬럼을 문자열로 읽어야 앞의 0이 안 사라짐)
            df_backup = pd.read_excel(BACKUP_EXCEL_FILE, dtype={'Code': str})
            
            # Name, Code 컬럼이 있는지 확인
            if 'Name' in df_backup.columns and 'Code' in df_backup.columns:
                for _, row in df_backup.iterrows():
                    target_tickers[row['Name']] = row['Code']
                print(f"   -> 엑셀 로드 성공: 총 {len(target_tickers)}개 종목 후보 확보")
            else:
                send_telegram("❌ 엑셀 파일 형식이 올바르지 않습니다 (Name, Code 컬럼 필요)")
                return
        except Exception as ex_excel:
            send_telegram(f"❌ 엑셀 파일 로드 실패: {ex_excel}\n파일({BACKUP_EXCEL_FILE})이 있는지 확인하세요.")
            return

    # 달러 선물 추가 (필수)
    target_tickers['KODEX 미국달러선물'] = '261240'

    # 2. 데이터 다운로드 (Hybrid 방식 적용)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    kospi = None
    raw_data = pd.DataFrame()
    
    try:
        # 2-1. KOSPI 지수 (지수는 보통 FDR이 안정적이나 실패시 야후 사용)
        kospi_series = get_data_hybrid('^KS11', start_date, end_date) # 야후에선 ^KS11
        if kospi_series is None:
             # FDR 코드로 재시도 (함수 내부 로직상 순서가 반대지만 명시적 호출)
             try: kospi_series = fdr.DataReader('KS11', start=start_date, end=end_date)['Close']
             except: pass
        
        if kospi_series is not None:
            kospi = kospi_series.ffill()
        else:
            raise Exception("코스피 지수 데이터를 가져올 수 없습니다.")

        # 2-2. 개별 종목 데이터 수집 Loop
        df_list = []
        total_count = len(target_tickers)
        
        print(f"   [2단계] 개별 종목 데이터 수집 중 (Hybrid Mode)...")
        for i, (name, code) in enumerate(target_tickers.items()):
            if i % 20 == 0: print(f"   진행 중... ({i}/{total_count})")
            
            # 여기서 수정된 get_data_hybrid 함수 사용
            series = get_data_hybrid(code, start_date, end_date)
            
            if series is not None:
                # 데이터 길이 체크 (최소 120일)
                if len(series) >= 120:
                    series.name = name # 시리즈 이름을 종목명으로 설정
                    df_list.append(series)
            
            time.sleep(0.01) # 너무 빠른 요청 방지
        
        if df_list:
            raw_data = pd.concat(df_list, axis=1).ffill().dropna(how='all')
        else:
            raise Exception("유효한 데이터를 하나도 가져오지 못했습니다.")

    except Exception as e:
        send_telegram(f"❌ 데이터 다운로드 치명적 오류: {e}")
        return

    # 3. 전략 계산 (기존 로직 유지)
    try:
        daily_rets = raw_data.pct_change()
        
        ret_3m = raw_data.pct_change(60).iloc[-1]
        ret_6m = raw_data.pct_change(120).iloc[-1]
        
        vol_3m = daily_rets.rolling(60).std().iloc[-1]
        vol_6m = daily_rets.rolling(120).std().iloc[-1]
        
        epsilon = 1e-6 
        score_3m = ret_3m / (vol_3m + epsilon)
        score_6m = ret_6m / (vol_6m + epsilon)
        
        weighted_score = (score_3m.fillna(0) * 0.4) + (score_6m.fillna(0) * 0.6)

        kospi_ma120 = kospi.rolling(window=120).mean().iloc[-1]
        current_kospi = kospi.iloc[-1]
        
        if hasattr(current_kospi, 'item'): current_kospi = current_kospi.item()
        if hasattr(kospi_ma120, 'item'): kospi_ma120 = kospi_ma120.item()

        is_bull_market = current_kospi > kospi_ma120
    except Exception as e:
        send_telegram(f"❌ 지표 계산 중 오류: {e}")
        return

    # 4. 목표 종목 선정 (기존 로직 유지)
    final_targets = [] 
    reason = ""

    if is_bull_market:
        scores = weighted_score.drop('KODEX 미국달러선물', errors='ignore')
        top_assets = scores.sort_values(ascending=False)
        
        if top_assets.empty or top_assets.iloc[0] <= 0:
            final_targets = [('KODEX 미국달러선물', 1.0)]
            reason = "주도주 부재(전체 하락세) -> 달러 방어"
        else:
            selected = []
            for name, score in top_assets.items():
                if score > 0: selected.append(name)
                if len(selected) >= 3: break
            
            count = len(selected)
            if count > 0:
                weight = 1.0 / count
                for s in selected:
                    final_targets.append((s, weight))
                reason = f"TOP {count} 변동성조절 모멘텀"
            else:
                final_targets = [('KODEX 미국달러선물', 1.0)]
                reason = "대상 종목 없음 -> 달러 방어"
    else:
        final_targets = [('KODEX 미국달러선물', 1.0)]
        reason = "하락장 방어(코스피 이탈)"

    # 5. 메시지 전송 (기존 로직 유지)
    today_dt = datetime.now()
    next_rebalance_date = (today_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    is_rebalance_period = (REBALANCE_PERIOD_START <= today_dt.day <= REBALANCE_PERIOD_END)
    
    msg = f"📅 [{today_dt.strftime('%Y-%m-%d')}] 국내 주식 봇 (Hybrid)\n"
    msg += f"전략: 변동성조절 모멘텀 (TOP 3)\n"
    msg += f"시장: {'🔴상승장' if is_bull_market else '🔵하락장'}\n"
    msg += "-" * 20 + "\n"
    
    target_list_msg = ""
    for name, weight in final_targets:
        try:
            current_score = weighted_score[name]
        except:
            current_score = 0.0
        
        score_emoji = ""
        if current_score >= 2.0: score_emoji = "🔥🔥"
        elif current_score >= 1.0: score_emoji = "🔥"
        elif current_score > 0: score_emoji = "🙂"
        else: score_emoji = "🛡️"

        if name in raw_data.columns:
            current_price = raw_data[name].iloc[-1]
            buy_budget = MY_TOTAL_ASSETS * weight
            buy_qty = int(buy_budget // current_price)
            
            target_list_msg += f"👉 {name} (점수: {current_score:.2f} {score_emoji})\n"
            target_list_msg += f"   비중: {int(weight*100)}% ({buy_qty}주)\n"
        else:
             target_list_msg += f"👉 {name} (점수: {current_score:.2f})\n"

    if is_rebalance_period:
        msg += "🔔 [리밸런싱 주간입니다]\n"
        msg += f"사유: {reason}\n\n"
        msg += target_list_msg
    else:
        msg += f"☕ [관망 모드]\n이번 달 목표 (실시간 순위):\n"
        msg += target_list_msg
        msg += f"\n다음 리밸런싱: {next_rebalance_date.strftime('%Y-%m-%d')}\n"

    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    get_todays_signal()