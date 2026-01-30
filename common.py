# dev/common.py

import requests
import config
import pandas as pd
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def fetch_data_in_parallel(tickers, start_date, end_date):
    """
    여러 종목의 시세 데이터를 병렬로 수집합니다.
    :param tickers: {'종목명': '종목코드'} 형태의 딕셔너리
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :return: pd.DataFrame, 각 종목의 종가가 컬럼으로 구성됨
    """
    
    # 개별 종목 데이터를 가져오는 내부 함수
    def _fetch_one(name, code):
        try:
            df = fdr.DataReader(code, start=start_date, end=end_date)
            time.sleep(0.2)  # API 과부하 방지용 딜레이
            if df.empty:
                return None, f"{name}({code}) 데이터 없음"
            # 종가 시리즈 반환
            return df['Close'].rename(name), None
        except Exception as e:
            return None, f"{name}({code}) 수집 실패: {e}"

    df_list = []
    total_count = len(tickers)
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # 작업 제출
        future_to_ticker = {executor.submit(_fetch_one, name, code): name for name, code in tickers.items()}
        
        # 진행 상황 표시
        for i, future in enumerate(as_completed(future_to_ticker)):
            name = future_to_ticker[future]
            
            # 진행률 출력 (콘솔에 한 줄로 업데이트)
            print(f"\r   수집 진행률: {i+1}/{total_count} ({name})", end='', flush=True)

            series, error_msg = future.result()
            if error_msg:
                # print(f"\n⚠️  {error_msg}") # 실패 시 상세 로그 (필요시 활성화)
                continue
            
            if series is not None and not series.empty:
                df_list.append(series)

    print("\n✅ 병렬 데이터 수집 완료!")
    
    if not df_list:
        return pd.DataFrame()
        
    # 데이터프레임 병합
    raw_data = pd.concat(df_list, axis=1).ffill().dropna(how='all')
    return raw_data

def send_telegram(msg, chat_id=None, token=None, parse_mode='HTML'):
    """
    텔레그램 메시지를 전송합니다.
    chat_id, token이 지정되지 않으면 config의 기본값을 사용합니다.
    """
    # 파라미터로 받은 값이 없으면 config 파일의 기본값을 사용
    effective_token = token if token else config.TELEGRAM_TOKEN
    effective_chat_id = chat_id if chat_id else config.CHAT_ID

    if not effective_token or not effective_chat_id:
        print("⚠️ 텔레그램 TOKEN 또는 CHAT_ID가 설정되지 않았습니다.")
        # HTML 태그 제거 후 미리보기 출력
        import re
        clean_msg = re.sub('<.*?>', '', msg)
        print(f"--- 메시지 미리보기 ---\n{clean_msg}\n--------------------")
        return

    url = f"https://api.telegram.org/bot{effective_token}/sendMessage"
    params = {
        'chat_id': effective_chat_id,
        'text': msg,
        'parse_mode': parse_mode
    }
    
    try: 
        response = requests.get(url, params=params)
        response.raise_for_status() # 200번대 코드가 아닐 경우 예외 발생
        print(f"✅ 텔레그램 전송 완료 (Chat ID: {effective_chat_id})")
    except requests.exceptions.RequestException as e: 
        print(f"❌ 텔레그램 전송 실패: {e}")
        # 실패 시 응답 내용 출력
        if e.response:
            print(f"    - Status Code: {e.response.status_code}")
            print(f"    - Response: {e.response.text}")

if __name__ == '__main__':
    # 간단한 테스트 코드
    print("common.py 테스트: 기본 설정으로 메시지 전송")
    send_telegram("<b>테스트 메시지</b>\ncommon.py에서 보냄")
    
    print("\ncommon.py 테스트: 특정 ID/Token으로 메시지 전송")
    # 아래는 예시이며, 실제 토큰/ID를 넣어야 동작합니다.
    # send_telegram("개별 테스트", chat_id="123456", token="your_another_token")
