# total_daily_report.py

import sys
import os
import datetime
import pytz
import config as cfg
from common import send_telegram

# 각 봇 모듈 임포트
# 파일 이름이 숫자로 시작해서 importlib 사용 혹은 별칭으로 import 해야 할 수도 있지만, 
# 파이썬에서는 숫자로 시작하는 모듈 import가 까다로움.
# 여기서는 importlib을 사용하여 동적으로 가져오겠습니다.
import importlib

def import_module_by_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 모듈 로드
etf_bot = import_module_by_path("etf_bot", "1m_auto_bot_upload_etf.py")
stock_bot = import_module_by_path("stock_bot", "1m_auto_bot_upload_stock.py")
us_bot = import_module_by_path("us_bot", "1m_auto_bot_upload_US.py")
mosig_bot = import_module_by_path("mosig_bot", "mosig_bot.py")

def main():
    print("🚀 [통합 봇] 일일 투자 분석 시작...")
    
    # 1. 각 전략 실행 (순차 실행)
    print(">>> 1. 한국 ETF 분석 중...")
    etf_result = etf_bot.analyze_etf_strategy()
    
    print(">>> 2. 한국 개별주 분석 중...")
    stock_result = stock_bot.analyze_stock_strategy()
    
    print(">>> 3. 미국 주식 분석 중...")
    us_result = us_bot.analyze_us_stock_strategy()
    
    print(">>> 4. 모멘텀 급등주 스캔 중...")
    mosig_candidates = mosig_bot.analyze_mosig_strategy()
    
    # 2. 통합 리포트 작성 (ETF + Stock + US)
    report_msg = create_consolidated_report(etf_result, stock_result, us_result, mosig_candidates)
    
    # 3. 통합 리포트 전송 (메인 채팅방)
    print("📡 [통합 봇] 메인 리포트 전송 중...")
    send_telegram(report_msg, parse_mode='HTML')
    
    # 4. Mosig 알림 전송 (별도 채팅방)
    print("📡 [통합 봇] 급등주 알림 전송 중...")
    mosig_msg = mosig_bot.format_message(mosig_candidates)
    send_telegram(mosig_msg, chat_id=cfg.CHAT_ID_1P, parse_mode='Markdown')
    
    print("✅ 모든 작업 완료!")

def create_consolidated_report(etf, stock, us, mosig_list):
    """3개 전략 결과를 하나의 메시지로 요약"""
    today_dt = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    
    # 이모지 매핑
    status_emoji = {
        '🔴 상승장': '🔴', '🟠 중립장': '🟠', '🔵 하락장': '🔵', 
        '정보 없음': '❓'
    }
    
    # --- 헤더 ---
    msg = f"<b>📊 통합 투자 리포트 [{today_dt.strftime('%m/%d %H:%M')}]</b>"
    msg += f"━━━━━━━━━━━━━━━━━━━━"
    
    # --- 1. 요약 섹션 ---
    etf_status = etf.get('market_status', '정보 없음')
    stock_status = stock.get('market_status', '정보 없음')
    us_status = us.get('market_status', '정보 없음')
    
    msg += f"<b>📝 시장 요약</b>"
    msg += f"🇰🇷 ETF : {status_emoji.get(etf_status, '')} {etf_status}"
    msg += f"🇰🇷 국장 : {status_emoji.get(stock_status, '')} {stock_status}"
    msg += f"🇺🇸 미장 : {status_emoji.get(us_status, '')} {us_status}"
    msg += f"🔎 포착 : {len(mosig_list)}개 종목"
    
    msg += f"━━━━━━━━━━━━━━━━━━━━"

    # --- 2. 한국 ETF 전략 ---
    msg += f"<b>1️⃣ 🇰🇷 한국 ETF</b>"
    if etf.get('error'):
        msg += f"⚠️ 오류: {etf['error']}"
    else:
        # 시장 지수
        idx_val = etf.get('market_index_val', 0)
        # 만약 시리즈라면 마지막 값 추출 (안전장치)
        if hasattr(idx_val, 'iloc'): idx_val = idx_val.iloc[-1]
        
        msg += f"• 코스피: {idx_val:,.0f}"
        
        targets = etf.get('final_targets', [])
        if not targets:
            msg += f"• 추천: 없음"
        else:
            msg += f"• <b>Top Pick:</b>"
            for name, weight in targets:
                # 점수 찾기
                score = etf['weighted_score'].get(name, 0.0) if 'weighted_score' in etf else 0
                msg += f"  - {name} ({int(weight*100)}%)"
    
    msg += ""

    # --- 3. 한국 개별주 전략 ---
    msg += f"<b>2️⃣ 🇰🇷 한국 개별주</b>"
    if stock.get('error'):
        msg += f"⚠️ 오류: {stock['error']}"
    else:
        targets = stock.get('final_targets', [])
        if not targets:
            msg += f"• 추천: 없음"
        else:
            # 방어 자산만 있는지 확인
            is_only_defense = len(targets) == 1 and targets[0][0] == cfg.STOCK_DEFENSE_ASSET
            
            if is_only_defense:
                msg += f"🛡️ <b>달러 방어 모드</b> (100%)"
            else:
                msg += f"• <b>Top Pick:</b>"
                for name, weight in targets:
                    if name == cfg.STOCK_DEFENSE_ASSET:
                        msg += f"  - 🛡️ {name} ({int(weight*100)}%)"
                    else:
                        msg += f"  - 🔥 {name} ({int(weight*100)}%)"

    msg += ""

    # --- 4. 미국 주식 전략 ---
    msg += f"<b>3️⃣ 🇺🇸 미국 주식</b>"
    if us.get('error'):
        msg += f"⚠️ 오류: {us['error']}"
    else:
        idx_val = us.get('market_index_val', 0)
        if hasattr(idx_val, 'iloc'): idx_val = idx_val.iloc[-1]

        msg += f"• S&P500: {idx_val:,.0f}"
        
        targets = us.get('final_targets', [])
        if not targets:
            msg += f"• 추천: 없음"
        else:
            is_only_defense = len(targets) == 1 and targets[0][0] == cfg.US_DEFENSE_ASSET
            if is_only_defense:
                 msg += f"🛡️ <b>현금/채권 방어</b> (BIL 100%)"
            else:
                msg += f"• <b>Top Pick:</b>"
                for name, weight in targets:
                    if name == cfg.US_DEFENSE_ASSET:
                        msg += f"  - 🛡️ {name} ({int(weight*100)}%)"
                    else:
                        msg += f"  - 🔥 {name} ({int(weight*100)}%)"

    msg += f"━━━━━━━━━━━━━━━━━━━━"
    msg += f"<i>💡 상세 내용은 각 터미널 로그 확인</i>"
    
    return msg

if __name__ == "__main__":
    main()
