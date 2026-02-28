# pip install pandas matplotlib numpy yfinance
import pandas as pd # 데이터 분석용
import numpy as np # 수치 계산용
import matplotlib.pyplot as plt # 시각화용
import yfinance as yf # 실시간 유가 API
import os

def get_realtime_wti():
    """Yahoo Finance에서 실시간 WTI 선물 가격 수집"""
    try:
        wti = yf.Ticker("CL=F") # WTI Crude Oil Future 심볼
        hist = wti.history(period="5d") # 최근 5일 데이터
        return hist['Close'].iloc[-1] # 가장 최신 종가 반환
    except Exception as e:
        print(f"API 호출 오류, 기본값 사용: {e}")
        return 75.0 # 오류 시 기본값(75달러) 반환

def run_expert_simulation(current_wti, scenario_level=2):
    """업로드 데이터 기반 산출된 상관계수 적용 시뮬레이션"""
    # [데이터 분석으로 도출된 고정 상수값]
    M_ABS = 8.21
    B_ABS = 1106.22
    
    # 리스크 시나리오 설정
    scenarios = {
        1: {'oil_jump': 1.05, 'logistics_premium': 50}, # 국지적: 5% 상승, 할증 50불
        2: {'oil_jump': 1.25, 'logistics_premium': 150}, # 봉쇄: 25% 상승, 할증 150불
        3: {'oil_jump': 1.50, 'logistics_premium': 400}  # 전면전: 50% 상승, 할증 400불
    }
    config = scenarios.get(scenario_level, scenarios[2])

    # 리스크 반영 예측 계산
    pred_wti = current_wti * config['oil_jump']
    pred_abs_base = (pred_wti * M_ABS) + B_ABS # 도출된 수식 적용 (y = mx + b)
    landed_cost = pred_abs_base + config['logistics_premium'] # 물류비 합산

    # 결과 데이터 생성
    result = {
        'Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'Current_WTI': current_wti,
        'Pred_WTI': pred_wti,
        'Pred_ABS': pred_abs_base,
        'Landed_Cost': landed_cost,
        'Scenario': f"Level {scenario_level}"
    }
    return pd.DataFrame([result])

def main():
    # 1. 실시간 데이터 수집
    current_wti = get_realtime_wti()
    print(f"현재 실시간 WTI 가격: ${current_wti:.2f}")

    # 2. 리스크 시뮬레이션 (현재 이란 상황 고려 Level 2 적용)
    sim_result = run_expert_simulation(current_wti, scenario_level=2)

    # 3. 결과 저장 (index.html 대시보드 연동용)
    sim_result.to_csv('simulation_result.csv', index=False)
    
    # 4. 시각화 리포트 생성
    plt.figure(figsize=(10,6))
    plt.bar(['Current WTI', 'Projected WTI', 'Projected ABS (Landed)'], 
            [current_wti, sim_result['Pred_WTI'][0], sim_result['Landed_Cost'][0]], 
            color=['gray', 'orange', 'red'])
    plt.title(f"Real-time Risk Impact Analysis (WTI ${current_wti:.2f})")
    plt.ylabel("Price (USD)")
    plt.savefig('risk_simulation_report.png')
    print("GitHub 대시보드 데이터 업데이트 완료.")

if __name__ == "__main__":
    main()
