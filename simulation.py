# pip install pandas matplotlib numpy yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

def run_sm_profitability_analysis():
    # 1. 실시간 WTI 유가 수집 (Yahoo Finance)
    try:
        wti_ticker = yf.Ticker("CL=F")
        current_wti = wti_ticker.history(period="1d")['Close'].iloc[-1]
    except:
        current_wti = 78.45 # API 호출 실패 시 최근 시세 적용

    # 2. 업로드 데이터 기반 상관계수 산출 결과
    # Benzene (Row 22), Ethylene (Row 8), SM (Row 24)
    bz_m, bz_b = 10.22, 151.78   # 벤젠 연동식
    et_m, et_b = 6.12, 629.75    # 에틸렌 연동식
    sm_m, sm_b = 11.36, 132.90   # SM 시장가 연동식

    # 3. 실시간 가격 및 원가 계산 (사용자 정의 로직 적용)
    benzene = (current_wti * bz_m) + bz_b
    ethylene = (current_wti * et_m) + et_b
    sm_market_price = (current_wti * sm_m) + sm_b
    
    # SM 원가 로직: 벤젠 * 0.8 + 에틸렌 * 0.3 [사용자 제시 로직]
    sm_mfg_cost = (benzene * 0.8) + (ethylene * 0.3)
    
    # 수익성 분석
    actual_margin = sm_market_price - sm_mfg_cost
    target_margin = 150 # 적정 마진 기준
    margin_gap = actual_margin - target_margin

    # 4. 결과 저장
    res_data = {
        'Update_Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'WTI': current_wti,
        'Benzene': benzene,
        'Ethylene': ethylene,
        'SM_Market': sm_market_price,
        'SM_Cost': sm_mfg_cost,
        'Margin': actual_margin,
        'Status': 'Margin Squeeze' if actual_margin < target_margin else 'Healthy'
    }
    pd.DataFrame([res_data]).to_csv('simulation_result.csv', index=False)

    # 5. 경영진 보고용 그래프 생성
    plt.figure(figsize=(12, 6), facecolor='#f8fafc')
    
    # SM Market vs Cost 비교
    plt.subplot(1, 2, 1)
    plt.bar(['Market Price', 'Mfg Cost'], [sm_market_price, sm_mfg_cost], color=['#3b82f6', '#ef4444'], alpha=0.8)
    plt.title("SM Price vs Cost (Real-time)", fontsize=13, fontweight='bold')
    plt.ylabel("USD/MT")
    for i, v in enumerate([sm_market_price, sm_mfg_cost]):
        plt.text(i, v + 10, f"${v:,.1f}", ha='center', fontweight='bold')

    # 마진 현황 (Target $150 대비)
    plt.subplot(1, 2, 2)
    plt.bar(['Target', 'Actual'], [target_margin, actual_margin], color=['#10b981', '#f59e0b'], alpha=0.8)
    plt.axhline(y=target_margin, color='red', linestyle='--', alpha=0.5)
    plt.title(f"SM Margin Gap (Target $150)", fontsize=13, fontweight='bold')
    for i, v in enumerate([target_margin, actual_margin]):
        plt.text(i, v + 5, f"${v:,.1f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('risk_simulation_report.png', dpi=150)
    print(f"분석 완료: WTI ${current_wti}, SM 마진 ${actual_margin:.1f}")

if __name__ == "__main__":
    run_sm_profitability_analysis()
