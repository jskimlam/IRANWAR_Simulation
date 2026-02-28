import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

def get_market_data():
    """실시간 WTI 가격 및 이란 사태 리스크 파라미터 적용"""
    try:
        # 실시간 WTI 선물 시세 (정확한 가격 추출)
        wti_ticker = yf.Ticker("CL=F")
        current_wti = wti_ticker.history(period="1d")['Close'].iloc[-1]
    except:
        current_wti = 78.50 # API 장애 시 최근 시장가 기준

    # 1. 업로드 데이터 기반 상관계수
    BZ_M, BZ_B = 10.22, 151.78 # 벤젠
    ET_M, ET_B = 6.12, 629.75  # 에틸렌
    ABS_M, ABS_B = 8.23, 1103.54 # ABS
    SM_MARKET_M, SM_MARKET_B = 11.36, 132.90 # SM 시장가

    # 2. 실시간 가격 산출
    bz_p = (current_wti * BZ_M) + BZ_B
    et_p = (current_wti * ET_M) + ET_B
    abs_base = (current_wti * ABS_M) + ABS_B
    sm_market = (current_wti * SM_MARKET_M) + SM_MARKET_B
    
    # 3. SM 원가 로직 (사용자 정의: 벤젠 0.8 + 에틸렌 0.3)
    sm_cost = (bz_p * 0.8) + (et_p * 0.3)
    
    # 4. 이란 사태 리스크 할증 (Logistics & Insurance)
    risk_premium = 150.0
    abs_landed = abs_base + risk_premium
    sm_landed = sm_market + risk_premium

    return {
        'Update_Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'WTI': current_wti,
        'BZ': bz_p, 'ET': et_p,
        'ABS_Base': abs_base, 'ABS_Landed': abs_landed,
        'SM_Market': sm_market, 'SM_Cost': sm_cost, 'SM_Landed': sm_landed,
        'Margin': sm_market - sm_cost,
        'Risk_Premium': risk_premium
    }

def create_report(d):
    """이란 사태 대응 비상 경영 리포트 생성"""
    plt.figure(figsize=(14, 7), facecolor='#ffffff')
    
    # [좌측] ABS/SM 원가 상승 트렌드
    plt.subplot(1, 2, 1)
    labels = ['ABS Base', 'ABS Landed', 'SM Cost', 'SM Market']
    values = [d['ABS_Base'], d['ABS_Landed'], d['SM_Cost'], d['SM_Market']]
    plt.bar(labels, values, color=['#94a3b8', '#ef4444', '#fb923c', '#3b82f6'])
    plt.title("Iran Conflict: Cost Impact Analysis", fontsize=14, fontweight='bold')
    for i, v in enumerate(values):
        plt.text(i, v + 10, f"${v:,.1f}", ha='center', fontweight='bold')

    # [우측] SM 수익성 및 리스크 할증 비중
    plt.subplot(1, 2, 2)
    plt.pie([d['SM_Cost'], d['Risk_Premium'], max(0, d['Margin'])], 
            labels=['Production Cost', 'Iran Risk', 'Margin'],
            autopct='%1.1f%%', colors=['#cbd5e1', '#f87171', '#60a5fa'], startangle=140)
    plt.title("SM Revenue Structure (incl. Risk)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('risk_simulation_report.png', dpi=150)
    pd.DataFrame([d]).to_csv('simulation_result.csv', index=False)

if __name__ == "__main__":
    data = get_market_data()
    create_report(data)
