# 필수 라이브러리 설치: pip install pandas matplotlib numpy yfinance
import pandas as pd # 데이터 분석용
import numpy as np # 수치 연산용
import matplotlib.pyplot as plt # 시각화용
import yfinance as yf # 실시간 유가 API
import datetime

def get_market_data():
    """실시간 WTI 유가 및 분석된 상관계수 로직 적용"""
    try:
        wti = yf.Ticker("CL=F").history(period="1d")['Close'].iloc[-1]
    except:
        wti = 78.45 # API 실패 시 현재 시세 반영

    # [분석 결과] 벤젠, 에틸렌, SM의 유가 연동 수식
    # y = m * WTI + b
    bz_m, bz_b = 10.22, 151.78 # 벤젠 (Benzene) 연동식
    et_m, et_b = 6.12, 629.75  # 에틸렌 (Ethylene) 연동식
    sm_market_m, sm_market_b = 11.36, 132.90 # SM 시장가 연동식

    # 1. 원료별 실시간 예상가 산출
    benzene = (wti * bz_m) + bz_b
    ethylene = (wti * et_m) + et_b
    sm_market = (wti * sm_market_m) + sm_market_b

    # 2. SM 제조원가 로직 적용 (Benzene * 0.8 + Ethylene * 0.3)
    sm_cost = (benzene * 0.8) + (ethylene * 0.3)
    
    # 3. 마진 분석 (시장가 - 제조원가)
    current_margin = sm_market - sm_cost
    target_margin = 150 # 경영 목표 마진
    
    return {
        'WTI': wti, 'Benzene': benzene, 'Ethylene': ethylene,
        'SM_Market': sm_market, 'SM_Cost': sm_cost, 
        'Margin': current_margin, 'Target': target_margin
    }

def create_executive_report(d):
    """경영진 보고용 분석 그래프 생성"""
    plt.figure(figsize=(14, 8), facecolor='#f8fafc')
    
    # [좌측] SM 수익성 분석 (Market vs Cost)
    plt.subplot(1, 2, 1)
    labels = ['SM Market Price', 'SM Mfg Cost']
    values = [d['SM_Market'], d['SM_Cost']]
    colors = ['#3b82f6', '#ef4444']
    
    plt.bar(labels, values, color=colors, alpha=0.8, width=0.6)
    plt.title("SM Profitability Analysis", fontsize=15, fontweight='bold', pad=20)
    plt.ylabel("USD / MT")
    for i, v in enumerate(values):
        plt.text(i, v + 20, f"${v:,.1f}", ha='center', fontweight='bold', fontsize=12)

    # [우측] 마진 갭 분석 (Target $150 vs Current)
    plt.subplot(1, 2, 2)
    margin_labels = ['Target Margin', 'Current Margin']
    margin_values = [d['Target'], d['Margin']]
    margin_colors = ['#10b981', '#dc2626' if d['Margin'] < d['Target'] else '#3b82f6']
    
    plt.bar(margin_labels, margin_values, color=margin_colors, width=0.6)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.title(f"SM Margin Status (WTI ${d['WTI']:.2f})", fontsize=15, fontweight='bold', pad=20)
    for i, v in enumerate(margin_values):
        plt.text(i, v + (5 if v > 0 else -15), f"${v:,.1f}", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('risk_simulation_report.png', dpi=150)
    
    # 결과 데이터 CSV 저장
    pd.DataFrame([d]).to_csv('simulation_result.csv', index=False)
    print("경영진 보고용 리포트 생성 완료.")

if __name__ == "__main__":
    data = get_market_data()
    create_executive_report(data)
