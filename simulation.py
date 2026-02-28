# 필수 라이브러리 설치: pip install pandas matplotlib numpy yfinance
import pandas as pd # 데이터 핸들링용 판다스 임포트
import numpy as np # 수치 연산용 넘파이 임포트
import matplotlib.pyplot as plt # 시각화용 맷플롯립 임포트
import yfinance as yf # 실시간 금융 데이터 API 임포트
import datetime # 시간 정보 처리를 위한 라이브러리 임포트

def get_realtime_data():
    """실시간 유가 수집 및 분석된 상관계수 적용"""
    try:
        # 실시간 WTI 유가 가져오기 (Yahoo Finance)
        wti_ticker = yf.Ticker("CL=F")
        current_wti = wti_ticker.history(period="1d")['Close'].iloc[-1]
    except Exception:
        current_wti = 78.45 # API 오류 시 현재 시세 수동 반영

    # 1. 업로드 파일 기반 상관계수 산출 결과 (y = mx + b)
    BZ_M, BZ_B = 10.22, 151.78   # 벤젠(Benzene) 상관계수
    ET_M, ET_B = 6.12, 629.75    # 에틸렌(Ethylene) 상관계수
    SM_M, SM_B = 11.36, 132.90   # SM 시장가 상관계수

    # 2. 실시간 가격 도출
    benzene = (current_wti * BZ_M) + BZ_B
    ethylene = (current_wti * ET_M) + ET_B
    sm_market = (current_wti * SM_M) + SM_B
    
    # 3. 사용자 정의 SM 원가 로직: (Benzene * 0.8) + (Ethylene * 0.3)
    sm_cost = (benzene * 0.8) + (ethylene * 0.3)
    
    # 4. 수익성 및 마진 분석 (Margin = Market - Cost)
    actual_margin = sm_market - sm_cost
    target_margin = 150.0 # 경영 목표 마진
    
    return {
        'Update_Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'WTI': current_wti,
        'BZ': benzene, 'ET': ethylene,
        'SM_Market': sm_market, 'SM_Cost': sm_cost,
        'Margin': actual_margin, 'Target': target_margin,
        'BZ_M': BZ_M, 'BZ_B': BZ_B, 'ET_M': ET_M, 'ET_B': ET_B # 로직 설명용 계수 포함
    }

def save_visual_report(d):
    """경영진 보고용 수익성 차트 생성"""
    plt.figure(figsize=(12, 6), facecolor='#ffffff')
    
    # [좌측] 가격 비교: 시장가 vs 제조원가
    plt.subplot(1, 2, 1)
    plt.bar(['Market Price', 'Mfg Cost'], [d['SM_Market'], d['SM_Cost']], color=['#2563eb', '#dc2626'], alpha=0.8)
    plt.title("SM Market vs Theoretical Cost", fontsize=14, fontweight='bold', pad=15)
    for i, v in enumerate([d['SM_Market'], d['SM_Cost']]):
        plt.text(i, v + 10, f"${v:,.1f}", ha='center', fontweight='bold')

    # [우측] 마진 현황: 목표 $150 대비 실적
    plt.subplot(1, 2, 2)
    plt.bar(['Target Margin', 'Actual Margin'], [d['Target'], d['Margin']], color=['#10b981', '#f59e0b'], alpha=0.8)
    plt.axhline(y=d['Target'], color='red', linestyle='--', alpha=0.4)
    plt.title("SM Margin Analysis", fontsize=14, fontweight='bold', pad=15)
    for i, v in enumerate([d['Target'], d['Margin']]):
        plt.text(i, v + 5, f"${v:,.1f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('risk_simulation_report.png', dpi=150)
    
    # 상세 결과 CSV 저장 (대시보드 연동)
    pd.DataFrame([d]).to_csv('simulation_result.csv', index=False)
    print(f"업데이트 완료: WTI ${d['WTI']:.2f} 기준 SM 마진 ${d['Margin']:.1f}")

if __name__ == "__main__":
    report_data = get_realtime_data()
    save_visual_report(report_data)
