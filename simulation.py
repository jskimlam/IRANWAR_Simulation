# pip install pandas matplotlib numpy
import pandas as pd # 데이터 분석을 위한 판다스 라이브러리 임포트
import numpy as np # 수치 계산을 위한 넘파이 라이브러리 임포트
import matplotlib.pyplot as plt # 시각화를 위한 맷플롯립 라이브러리 임포트
import os # 파일 경로 및 환경 설정을 위한 os 라이브러리 임포트

def generate_sample_data():
    """실무 분석을 위한 가상의 과거 시장 데이터 생성 함수"""
    try:
        # 최근 30일간의 가상 기초 데이터 생성
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30) # 오늘 기준 과거 30일 날짜 생성
        oil_prices = np.linspace(75, 85, 30) + np.random.normal(0, 1, 30) # 배럴당 75~85달러 유가 생성
        
        # 석유화학 원료 간 상관관계를 반영한 가상 가격 로직 (유가 -> 나프타 -> ABS)
        data = {
            'Date': dates, # 날짜 데이터
            'WTI_Oil': oil_prices, # WTI 유가
            'Naphtha': oil_prices * 10 + 50, # 나프타 가격 (유가 연동 가상 로직)
            'SM': oil_prices * 15 + 200, # 스티렌 모노머 가격
            'BD': oil_prices * 12 + 150, # 부타디엔 가격
            'ABS_Cost': oil_prices * 18 + 500 # 최종 ABS 제조 원가
        }
        return pd.DataFrame(data) # 생성된 데이터를 판다스 데이터프레임으로 반환
    except Exception as e:
        print(f"데이터 생성 중 오류 발생: {e}") # 에러 메시지 출력
        return None # 오류 시 빈 값 반환

def run_risk_simulation(df, scenario_level=2):
    """지정학적 리스크 시나리오별 가격 변동 예측 함수"""
    # 시나리오 설정: 1(국지적), 2(해협 봉쇄), 3(전면전)
    scenarios = {
        1: {'oil_jump': 1.10, 'logistics_premium': 1.05}, # 유가 10% 상승, 물류비 5% 상승
        2: {'oil_jump': 1.30, 'logistics_premium': 1.50}, # 유가 30% 상승, 물류비 50% 상승
        3: {'oil_jump': 1.60, 'logistics_premium': 3.00}  # 유가 60% 상승, 물류비 200% 상승
    }
    
    config = scenarios.get(scenario_level, scenarios[1]) # 선택된 시나리오 로직 불러오기
    
    try:
        # 시뮬레이션 결과 데이터프레임 복사
        sim_df = df.tail(1).copy() # 가장 최근 데이터 기준으로 시뮬레이션 시작
        sim_df['Scenario'] = f"Level {scenario_level}" # 현재 적용된 시나리오 레벨 기록
        
        # 리스크가 반영된 미래 가격 예측 계산
        sim_df['Pred_WTI'] = sim_df['WTI_Oil'] * config['oil_jump'] # 예측 유가 계산
        sim_df['Pred_ABS'] = sim_df['ABS_Cost'] * (1 + (config['oil_jump'] - 1) * 0.8) # ABS 원가 전이율 80% 가정
        sim_df['Landed_Cost'] = sim_df['Pred_ABS'] * config['logistics_premium'] # 물류 프리미엄이 포함된 최종 도착가
        
        return sim_df # 시뮬레이션 결과 반환
    except Exception as e:
        print(f"시뮬레이션 계산 중 '파일을 찾을 수 없거나 데이터 오류'가 발생했습니다: {e}") # 한글 오류 안내
        return None

def visualize_and_save(df, sim_result):
    """결과를 그래프로 시각화하고 파일로 저장하는 함수"""
    try:
        plt.figure(figsize=(12, 6)) # 그래프 크기 설정
        plt.plot(df['Date'], df['ABS_Cost'], label='Historical ABS Cost', color='blue', marker='o') # 과거 가격 선 그래프
        
        # 예측 데이터 포인트 추가
        future_date = df['Date'].iloc[-1] + pd.Timedelta(days=7) # 현재로부터 7일 후 날짜 설정
        plt.scatter(future_date, sim_result['Landed_Cost'], color='red', s=200, label='Projected Landed Cost (Risk)') # 리스크 반영 지점 표시
        
        plt.title('Middle East Conflict: ABS Cost & Logistics Risk Projection', fontsize=15) # 그래프 제목
        plt.xlabel('Date') # X축 라벨
        plt.ylabel('Cost (USD/MT)') # Y축 라벨
        plt.legend() # 범례 표시
        plt.grid(True, linestyle='--') # 그리드 추가
        
        # 결과 저장
        plt.savefig('risk_simulation_report.png') # 그래프를 이미지 파일로 저장
        sim_result.to_csv('simulation_result.csv', index=False) # 상세 수치를 CSV로 저장
        print("시뮬레이션 리포트(이미지 및 CSV)가 성공적으로 생성되었습니다.") # 완료 메시지
    except Exception as e:
        print(f"시각화 파일 생성 중 오류 발생: {e}") # 에러 메시지 출력

def main():
    """메인 실행 로직"""
    # 1. 데이터 로드
    market_data = generate_sample_data() # 가상 데이터 생성
    if market_data is None: return # 데이터 없으면 종료
    
    # 2. 리스크 시뮬레이션 실행 (현재 상황을 고려하여 Level 2: 해협 봉쇄 시나리오 적용)
    simulation_result = run_risk_simulation(market_data, scenario_level=2) # 시나리오 실행
    
    # 3. 결과 저장 및 시각화
    if simulation_result is not None:
        visualize_and_save(market_data, simulation_result) # 파일 저장 실행

if __name__ == "__main__":
    main() # 스크립트 실행
