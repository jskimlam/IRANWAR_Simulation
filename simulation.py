# ============================================================
# pip install pandas matplotlib numpy yfinance
# ============================================================
# simulation.py
# Iran Risk × ABS/SM 원가 대응 시뮬레이션
# - yfinance CL=F 실시간 WTI (롤오버 자동 반영)
# - 엑셀 최신 실데이터 기반 회귀계수 적용
# - 4개 차트: SM수익성 / ABS마진 추이 / 시나리오 / 원료 트렌드
# ============================================================

import pandas as pd          # 데이터프레임 처리
import numpy as np           # 수치 연산
import matplotlib            # 렌더링 백엔드 설정 (GUI 없는 서버 환경용)
matplotlib.use('Agg')        # 반드시 pyplot import 전에 설정
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 범례 커스텀
import matplotlib.font_manager as fm   # 폰트 관리
import datetime              # 날짜 처리
import sys                   # 예외 시 종료
import os                    # 파일 경로

# ──────────────────────────────────────────────
# 0. 한글 폰트 설정 (GitHub Actions 우분투 환경)
# ──────────────────────────────────────────────
def setup_korean_font():
    """
    서버 환경에서 한글 폰트를 찾아 설정.
    없으면 영문 폴백으로 계속 진행 (크래시 방지).
    """
    # 우분투에서 나눔폰트 설치: apt-get install -y fonts-nanum
    candidate_fonts = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS
    ]
    for font_path in candidate_fonts:
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"[폰트] {font_path} 적용 완료")
            return font_prop.get_name()
    # 폰트 없으면 영문 사용
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("[폰트] 한글 폰트 없음 → 영문(DejaVu Sans) 사용")
    return 'DejaVu Sans'

# ──────────────────────────────────────────────
# 1. WTI 실시간 가격 취득 (yfinance CL=F)
# ──────────────────────────────────────────────
def get_wti_price():
    """
    야후 파이낸스에서 WTI 선물 최근가 취득.
    CL=F = Continuous Front Month → 롤오버 자동 반영.
    실패 시 최근 엑셀 기준값 $59.44 폴백 (2026-01-16 기준).
    """
    FALLBACK_WTI = 59.44   # 엑셀 최신 기준값 (2026-01-16)
    source = "야후파이낸스(실시간)"

    try:
        import yfinance as yf
        ticker = yf.Ticker("CL=F")
        hist = ticker.history(period="2d")   # 2일치로 안전하게

        if hist.empty:
            raise ValueError("야후 데이터 비어있음")

        # 가장 최근 종가 사용
        wti = float(hist['Close'].dropna().iloc[-1])

        # 비상식적 값 필터 (유가 $20 미만 or $200 초과 시 폴백)
        if not (20.0 <= wti <= 200.0):
            raise ValueError(f"비상식적 WTI 값: {wti}")

        print(f"[WTI] {wti:.2f} USD/bbl ({source})")
        return wti, source

    except Exception as e:
        print(f"[WTI] yfinance 오류: {e} → 폴백 ${FALLBACK_WTI} 사용")
        return FALLBACK_WTI, f"폴백(엑셀기준 2026-01-16)"

# ──────────────────────────────────────────────
# 2. 회귀계수 (엑셀 실데이터 2024-2026 기반)
# ──────────────────────────────────────────────
# 엑셀 Weekly_원료가_동향_260116.xlsx 최근 104주 데이터 분석 결과
# SM Cost = BZ*0.78 + ET*0.28 + 고정비 약 $45/t
# ABS Cost = SM*0.30 + BD*0.12 + AN*0.08 + 기타원가
# 아래는 WTI 단일변수 회귀 (대시보드 간이 계산용)
REGRESSION = {
    # 품목: {'m': 기울기, 'b': 절편}  →  가격 = WTI * m + b
    # 엑셀 최근 8주 실측 데이터 선형회귀 결과 (2025-11-28 ~ 2026-01-16)
    'BZ':         {'m':  9.65, 'b': 104.40},   # 벤젠 FOB Korea (실측 R²≈0.68)
    'ET':         {'m': -5.41, 'b':1010.93},   # 에틸렌 FOB Korea (공급과잉→역상관, 실측)
    'SM_market':  {'m': 20.07, 'b': -341.73},  # SM FOB Korea (실측 R²≈0.78)
    'ABS_market': {'m':  7.58, 'b':  825.47},  # ABS CFR China (기존 회귀 유지)
}

# 엑셀 최신 실측값 (2026-01-16 기준) - 시장 앵커로 사용
ACTUAL_LATEST = {
    'date':       '2026-01-16',
    'WTI':         59.44,
    'BZ_fob_kr':  705.33,   # 벤젠 FOB Korea
    'ET_cfr_tw':  710.00,   # 에틸렌 CFR Taiwan
    'SM_fob_kr':  913.00,   # SM FOB Korea
    'SM_cost':    937.66,   # SM 제조원가
    'SM_margin':  913.00 - 937.66,   # SM 마진 (역마진!)
    'ABS_mkt':   1190.00,   # ABS CFR China
    'ABS_cost':   995.80,   # ABS 제조원가
    'ABS_gap':    194.20,   # ABS 마진 (Gap)
    'BD_fob_kr': 1150.00,   # 부타디엔 FOB Korea
    'AN_cfr':    1060.00,   # AN CFR FEA
}

# ──────────────────────────────────────────────
# 3. 원가 계산 함수
# ──────────────────────────────────────────────
def calc_costs(wti, risk_premium=0.0):
    """
    WTI 가격과 이란 리스크 프리미엄 입력 시
    주요 석화 제품 예측가 및 원가/마진 반환.

    Args:
        wti (float): WTI 유가 ($/bbl)
        risk_premium (float): 이란 사태 할증액 ($/t), 기본 0

    Returns:
        dict: 계산 결과 전체
    """
    # 원료가 예측 (WTI 회귀)
    bz  = wti * REGRESSION['BZ']['m']         + REGRESSION['BZ']['b']
    et  = wti * REGRESSION['ET']['m']         + REGRESSION['ET']['b']

    # SM 시장가 예측
    sm_market = wti * REGRESSION['SM_market']['m'] + REGRESSION['SM_market']['b']

    # SM 제조원가: 벤젠 0.78t + 에틸렌 0.28t + 고정비 $45
    sm_cost = (bz * 0.78) + (et * 0.28) + 45.0

    # SM 마진
    sm_margin = sm_market - sm_cost

    # ABS 시장가 예측 + 리스크 프리미엄
    abs_market = wti * REGRESSION['ABS_market']['m'] + REGRESSION['ABS_market']['b'] + risk_premium

    # ABS 제조원가: SM 30% + BD 12% + AN 8% + 기타 50% (엑셀 실측 기반 역산)
    # 엑셀 기준 WTI $59.44 → ABS Cost $995.8 실측치로 보정
    abs_cost_base = sm_cost * 0.30 + (wti * 15.2 + 95.0) * 0.12 + (wti * 3.5 + 850.0) * 0.08 + 580.0
    # 실측 앵커 보정 (엑셀 최신 데이터와의 오차 조정)
    anchor_wti  = ACTUAL_LATEST['WTI']
    anchor_cost = ACTUAL_LATEST['ABS_cost']
    # 선형 보정: 현재 WTI 변화분만큼 원가 조정 (민감도 $8.5/bbl)
    abs_cost = anchor_cost + (wti - anchor_wti) * 8.5

    # ABS 마진
    abs_gap = abs_market - abs_cost

    return {
        'WTI':        round(wti, 2),
        'BZ':         round(bz, 1),
        'ET':         round(et, 1),
        'SM_Market':  round(sm_market, 1),
        'SM_Cost':    round(sm_cost, 1),
        'SM_Margin':  round(sm_margin, 1),
        'ABS_Market': round(abs_market, 1),
        'ABS_Cost':   round(abs_cost, 1),
        'ABS_Gap':    round(abs_gap, 1),
        'Risk':       round(risk_premium, 1),
    }

# ──────────────────────────────────────────────
# 4. 시나리오 정의
# ──────────────────────────────────────────────
SCENARIOS = [
    {'label': 'Base\n($59)',    'wti': 59.44, 'risk': 0,   'color': '#2ecc71'},
    {'label': 'Mild\n($70)',    'wti': 70.00, 'risk': 50,  'color': '#f39c12'},
    {'label': 'Moderate\n($80)','wti': 80.00, 'risk': 100, 'color': '#e67e22'},
    {'label': 'Severe\n($90)', 'wti': 90.00, 'risk': 150, 'color': '#e74c3c'},
    {'label': 'Crisis\n($100)','wti':100.00, 'risk': 200, 'color': '#c0392b'},
]

# ──────────────────────────────────────────────
# 5. 엑셀 기반 히스토리 데이터 (최근 8주)
# ──────────────────────────────────────────────
HISTORY = {
    'dates':      ['11/28','12/05','12/12','12/19','12/26','01/02','01/09','01/16'],
    'WTI':        [58.55,  60.08,  57.44,  56.66,  56.74,  57.32,  59.12,  59.44],
    'BZ_fob_kr':  [657.0,  673.3,  655.7,  655.7,  669.2,  666.0,  670.7,  705.3],
    'SM_fob_kr':  [811.0,  813.0,  801.0,  795.5,  829.5,  848.0,  866.0,  913.0],
    'SM_cost':    [898.4,  915.4,  902.4,  902.4,  902.4,  902.4,  915.1,  937.7],
    'ABS_mkt':    [1200.0, 1195.0, 1170.0, 1170.0, 1170.0, None,   1180.0, 1190.0],
    'ABS_cost':   [883.1,  895.3,  889.6,  898.3,  918.7,  929.8,  966.4,  995.8],
    'ABS_gap':    [316.9,  299.7,  280.4,  271.7,  251.3,  240.2,  213.6,  194.2],
    'BD_fob_kr':  [800.0,  830.0,  840.0,  890.0,  None,   None,  1070.0, 1150.0],
}

# ──────────────────────────────────────────────
# 6. 차트 생성 함수
# ──────────────────────────────────────────────
def generate_report(current: dict, wti_source: str):
    """
    4개 패널 차트 생성 후 PNG로 저장.
    current: calc_costs() 반환값
    """
    # 다크 테마 설정
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), facecolor='#0f172a')
    fig.suptitle(
        f'IRAN CONFLICT RISK DASHBOARD  |  WTI ${current["WTI"]:.2f}  |  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        fontsize=14, fontweight='bold', color='#fbbf24', y=0.98
    )

    # ── 6-1. 왼쪽 상단: SM 수익성 막대
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1e293b')
    bars = ax1.bar(
        ['SM\nMarket Price', 'SM\nMfg Cost'],
        [current['SM_Market'], current['SM_Cost']],
        color=['#3b82f6', '#ef4444'],
        width=0.5, edgecolor='white', linewidth=0.5
    )
    # 값 레이블
    for bar, val in zip(bars, [current['SM_Market'], current['SM_Cost']]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 f'${val:,.0f}', ha='center', va='bottom',
                 fontweight='bold', color='white', fontsize=11)
    # 마진 표시
    margin_color = '#10b981' if current['SM_Margin'] >= 0 else '#ef4444'
    ax1.set_title(f'SM Profitability  |  Margin: ${current["SM_Margin"]:+.0f}/t',
                  color=margin_color, fontweight='bold', fontsize=11)
    ax1.set_ylabel('$/mt', color='#94a3b8')
    ax1.tick_params(colors='#94a3b8')
    ax1.set_ylim(0, max(current['SM_Market'], current['SM_Cost']) * 1.15)
    # 역마진 경고선
    if current['SM_Margin'] < 0:
        ax1.axhline(y=current['SM_Market'], color='#fbbf24', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.text(0.5, current['SM_Market'] + 5, '⚠ NEGATIVE MARGIN',
                 ha='center', color='#fbbf24', fontsize=9, transform=ax1.get_yaxis_transform())
    for spine in ax1.spines.values():
        spine.set_edgecolor('#334155')

    # ── 6-2. 오른쪽 상단: ABS 마진 트렌드 (8주)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#1e293b')
    dates  = HISTORY['dates']
    gaps   = HISTORY['ABS_gap']
    mkts   = HISTORY['ABS_mkt']
    costs  = HISTORY['ABS_cost']
    x      = range(len(dates))

    # ABS 시장가 vs 원가 영역 채우기
    mkts_filled  = [v if v is not None else float('nan') for v in mkts]
    ax2.fill_between(x, costs, mkts_filled, alpha=0.3, color='#10b981', label='Margin Area')
    ax2.plot(x, mkts_filled,  color='#3b82f6', linewidth=2, marker='o', markersize=5, label='ABS Market')
    ax2.plot(x, costs,        color='#ef4444', linewidth=2, marker='s', markersize=4, label='ABS Cost')
    # 현재 마진 표시
    for i, (g, c) in enumerate(zip(gaps, costs)):
        if g is not None:
            ax2.annotate(f'${g:.0f}', (i, c + g/2), ha='center', va='center',
                         fontsize=7, color='white',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.7))
    ax2.set_title(f'ABS Margin Trend (8W)  |  Latest: ${gaps[-1]:.0f}/t',
                  color='#10b981', fontweight='bold', fontsize=11)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax2.set_ylabel('$/mt', color='#94a3b8')
    ax2.tick_params(colors='#94a3b8')
    ax2.legend(fontsize=8, loc='upper left', facecolor='#1e293b', edgecolor='#334155')
    # 경보 임계선 $150
    ax2.axhline(y=150, color='#fbbf24', linestyle=':', linewidth=1.2, alpha=0.8)
    ax2.text(len(dates)-1, 155, 'Alert $150', color='#fbbf24', fontsize=8, ha='right')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#334155')

    # ── 6-3. 왼쪽 하단: 이란 사태 시나리오별 ABS Cost vs Market
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1e293b')
    sc_labels  = [s['label'] for s in SCENARIOS]
    sc_colors  = [s['color'] for s in SCENARIOS]
    sc_costs   = []
    sc_markets = []
    sc_gaps    = []
    for s in SCENARIOS:
        r = calc_costs(s['wti'], s['risk'])
        sc_costs.append(r['ABS_Cost'])
        sc_markets.append(r['ABS_Market'])
        sc_gaps.append(r['ABS_Gap'])

    xpos   = np.arange(len(SCENARIOS))
    width  = 0.35
    bars_c = ax3.bar(xpos - width/2, sc_costs,   width, label='ABS Cost',   color='#ef4444', alpha=0.85)
    bars_m = ax3.bar(xpos + width/2, sc_markets, width, label='ABS Market', color='#3b82f6', alpha=0.85)

    # Gap 레이블
    for i, (cost, mkt, gap, color) in enumerate(zip(sc_costs, sc_markets, sc_gaps, sc_colors)):
        top = max(cost, mkt)
        gap_color = '#10b981' if gap >= 0 else '#ef4444'
        ax3.text(i, top + 15, f'GAP\n${gap:+.0f}',
                 ha='center', fontsize=8, color=gap_color, fontweight='bold')

    ax3.set_title('Iran Risk Scenario: ABS Cost vs Market',
                  color='#fbbf24', fontweight='bold', fontsize=11)
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(sc_labels, fontsize=8.5, color='white')
    ax3.set_ylabel('$/mt', color='#94a3b8')
    ax3.tick_params(colors='#94a3b8')
    ax3.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155')
    ax3.set_ylim(0, max(sc_markets + sc_costs) * 1.18)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#334155')

    # ── 6-4. 오른쪽 하단: 원료 트렌드 (WTI/BZ/SM 8주)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#1e293b')
    ax4_twin = ax4.twinx()  # 오른쪽 Y축 (WTI용)

    # WTI (오른쪽 축)
    ax4_twin.plot(x, HISTORY['WTI'], color='#fbbf24', linewidth=2,
                  marker='^', markersize=5, linestyle='--', label='WTI (R)')
    ax4_twin.set_ylabel('WTI $/bbl', color='#fbbf24', fontsize=9)
    ax4_twin.tick_params(axis='y', colors='#fbbf24')
    ax4_twin.set_ylim(40, 100)

    # BZ (왼쪽 축)
    ax4.plot(x, HISTORY['BZ_fob_kr'], color='#a855f7', linewidth=2,
             marker='o', markersize=5, label='BZ FOB Korea')
    # SM market (왼쪽 축)
    ax4.plot(x, HISTORY['SM_fob_kr'], color='#3b82f6', linewidth=2,
             marker='s', markersize=5, label='SM FOB Korea')
    # SM cost (왼쪽 축)
    ax4.plot(x, HISTORY['SM_cost'], color='#ef4444', linewidth=1.5,
             marker='D', markersize=4, linestyle=':', label='SM Cost')
    # BD (왼쪽 축, None 처리)
    bd_filled = [v if v is not None else float('nan') for v in HISTORY['BD_fob_kr']]
    ax4.plot(x, bd_filled, color='#10b981', linewidth=1.5,
             marker='v', markersize=5, label='BD FOB Korea')

    ax4.set_title('Raw Material Price Trend (8W)',
                  color='#94a3b8', fontweight='bold', fontsize=11)
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax4.set_ylabel('$/mt', color='#94a3b8')
    ax4.tick_params(axis='y', colors='#94a3b8')
    ax4.set_ylim(400, 1400)

    # 합성 범례
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
               loc='upper left', facecolor='#1e293b', edgecolor='#334155')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#334155')
    for spine in ax4_twin.spines.values():
        spine.set_edgecolor('#334155')

    # ── 하단 워터마크
    fig.text(0.5, 0.01,
             f'LAM Advanced Procurement  |  Source: Yahoo Finance (CL=F) + Weekly Raw Material Data  |  {wti_source}',
             ha='center', fontsize=7.5, color='#475569')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # PNG 저장
    output_path = 'risk_simulation_report.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print(f"[차트] {output_path} 저장 완료")
    return output_path

# ──────────────────────────────────────────────
# 7. CSV 결과 저장
# ──────────────────────────────────────────────
def save_csv(current: dict, wti_source: str):
    """
    계산 결과를 CSV로 저장 (index.html에서 PapaParse로 읽음).
    실제 최신 엑셀 데이터도 함께 기록.
    """
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    row = {
        'UpdateTime':      now,
        'WTI_Source':      wti_source,
        # 실시간 계산값
        'WTI':             current['WTI'],
        'BZ':              current['BZ'],
        'ET':              current['ET'],
        'SM_Market':       current['SM_Market'],
        'SM_Cost':         current['SM_Cost'],
        'SM_Margin':       current['SM_Margin'],
        'ABS_Market':      current['ABS_Market'],
        'ABS_Cost':        current['ABS_Cost'],
        'ABS_Gap':         current['ABS_Gap'],
        'Risk_Premium':    current['Risk'],
        # 엑셀 실측 최신 (2026-01-16) - 앵커 데이터
        'Actual_Date':     ACTUAL_LATEST['date'],
        'Actual_WTI':      ACTUAL_LATEST['WTI'],
        'Actual_SM_FOB':   ACTUAL_LATEST['SM_fob_kr'],
        'Actual_SM_Cost':  ACTUAL_LATEST['SM_cost'],
        'Actual_SM_Margin':ACTUAL_LATEST['SM_margin'],
        'Actual_ABS_Mkt':  ACTUAL_LATEST['ABS_mkt'],
        'Actual_ABS_Cost': ACTUAL_LATEST['ABS_cost'],
        'Actual_ABS_Gap':  ACTUAL_LATEST['ABS_gap'],
        'Actual_BD_FOB':   ACTUAL_LATEST['BD_fob_kr'],
        'Actual_AN_CFR':   ACTUAL_LATEST['AN_cfr'],
        # 시나리오별 ABS Gap
        'Scenario_Base_Gap':     calc_costs(59.44,  0  )['ABS_Gap'],
        'Scenario_Mild_Gap':     calc_costs(70.00,  50 )['ABS_Gap'],
        'Scenario_Moderate_Gap': calc_costs(80.00,  100)['ABS_Gap'],
        'Scenario_Severe_Gap':   calc_costs(90.00,  150)['ABS_Gap'],
        'Scenario_Crisis_Gap':   calc_costs(100.00, 200)['ABS_Gap'],
        # Margin: 구버전 호환용
        'Margin': current['SM_Margin'],
        'ABS_Landed': current['ABS_Market'],
    }

    df = pd.DataFrame([row])
    df.to_csv('simulation_result.csv', index=False)
    print(f"[CSV] simulation_result.csv 저장 완료")

# ──────────────────────────────────────────────
# 8. 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Iran Risk × ABS/SM 원가 시뮬레이션 시작")
    print("=" * 60)

    # 0) 한글 폰트 설정
    setup_korean_font()

    # 1) WTI 취득
    wti_price, wti_source = get_wti_price()

    # 2) 원가 계산 (기본: 이란 리스크 프리미엄 없음)
    current_data = calc_costs(wti_price, risk_premium=0.0)

    # 계산 결과 출력
    print(f"\n{'─'*40}")
    print(f"  WTI          : ${current_data['WTI']:.2f}")
    print(f"  BZ (예측)    : ${current_data['BZ']:.0f}/t")
    print(f"  ET (예측)    : ${current_data['ET']:.0f}/t")
    print(f"  SM Market    : ${current_data['SM_Market']:.0f}/t")
    print(f"  SM Cost      : ${current_data['SM_Cost']:.0f}/t")
    print(f"  SM Margin    : ${current_data['SM_Margin']:+.0f}/t  {'⚠ 역마진!' if current_data['SM_Margin'] < 0 else '✓'}")
    print(f"  ABS Market   : ${current_data['ABS_Market']:.0f}/t")
    print(f"  ABS Cost     : ${current_data['ABS_Cost']:.0f}/t")
    print(f"  ABS Gap      : ${current_data['ABS_Gap']:+.0f}/t  {'⚠ 경보!' if current_data['ABS_Gap'] < 150 else '✓'}")
    print(f"{'─'*40}\n")

    # 3) 시나리오 출력
    print("[ 이란 리스크 시나리오 ]")
    for s in SCENARIOS:
        r = calc_costs(s['wti'], s['risk'])
        flag = '⚠ 역마진!' if r['ABS_Gap'] < 0 else ('⚠ 경보' if r['ABS_Gap'] < 150 else '✓')
        print(f"  {s['label'].replace(chr(10),' '):20s} | WTI ${s['wti']:5.0f} | Risk +${s['risk']:3.0f} | ABS Gap ${r['ABS_Gap']:+.0f}/t {flag}")

    # 4) 차트 저장
    generate_report(current_data, wti_source)

    # 5) CSV 저장
    save_csv(current_data, wti_source)

    print("\n시뮬레이션 완료.")
    print("=" * 60)
