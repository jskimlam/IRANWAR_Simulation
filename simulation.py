# ============================================================
# pip install pandas matplotlib numpy yfinance requests openpyxl
# ============================================================
# simulation.py - Iran Risk × ABS/SM 원가 대응 시뮬레이션
# v6.2 - 이란 리스크 프리미엄 전 원료 회귀 반영
# ─────────────────────────────────────────────────────────────
# 핵심 변경 (v6.1 → v6.2):
#
#   [문제] 리스크 프리미엄이 ABS Market에만 flat 가산
#          → 원료가는 그대로 → Gap이 프리미엄만큼 개선되는 왜곡
#
#   [수정] 리스크 프리미엄 → WTI 등가(wti_equiv) 환산
#          → wti_equiv를 전 원료 회귀에 추가 반영
#          → Gap은 원가 상승으로 자연스럽게 악화
#
#   [공식]
#     NAP 민감도(6.52)가 가장 높은 WTI 연동률 → 환산 기준으로 사용
#     wti_equiv = risk_premium / NAP_SENS  (NAP은 WTI 거의 100% 연동)
#
#     각 원료 보정값 = 실측값
#                    + (wti_rt - wti_gs) × 회귀민감도       ← 실시간 WTI 델타
#                    + wti_equiv         × 회귀민감도       ← 리스크 프리미엄 원료 환산
#
#     ABS Gap = (ABS Market + risk_premium) - ABS Cost(원료 상승 반영)
#             → 프리미엄 올릴수록 원가도 함께 올라 Gap 개선 착시 제거
#
#   [이론 마진 기준선] BZ×0.80 + ET×0.30 + 150 (교과서 적정 배합비)
#     실측 원가 vs 이론 마진 GAP → 수급 이상 신호로 활용
# ─────────────────────────────────────────────────────────────
# 민감도 (557일 일간 데이터 확정):
#   ABS Gap = 실측 Gap + WTI 델타 × (-5.76)  → WTI↑=Gap↓
#   SM Margin = WTI 무상관 (R²=0.000)
#   NAP: R²=0.880 최고 → WTI 환산 기준 채택
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime
import os
import io

# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
GSHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQfp5e4ufXsgu7YvZ5EFEHatkJ7BdgW3vma78THWYn66wHUrau8hYB4q8TY2OXuc9PBguq-v09CkmKZ"
    "/pub?gid=0&single=true&output=csv"
)

COL_MAP = {
    'wti':     'NYMEX Light Sweet Crude Settlement Mo01',
    'nap':     'Naphtha C+F Japan Cargo $/mt (NextGen MOC)',
    'sm_cn':   'Styrene CFR China Marker LC 90 days',
    'sm_fob':  'Styrene Monomer FOB China Marker',
    'et':      'Ethylene CFR NE Asia',
    'bz':      'Benzene FOB Korea Marker',
    'bz_ara':  'Benzene CIF ARA',
    'bz_usg':  'Benzene FOB USG Mo02 cts/gal',
    'pr':      'Propylene Poly Grade CFR China',
    'bd':      'Butadiene CFR China',
    'an':      'ACN CFR FE Asia Weekly',
    'abs_mkt': 'ABS Inj CFR China Weekly',
}

ABS_RATIO       = {'sm': 0.60, 'an': 0.25, 'bd': 0.15}
SM_COST_RATIO   = {'bz': 0.67, 'et': 0.25, 'nap': 0.05, 'fixed': 150}  # 실측 회귀 기반
SM_THEORY_RATIO = {'bz': 0.80, 'et': 0.30, 'fixed': 150}                # 이론 적정 마진

DEFAULT_SENS = {
    'bz':        16.81,
    'et':         6.13,
    'sm':        14.75,
    'an':         7.87,
    'bd':        20.48,
    'nap':        6.52,  # R²=0.880 최고 → WTI 환산 기준
    'pr':         4.18,
    'bz_ara':    15.20,
    'bz_usg':     0.85,
    'sm_cost':   14.50,
    'sm_margin':  -0.08,
    'abs_cost':  13.89,
    'abs_mkt':    8.12,
    'abs_gap':   -5.76,
}

# ★ v6.2 핵심: 리스크 프리미엄 → WTI 등가 환산 기준
# NAP R²=0.880 최고, 이란 충격은 NAP을 통해 전파 → 기준으로 채택
# wti_equiv = risk_premium / NAP_SENS_FOR_EQUIV
NAP_SENS_FOR_EQUIV = DEFAULT_SENS['nap']  # 자동 갱신됨

BZ_USG_TO_MT = 26.42  # cts/gal → $/mt


# ──────────────────────────────────────────────
# 1. 한글 폰트
# ──────────────────────────────────────────────
def setup_font():
    candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    ]
    for fp in candidates:
        if os.path.exists(fp):
            plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"[폰트] {fp}")
            return
    plt.rcParams['font.family'] = 'DejaVu Sans'


# ──────────────────────────────────────────────
# 2. 구글시트 파싱
# ──────────────────────────────────────────────
def load_gsheet():
    import requests
    print("[구글시트] 데이터 로드 중...")
    try:
        resp = requests.get(GSHEET_CSV_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"[구글시트] 로드 실패: {e}")
        return None, None, None

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df[COL_MAP['wti']].notna()].copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    if df.empty:
        print("[구글시트] 유효 데이터 없음")
        return None, None, None

    # 파생 컬럼
    df['_abs_cost'] = (df[COL_MAP['sm_cn']] * ABS_RATIO['sm'] +
                       df[COL_MAP['an']]    * ABS_RATIO['an'] +
                       df[COL_MAP['bd']]    * ABS_RATIO['bd'])
    df['_abs_gap']  = df[COL_MAP['abs_mkt']] - df['_abs_cost']

    # SM Cost 실측 회귀 기반
    df['_sm_cost']  = (df[COL_MAP['bz']]  * SM_COST_RATIO['bz']  +
                       df[COL_MAP['et']]  * SM_COST_RATIO['et']  +
                       df[COL_MAP['nap']] * SM_COST_RATIO['nap'] +
                       SM_COST_RATIO['fixed'])
    df['_sm_margin'] = df[COL_MAP['sm_cn']] - df['_sm_cost']

    # SM Cost 이론 기준선 (BZ×0.80 + ET×0.30 + 150)
    df['_sm_cost_theory']    = (df[COL_MAP['bz']] * SM_THEORY_RATIO['bz'] +
                                df[COL_MAP['et']] * SM_THEORY_RATIO['et'] +
                                SM_THEORY_RATIO['fixed'])
    df['_sm_margin_theory']  = df[COL_MAP['sm_cn']] - df['_sm_cost_theory']
    # 수급 이상 신호: 실측 원가 - 이론 원가 (양수=수급 타이트)
    df['_sm_cost_gap_signal'] = df['_sm_cost'] - df['_sm_cost_theory']

    # BZ 글로벌 스프레드
    df['_bz_spread_ara'] = df[COL_MAP['bz']] - df[COL_MAP['bz_ara']]
    df['_bz_usg_mt']     = df[COL_MAP['bz_usg']] * BZ_USG_TO_MT
    df['_bz_spread_usg'] = df[COL_MAP['bz']] - df['_bz_usg_mt']

    latest = df.iloc[-1]
    hist8  = df.tail(8).copy()

    print(f"[구글시트] {len(df)}주 데이터 | 최신: {latest[date_col].strftime('%Y-%m-%d')}")
    print(f"  WTI={latest[COL_MAP['wti']]:.2f} | NAP={latest[COL_MAP['nap']]:.0f} | "
          f"BZ={latest[COL_MAP['bz']]:.0f} | ET={latest[COL_MAP['et']]:.0f} | "
          f"SM={latest[COL_MAP['sm_cn']]:.0f} | AN={latest[COL_MAP['an']]:.0f} | "
          f"BD={latest[COL_MAP['bd']]:.0f} | PR={latest[COL_MAP['pr']]:.0f}")
    print(f"  SM Cost 실측={latest['_sm_cost']:.0f} | 이론={latest['_sm_cost_theory']:.0f} | "
          f"수급신호={latest['_sm_cost_gap_signal']:+.0f} | ABS Gap={latest['_abs_gap']:.0f}")
    return latest, df, hist8


# ──────────────────────────────────────────────
# 3. 자동 회귀계수
# ──────────────────────────────────────────────
def calc_regression(df):
    global NAP_SENS_FOR_EQUIV
    wti_col = COL_MAP['wti']
    raw_targets = {
        'bz': COL_MAP['bz'], 'et': COL_MAP['et'], 'sm': COL_MAP['sm_cn'],
        'an': COL_MAP['an'], 'bd': COL_MAP['bd'], 'nap': COL_MAP['nap'],
        'pr': COL_MAP['pr'], 'bz_ara': COL_MAP['bz_ara'], 'bz_usg': COL_MAP['bz_usg'],
    }
    derived_targets = {
        'sm_cost': '_sm_cost', 'sm_margin': '_sm_margin',
        'abs_cost': '_abs_cost', 'abs_mkt': COL_MAP['abs_mkt'], 'abs_gap': '_abs_gap',
    }

    sens = dict(DEFAULT_SENS)
    r2   = {k: None for k in sens}
    n    = 0
    R2_THRESHOLD = 0.5

    print("\n[ 자동 회귀계수 v6.2 ]")
    for key, col in {**raw_targets, **derived_targets}.items():
        if col not in df.columns:
            print(f"  {key:14s}: 컬럼 없음 → 기본값 {DEFAULT_SENS.get(key, 'N/A')}")
            continue
        valid = df[[wti_col, col]].dropna()
        n_pts = len(valid)
        if n_pts < 6:
            print(f"  {key:14s}: 데이터 부족 → 기본값")
            continue
        x = valid[wti_col].values
        y = valid[col].values
        m, b = np.polyfit(x, y, 1)
        yp   = m * x + b
        ss_r = np.sum((y - yp) ** 2)
        ss_t = np.sum((y - np.mean(y)) ** 2)
        r2v  = round(1 - ss_r / ss_t, 3) if ss_t > 0 else 0
        r2[key] = r2v
        if r2v >= R2_THRESHOLD:
            sens[key] = round(m, 3)
            flag = '✓ 자동갱신'
        else:
            flag = f'→ 기본값 {DEFAULT_SENS.get(key, m):+.2f} 유지'
        print(f"  {key:14s}: m={m:+7.2f}  R²={r2v:.3f}  n={n_pts}  {flag}")
        n = max(n, n_pts)

    # ★ NAP 민감도 갱신 → WTI 환산 기준 동기화
    NAP_SENS_FOR_EQUIV = sens['nap']
    print(f"\n  ★ v6.2 WTI 등가 환산 기준: NAP sens={NAP_SENS_FOR_EQUIV:.2f}")
    print(f"    리스크 $100 → WTI 등가 +${100/NAP_SENS_FOR_EQUIV:.1f}/bbl → 전 원료 전파")
    return sens, r2, n


# ──────────────────────────────────────────────
# 4. WTI 실시간
# ──────────────────────────────────────────────
def get_wti(fallback=67.02):
    try:
        import yfinance as yf
        h = yf.Ticker("CL=F").history(period="2d")
        if h.empty: raise ValueError("빈 데이터")
        wti = float(h['Close'].dropna().iloc[-1])
        if not (20 <= wti <= 200): raise ValueError(f"비정상값: {wti}")
        print(f"[WTI] 실시간 ${wti:.2f}")
        return wti, "야후파이낸스(실시간)"
    except Exception as e:
        print(f"[WTI] 폴백 ${fallback:.2f} ({e})")
        return fallback, f"폴백 ${fallback:.2f}"


# ──────────────────────────────────────────────
# 5. 원가 계산 v6.2 ★ 핵심
# ──────────────────────────────────────────────
def calc_costs(latest, wti_rt, sens, risk_premium=0):
    """
    ★ v6.2 핵심:
    리스크 프리미엄 → WTI 등가(wti_equiv) 환산 후 전 원료 회귀에 반영

    total_wti_delta = (wti_rt - wti_gs)          ← 실시간 WTI 변동
                    + (risk_premium / NAP_SENS)   ← ★ 리스크 WTI 등가

    각 원료 = 실측값 + total_wti_delta × 회귀민감도

    ABS Gap = ABS Market - ABS Cost (원료 상승 반영)
    → 리스크↑ = 원가도 함께 상승 = Gap 개선 착시 제거
    → 현실: 수요 없는 지정학 리스크 → Gap 악화 수학적 반영
    """
    wti_gs         = float(latest[COL_MAP['wti']])
    bz_act         = float(latest[COL_MAP['bz']])
    et_act         = float(latest[COL_MAP['et']])
    nap_act        = float(latest[COL_MAP['nap']])
    sm_act         = float(latest[COL_MAP['sm_cn']])
    bd_act         = float(latest[COL_MAP['bd']])
    an_act         = float(latest[COL_MAP['an']])
    pr_act         = float(latest[COL_MAP['pr']])
    abs_mkt_act    = float(latest[COL_MAP['abs_mkt']])
    abs_cost_act   = float(latest['_abs_cost'])
    abs_gap_act    = float(latest['_abs_gap'])
    sm_cost_act    = float(latest['_sm_cost'])
    sm_margin_act  = float(latest['_sm_margin'])
    sm_cost_th_act = float(latest['_sm_cost_theory'])
    sm_marg_th_act = float(latest['_sm_margin_theory'])
    sm_sig_act     = float(latest['_sm_cost_gap_signal'])

    try:    bz_ara_act    = float(latest[COL_MAP['bz_ara']])
    except: bz_ara_act    = float('nan')
    try:    bz_usg_mt_act = float(latest['_bz_usg_mt'])
    except: bz_usg_mt_act = float('nan')

    # ── WTI 실시간 델타
    d_wti = wti_rt - wti_gs

    # ── ★ v6.2 리스크 프리미엄 → WTI 등가 환산
    # NAP은 WTI 거의 100% 연동(R²=0.880) → 환산 기준
    # $1 리스크 프리미엄 = NAP_SENS 분의 1 WTI bbl 상승과 동등
    wti_risk_equiv = risk_premium / NAP_SENS_FOR_EQUIV

    # ── ★ 전 원료에 적용되는 총 WTI 델타
    d_total = d_wti + wti_risk_equiv

    # ── 전 원료 보정 (실측 + 총 WTI 델타 × 회귀민감도)
    bz_adj     = round(bz_act   + d_total * sens['bz'],  1)
    et_adj     = round(et_act   + d_total * sens['et'],  1)
    nap_adj    = round(nap_act  + d_total * sens['nap'], 1)
    sm_adj     = round(sm_act   + d_total * sens['sm'],  1)
    bd_adj     = round(bd_act   + d_total * sens['bd'],  1)
    an_adj     = round(an_act   + d_total * sens['an'],  1)
    pr_adj     = round(pr_act   + d_total * sens['pr'],  1)
    bz_ara_adj = round(bz_ara_act    + d_total * sens.get('bz_ara', DEFAULT_SENS['bz_ara']), 1) \
                 if not np.isnan(bz_ara_act)    else float('nan')
    bz_usg_adj = round(bz_usg_mt_act + d_total * sens.get('bz_usg', DEFAULT_SENS['bz_usg']) * BZ_USG_TO_MT, 1) \
                 if not np.isnan(bz_usg_mt_act) else float('nan')

    # ── SM Cost 실측 회귀 기반 (원료 상승 반영)
    sm_cost_adj   = round(sm_cost_act   + d_total * sens['sm_cost'],   1)
    # SM Margin: WTI 무상관(R²≈0) → 실시간 WTI 델타만 적용 (리스크 등가 제외)
    sm_margin_adj = round(sm_margin_act + d_wti   * sens['sm_margin'], 1)

    # ── SM Cost 이론 기준선 (BZ/ET 보정값 직접 계산)
    sm_cost_th_adj = round(bz_adj * SM_THEORY_RATIO['bz'] +
                           et_adj * SM_THEORY_RATIO['et'] +
                           SM_THEORY_RATIO['fixed'], 1)
    sm_marg_th_adj = round(sm_adj - sm_cost_th_adj, 1)
    # 수급 이상 신호: 실측 원가 - 이론 원가 (양수=수급 타이트)
    sm_sig_adj     = round(sm_cost_adj - sm_cost_th_adj, 1)

    # ── ABS Cost (원료 상승 반영)
    abs_cost_adj = round(abs_cost_act + d_total * sens['abs_cost'], 1)

    # ── ABS Market: 실시간 WTI 델타(회귀) + 리스크 프리미엄(flat 가산)
    # 리스크는 원료에 wti_equiv로 이미 전파, Market에는 그대로 가산
    abs_mkt_adj = round(abs_mkt_act + d_wti * sens['abs_mkt'] + risk_premium, 1)

    # ── ABS Gap = Market - Cost (원가 상승 반영 → 착시 제거)
    abs_gap_adj = round(abs_mkt_adj - abs_cost_adj, 1)

    bz_spread_ara = round(bz_adj - bz_ara_adj, 1) if not np.isnan(bz_ara_adj) else float('nan')
    bz_spread_usg = round(bz_adj - bz_usg_adj,  1) if not np.isnan(bz_usg_adj) else float('nan')

    return {
        'WTI_RT':                    round(wti_rt, 2),
        'WTI_GS':                    round(wti_gs, 2),
        'WTI_Delta':                 round(d_wti, 2),
        'WTI_Risk_Equiv':            round(wti_risk_equiv, 2),
        'WTI_Total_Delta':           round(d_total, 2),
        'Risk_Premium':              risk_premium,
        'NAP':                       nap_adj,
        'BZ':                        bz_adj,     'BZ_Actual':       round(bz_act, 1),
        'ET':                        et_adj,     'ET_Actual':       round(et_act, 1),
        'SM_Market':                 sm_adj,     'SM_Actual':       round(sm_act, 1),
        'BD':                        bd_adj,     'BD_Actual':       round(bd_act, 1),
        'AN':                        an_adj,     'AN_Actual':       round(an_act, 1),
        'PR':                        pr_adj,     'PR_Actual':       round(pr_act, 1),
        'BZ_ARA':                    bz_ara_adj, 'BZ_ARA_Actual':   round(bz_ara_act, 1),
        'BZ_USG_MT':                 bz_usg_adj, 'BZ_USG_Actual':   round(bz_usg_mt_act, 1),
        'BZ_Spread_ARA':             bz_spread_ara,
        'BZ_Spread_USG':             bz_spread_usg,
        'SM_Cost':                   sm_cost_adj,   'SM_Cost_Actual':         round(sm_cost_act, 1),
        'SM_Margin':                 sm_margin_adj, 'SM_Margin_Actual':       round(sm_margin_act, 1),
        'SM_Cost_Theory':            sm_cost_th_adj, 'SM_Cost_Theory_Actual': round(sm_cost_th_act, 1),
        'SM_Margin_Theory':          sm_marg_th_adj, 'SM_Margin_Theory_Actual': round(sm_marg_th_act, 1),
        'SM_Cost_Gap_Signal':        sm_sig_adj,  'SM_Cost_Gap_Signal_Actual': round(sm_sig_act, 1),
        'ABS_Market':                abs_mkt_adj,   'ABS_Mkt_Actual':  round(abs_mkt_act, 1),
        'ABS_Cost':                  abs_cost_adj,  'ABS_Cost_Actual': round(abs_cost_act, 1),
        'ABS_Gap':                   abs_gap_adj,   'ABS_Gap_Actual':  round(abs_gap_act, 1),
    }


# ──────────────────────────────────────────────
# 6. 시나리오
# ──────────────────────────────────────────────
SCENARIOS = [
    {'label': 'Base\n($59)',      'wti':  59.44, 'risk':   0, 'color': '#2ecc71'},
    {'label': 'Mild\n($70)',      'wti':  70.00, 'risk':  50, 'color': '#f39c12'},
    {'label': 'Moderate\n($80)', 'wti':  80.00, 'risk': 100, 'color': '#e67e22'},
    {'label': 'Severe\n($90)',   'wti':  90.00, 'risk': 150, 'color': '#e74c3c'},
    {'label': 'Crisis\n($100)',  'wti': 100.00, 'risk': 200, 'color': '#c0392b'},
]


def calc_scenario(latest, sens, wti_target, risk_premium):
    """★ v6.2: risk_premium → calc_costs에 직접 전달 → 전 원료 WTI 등가 반영"""
    return calc_costs(latest, wti_target, sens, risk_premium=risk_premium)


# ──────────────────────────────────────────────
# 7. 차트 (9패널 3×3) v6.2
# ──────────────────────────────────────────────
def generate_report(current, hist8, latest, sens, r2, n_reg, wti_source):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 18), facecolor='#0f172a')

    date_col = hist8.columns[0]
    gs_date  = pd.to_datetime(latest[date_col]).strftime('%Y-%m-%d')
    fig.suptitle(
        f'IRAN CONFLICT RISK DASHBOARD  v6.2  |  WTI ${current["WTI_RT"]:.2f}/bbl  |  '
        f'실측앵커: {gs_date}  |  리스크→WTI등가 전 원료 반영  |  실측vs이론 수급신호  |  '
        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        fontsize=10, fontweight='bold', color='#fbbf24', y=0.99
    )

    dates          = [pd.to_datetime(d).strftime('%m/%d') for d in hist8[date_col]]
    x              = range(len(dates))
    abs_gap_h      = hist8['_abs_gap'].tolist()
    abs_cost_h     = hist8['_abs_cost'].tolist()
    abs_h          = hist8[COL_MAP['abs_mkt']].tolist()
    sm_margin_h    = hist8['_sm_margin'].tolist()
    sm_margin_th_h = hist8['_sm_margin_theory'].tolist()
    sm_cost_h      = hist8['_sm_cost'].tolist()
    sm_cost_th_h   = hist8['_sm_cost_theory'].tolist()
    sm_cost_sig_h  = hist8['_sm_cost_gap_signal'].tolist()
    sm_h           = hist8[COL_MAP['sm_cn']].tolist()
    wti_h          = hist8[COL_MAP['wti']].tolist()
    bz_h           = hist8[COL_MAP['bz']].tolist()
    et_h           = hist8[COL_MAP['et']].tolist()
    an_h           = hist8[COL_MAP['an']].tolist()
    bd_h           = hist8[COL_MAP['bd']].tolist()
    pr_h           = hist8[COL_MAP['pr']].tolist()
    bz_ara_h       = hist8[COL_MAP['bz_ara']].tolist()
    bz_usg_mt_h    = hist8['_bz_usg_mt'].tolist()

    def c(lst):
        return [v if (v and not (isinstance(v, float) and np.isnan(v))) else float('nan') for v in lst]

    def border(ax):
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

    # ① ABS Gap 8주
    ax1 = fig.add_subplot(3, 3, 1); ax1.set_facecolor('#1e293b')
    g_colors = ['#10b981' if g >= 150 else '#f59e0b' if g >= 0 else '#ef4444' for g in abs_gap_h]
    ax1.bar(list(x), abs_gap_h, color=g_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for i, g in enumerate(abs_gap_h):
        if not np.isnan(g):
            ax1.text(i, g + (8 if g >= 0 else -22), f'${g:.0f}', ha='center', fontsize=7, color='white', fontweight='bold')
    ax1.axhline(y=150, color='#fbbf24', linestyle='--', linewidth=1.5, label='경보선 $150')
    ax1.axhline(y=0,   color='#ef4444', linestyle='-',  linewidth=1,   alpha=0.5)
    gc = '#ef4444' if current['ABS_Gap'] < 0 else '#f59e0b' if current['ABS_Gap'] < 150 else '#10b981'
    ax1.set_title(f'① ABS Gap 8주 | 실측 ${current["ABS_Gap_Actual"]:+.0f} → v6.2 ${current["ABS_Gap"]:+.0f}/t',
                  color=gc, fontweight='bold', fontsize=9)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax1.set_ylabel('$/mt', color='#94a3b8'); ax1.tick_params(colors='#94a3b8')
    ax1.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155'); border(ax1)

    # ② ABS Market vs Cost
    ax2 = fig.add_subplot(3, 3, 2); ax2.set_facecolor('#1e293b')
    ax2.fill_between(x, abs_cost_h, c(abs_h), alpha=0.15,
                     where=[a > b for a, b in zip(c(abs_h), abs_cost_h)], color='#10b981')
    ax2.fill_between(x, abs_cost_h, c(abs_h), alpha=0.15,
                     where=[a <= b for a, b in zip(c(abs_h), abs_cost_h)], color='#ef4444')
    ax2.plot(x, c(abs_h),   color='#3b82f6', linewidth=2, marker='o', markersize=5, label='ABS Market')
    ax2.plot(x, abs_cost_h, color='#ef4444', linewidth=2, marker='s', markersize=4, label='ABS Cost')
    ax2.set_title(f'② ABS Market vs Cost | Gap ${current["ABS_Gap"]:+.0f}/t',
                  color='#10b981', fontweight='bold', fontsize=9)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax2.set_ylabel('$/mt', color='#94a3b8'); ax2.tick_params(colors='#94a3b8')
    ax2.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155'); border(ax2)

    # ③ SM 실측 vs 이론 마진 (수급신호) ★ v6.2
    ax3 = fig.add_subplot(3, 3, 3); ax3.set_facecolor('#1e293b')
    sm_cols   = ['#10b981' if m >= 0 else '#ef4444' for m in sm_margin_h]
    th_cols   = ['#3b82f6' if m >= 0 else '#a855f7' for m in sm_margin_th_h]
    ax3.bar([i - 0.2 for i in x], sm_margin_h,    width=0.35, color=sm_cols, alpha=0.80, label='SM Margin 실측')
    ax3.bar([i + 0.2 for i in x], sm_margin_th_h, width=0.35, color=th_cols, alpha=0.50, label='SM Margin 이론(BZ×0.80)')
    ax3.axhline(y=0, color='white', linewidth=1, alpha=0.4)
    ax3r = ax3.twinx()
    ax3r.plot(x, c(sm_cost_sig_h), color='#fbbf24', linewidth=1.5, linestyle=':', marker='D', markersize=4, label='수급신호(실측-이론)')
    ax3r.axhline(y=0, color='#fbbf24', linewidth=1, alpha=0.3)
    ax3r.set_ylabel('수급신호 $/mt', color='#fbbf24', fontsize=7)
    ax3r.tick_params(axis='y', colors='#fbbf24')
    sig = current['SM_Cost_Gap_Signal']
    sig_str = f'+${sig:.0f} 수급타이트' if sig > 0 else f'${sig:.0f} 수급여유'
    ax3.set_title(f'③ SM 실측 vs 이론마진 | 수급신호: {sig_str}',
                  color='#10b981', fontweight='bold', fontsize=8)
    ax3.set_xticks(list(x)); ax3.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax3.set_ylabel('$/mt', color='#94a3b8'); ax3.tick_params(colors='#94a3b8')
    l1, lb1 = ax3.get_legend_handles_labels(); l2, lb2 = ax3r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, fontsize=6, facecolor='#1e293b', edgecolor='#334155')
    border(ax3)
    for sp in ax3r.spines.values(): sp.set_edgecolor('#334155')

    # ④ SM Market vs Cost (실측/이론 병기) ★ v6.2
    ax4 = fig.add_subplot(3, 3, 4); ax4.set_facecolor('#1e293b')
    ax4.fill_between(x, sm_cost_h, sm_h, alpha=0.12,
                     where=[a > b for a, b in zip(sm_h, sm_cost_h)], color='#3b82f6')
    ax4.fill_between(x, sm_cost_h, sm_h, alpha=0.12,
                     where=[a <= b for a, b in zip(sm_h, sm_cost_h)], color='#ef4444')
    ax4.plot(x, sm_h,            color='#3b82f6', linewidth=2, marker='o', markersize=4, label='SM CFR China')
    ax4.plot(x, sm_cost_h,       color='#ef4444', linewidth=2, marker='s', markersize=3, label='SM Cost 실측')
    ax4.plot(x, c(sm_cost_th_h), color='#fbbf24', linewidth=1.5, linestyle='--', marker='^', markersize=3, label='SM Cost 이론')
    ax4.set_title(f'④ SM Market vs Cost | 실측 ${current["SM_Cost"]:.0f} | 이론 ${current["SM_Cost_Theory"]:.0f}',
                  color='#3b82f6', fontweight='bold', fontsize=8)
    ax4.set_xticks(list(x)); ax4.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax4.set_ylabel('$/mt', color='#94a3b8'); ax4.tick_params(colors='#94a3b8')
    ax4.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155'); border(ax4)

    # ⑤ Raw Material Trend
    ax5 = fig.add_subplot(3, 3, 5); ax5.set_facecolor('#1e293b')
    ax5r = ax5.twinx()
    ax5r.plot(x, wti_h, color='#fbbf24', linewidth=2, marker='^', markersize=4, linestyle='--', label='WTI(R)')
    ax5r.set_ylabel('WTI $/bbl', color='#fbbf24', fontsize=8)
    ax5r.tick_params(axis='y', colors='#fbbf24'); ax5r.set_ylim(40, 110)
    ax5.plot(x, bz_h,    color='#a855f7', linewidth=2, marker='o', markersize=4, label='BZ')
    ax5.plot(x, et_h,    color='#10b981', linewidth=2, marker='s', markersize=3, label='ET')
    ax5.plot(x, c(an_h), color='#f59e0b', linewidth=2, marker='D', markersize=3, label='AN')
    ax5.plot(x, c(bd_h), color='#f97316', linewidth=2, marker='v', markersize=3, label='BD')
    ax5.set_title('⑤ Raw Material Trend (8W)', color='#94a3b8', fontweight='bold', fontsize=9)
    ax5.set_xticks(list(x)); ax5.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax5.set_ylabel('$/mt', color='#94a3b8'); ax5.tick_params(axis='y', colors='#94a3b8')
    ax5.set_ylim(400, 1700)
    l1, lb1 = ax5.get_legend_handles_labels(); l2, lb2 = ax5r.get_legend_handles_labels()
    ax5.legend(l1 + l2, lb1 + lb2, fontsize=6, facecolor='#1e293b', edgecolor='#334155')
    border(ax5)
    for sp in ax5r.spines.values(): sp.set_edgecolor('#334155')

    # ⑥ Iran Risk Scenario ABS Gap ★ v6.2 원가 상승 반영
    ax6 = fig.add_subplot(3, 3, 6); ax6.set_facecolor('#1e293b')
    sc_labels, sc_gaps, sc_costs_s, sc_mkts_s, sc_wti_equiv = [], [], [], [], []
    for s in SCENARIOS:
        r = calc_scenario(latest, sens, s['wti'], s['risk'])
        sc_labels.append(s['label']); sc_gaps.append(r['ABS_Gap'])
        sc_costs_s.append(r['ABS_Cost']); sc_mkts_s.append(r['ABS_Market'])
        sc_wti_equiv.append(r['WTI_Risk_Equiv'])
    xp = np.arange(len(SCENARIOS)); w = 0.35
    ax6.bar(xp - w / 2, sc_costs_s, w, label='ABS Cost(원료상승포함)', color='#ef4444', alpha=0.85)
    ax6.bar(xp + w / 2, sc_mkts_s,  w, label='ABS Market(+리스크)',    color='#3b82f6', alpha=0.85)
    for i, (g, c_, m_, we) in enumerate(zip(sc_gaps, sc_costs_s, sc_mkts_s, sc_wti_equiv)):
        gc = '#10b981' if g >= 0 else '#ef4444'
        ax6.text(i, max(c_, m_) + 12, f'${g:+.0f}', ha='center', fontsize=8, color=gc, fontweight='bold')
        ax6.text(i, min(c_, m_) - 32, f'≡+${we:.1f}WTI', ha='center', fontsize=6, color='#fbbf24', alpha=0.8)
    ax6.axhline(y=0, color='#ef4444', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_title('⑥ Iran Scenario ABS Gap\n(리스크→WTI등가 전 원료 반영 v6.2)',
                  color='#fbbf24', fontweight='bold', fontsize=8)
    ax6.set_xticks(xp); ax6.set_xticklabels(sc_labels, fontsize=7, color='white')
    ax6.set_ylabel('$/mt', color='#94a3b8'); ax6.tick_params(colors='#94a3b8')
    ax6.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155')
    all_vals = sc_costs_s + sc_mkts_s
    ax6.set_ylim(min(min(sc_gaps) - 80, 0), max(all_vals) * 1.28)
    border(ax6)

    # ⑦ PR 트렌드
    ax7 = fig.add_subplot(3, 3, 7); ax7.set_facecolor('#1e293b')
    ax7r = ax7.twinx()
    ax7r.plot(x, wti_h, color='#fbbf24', linewidth=1.5, linestyle='--', marker='^', markersize=4, label='WTI(R)')
    ax7r.set_ylabel('WTI $/bbl', color='#fbbf24', fontsize=8)
    ax7r.tick_params(axis='y', colors='#fbbf24'); ax7r.set_ylim(40, 110)
    ax7.plot(x, c(pr_h), color='#06b6d4', linewidth=2, marker='o', markersize=5, label='PR CFR China')
    for i, p in enumerate(pr_h):
        if not np.isnan(p):
            ax7.text(i, p + 10, f'${p:.0f}', ha='center', fontsize=7, color='#06b6d4')
    ax7.set_title(f'⑦ PR(Propylene) 8W | ${current["PR_Actual"]:.0f} → ${current["PR"]:.0f}/t',
                  color='#06b6d4', fontweight='bold', fontsize=8)
    ax7.set_xticks(list(x)); ax7.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax7.set_ylabel('PR $/mt', color='#94a3b8'); ax7.tick_params(axis='y', colors='#94a3b8')
    l1, lb1 = ax7.get_legend_handles_labels(); l2, lb2 = ax7r.get_legend_handles_labels()
    ax7.legend(l1 + l2, lb1 + lb2, fontsize=7, facecolor='#1e293b', edgecolor='#334155')
    border(ax7)
    for sp in ax7r.spines.values(): sp.set_edgecolor('#334155')

    # ⑧ BZ 글로벌 스프레드
    ax8 = fig.add_subplot(3, 3, 8); ax8.set_facecolor('#1e293b')
    ax8.plot(x, bz_h,           color='#a855f7', linewidth=2, marker='o', markersize=5, label='BZ FOB Korea')
    ax8.plot(x, c(bz_ara_h),    color='#3b82f6', linewidth=2, marker='s', markersize=4, label='BZ CIF ARA')
    ax8.plot(x, c(bz_usg_mt_h), color='#f59e0b', linewidth=2, marker='D', markersize=4, label='BZ USG($/mt)')
    spread_all = [bz_h[i] - c(bz_ara_h)[i] if not np.isnan(c(bz_ara_h)[i]) else float('nan') for i in range(len(x))]
    ax8t = ax8.twinx()
    ax8t.bar(list(x), spread_all, alpha=0.25, color='#e879f9', label='Korea-ARA Spread')
    ax8t.axhline(y=0, color='#e879f9', linestyle='--', linewidth=1, alpha=0.4)
    ax8t.set_ylabel('Spread $/mt', color='#e879f9', fontsize=8)
    ax8t.tick_params(axis='y', colors='#e879f9')
    spread_now = current.get('BZ_Spread_ARA', float('nan'))
    spread_str = f'${spread_now:.0f}' if not np.isnan(spread_now) else 'N/A'
    ax8.set_title(f'⑧ BZ 글로벌 스프레드 | Korea-ARA={spread_str}',
                  color='#a855f7', fontweight='bold', fontsize=8)
    ax8.set_xticks(list(x)); ax8.set_xticklabels(dates, fontsize=7, color='#94a3b8', rotation=30)
    ax8.set_ylabel('$/mt', color='#94a3b8'); ax8.tick_params(axis='y', colors='#94a3b8')
    l1, lb1 = ax8.get_legend_handles_labels(); l2, lb2 = ax8t.get_legend_handles_labels()
    ax8.legend(l1 + l2, lb1 + lb2, fontsize=6, facecolor='#1e293b', edgecolor='#334155')
    border(ax8)
    for sp in ax8t.spines.values(): sp.set_edgecolor('#334155')

    # ⑨ 전 품목 WTI 민감도 비교
    ax9 = fig.add_subplot(3, 3, 9); ax9.set_facecolor('#1e293b')
    sens_items = [
        ('NAP(기준)', sens['nap'],                                    '#94a3b8'),
        ('BZ',        sens['bz'],                                     '#a855f7'),
        ('ET',        sens['et'],                                     '#10b981'),
        ('SM',        sens['sm'],                                     '#3b82f6'),
        ('AN',        sens['an'],                                     '#f59e0b'),
        ('BD',        sens['bd'],                                     '#f97316'),
        ('PR',        sens['pr'],                                     '#06b6d4'),
        ('BZ_ARA',    sens.get('bz_ara', DEFAULT_SENS['bz_ara']),    '#818cf8'),
        ('ABS_Cost',  sens['abs_cost'],                               '#ef4444'),
        ('ABS_Mkt',   sens['abs_mkt'],                                '#60a5fa'),
        ('ABS_Gap',   sens['abs_gap'],                                '#34d399'),
        ('SM_Cost',   sens['sm_cost'],                                '#fb7185'),
    ]
    labels_s = [i[0] for i in sens_items]
    values_s = [i[1] for i in sens_items]
    colors_s = [i[2] for i in sens_items]
    bars = ax9.barh(labels_s, values_s, color=colors_s, alpha=0.85, edgecolor='white', linewidth=0.4)
    for bar, v in zip(bars, values_s):
        ax9.text(v + (0.3 if v >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                 f'{v:+.2f}', va='center', fontsize=7, color='white', fontweight='bold')
    ax9.axvline(x=0, color='white', linewidth=1, alpha=0.5)
    we_100 = round(100 / NAP_SENS_FOR_EQUIV, 1)
    ax9.set_title(f'⑨ WTI $1/bbl → 각 품목 변화\n★ 리스크 $100 ≡ WTI +${we_100}/bbl 전 원료 전파',
                  color='#fbbf24', fontweight='bold', fontsize=8)
    ax9.set_xlabel('$/mt per $1 WTI', color='#94a3b8', fontsize=8)
    ax9.tick_params(colors='#94a3b8'); border(ax9)

    fig.text(0.5, 0.005,
             f'LAM Advanced Procurement  |  v6.2  |  '
             f'리스크 WTI등가=÷NAP{NAP_SENS_FOR_EQUIV:.2f}  |  '
             f'실측원가 vs 이론(BZ×0.80+ET×0.30) 수급신호  |  {wti_source}',
             ha='center', fontsize=7, color='#475569')

    plt.tight_layout(rect=[0, 0.015, 1, 0.98])
    plt.savefig('risk_simulation_report.png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print("[차트] risk_simulation_report.png 저장 완료 (9패널 v6.2)")


# ──────────────────────────────────────────────
# 8. CSV 저장
# ──────────────────────────────────────────────
def save_csv(current, sens, r2, n_reg, wti_source, gs_date):
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    row = {
        'UpdateTime': now, 'WTI_Source': wti_source, 'GSheet_Date': gs_date, 'Reg_N': n_reg,
        'WTI': current['WTI_RT'], 'WTI_GSheet': current['WTI_GS'],
        'WTI_Delta': current['WTI_Delta'],
        'WTI_Risk_Equiv': current['WTI_Risk_Equiv'],
        'WTI_Total_Delta': current['WTI_Total_Delta'],
        'Risk_Premium': current['Risk_Premium'],
        'NAP': current['NAP'],
        'BZ': current['BZ'], 'BZ_Actual': current['BZ_Actual'],
        'BZ_ARA': current.get('BZ_ARA', ''), 'BZ_ARA_Actual': current.get('BZ_ARA_Actual', ''),
        'BZ_USG_MT': current.get('BZ_USG_MT', ''), 'BZ_USG_Actual': current.get('BZ_USG_Actual', ''),
        'BZ_Spread_ARA': current.get('BZ_Spread_ARA', ''),
        'ET': current['ET'], 'ET_Actual': current['ET_Actual'],
        'SM_Market': current['SM_Market'], 'SM_Actual': current['SM_Actual'],
        'PR': current['PR'], 'PR_Actual': current['PR_Actual'],
        'BD': current['BD'], 'BD_Actual': current['BD_Actual'],
        'AN': current['AN'], 'AN_Actual': current['AN_Actual'],
        'SM_Cost': current['SM_Cost'], 'SM_Cost_Actual': current['SM_Cost_Actual'],
        'SM_Margin': current['SM_Margin'], 'SM_Margin_Actual': current['SM_Margin_Actual'],
        'SM_Cost_Theory': current['SM_Cost_Theory'], 'SM_Cost_Theory_Actual': current['SM_Cost_Theory_Actual'],
        'SM_Margin_Theory': current['SM_Margin_Theory'], 'SM_Margin_Theory_Actual': current['SM_Margin_Theory_Actual'],
        'SM_Cost_Gap_Signal': current['SM_Cost_Gap_Signal'],
        'SM_Cost_Gap_Signal_Actual': current['SM_Cost_Gap_Signal_Actual'],
        'ABS_Market': current['ABS_Market'], 'ABS_Mkt_Actual': current['ABS_Mkt_Actual'],
        'ABS_Cost': current['ABS_Cost'], 'ABS_Cost_Actual': current['ABS_Cost_Actual'],
        'ABS_Gap': current['ABS_Gap'], 'ABS_Gap_Actual': current['ABS_Gap_Actual'],
        'Sens_BZ': sens['bz'], 'R2_BZ': r2.get('bz', ''),
        'Sens_ET': sens['et'], 'R2_ET': r2.get('et', ''),
        'Sens_SM': sens['sm'], 'R2_SM': r2.get('sm', ''),
        'Sens_AN': sens['an'], 'R2_AN': r2.get('an', ''),
        'Sens_BD': sens['bd'], 'R2_BD': r2.get('bd', ''),
        'Sens_NAP': sens['nap'], 'R2_NAP': r2.get('nap', ''),
        'Sens_PR': sens['pr'], 'R2_PR': r2.get('pr', ''),
        'Sens_BZ_ARA': sens.get('bz_ara', DEFAULT_SENS['bz_ara']), 'R2_BZ_ARA': r2.get('bz_ara', ''),
        'Sens_BZ_USG': sens.get('bz_usg', DEFAULT_SENS['bz_usg']), 'R2_BZ_USG': r2.get('bz_usg', ''),
        'Sens_ABS_MKT': sens['abs_mkt'], 'R2_ABS_MKT': r2.get('abs_mkt', ''),
        'Sens_ABS_GAP': sens['abs_gap'], 'R2_ABS_GAP': r2.get('abs_gap', ''),
        'Sens_ABS_COST': sens['abs_cost'], 'R2_ABS_COST': r2.get('abs_cost', ''),
        'Sens_SM_MARGIN': sens['sm_margin'], 'R2_SM_MARGIN': r2.get('sm_margin', ''),
        'Sens_SM_COST': sens['sm_cost'], 'R2_SM_COST': r2.get('sm_cost', ''),
        'NAP_Sens_Equiv_Basis': NAP_SENS_FOR_EQUIV,
        'Margin': current['SM_Margin'],
        'ABS_Landed': current['ABS_Market'],
    }
    pd.DataFrame([row]).to_csv('simulation_result.csv', index=False)
    print("[CSV] simulation_result.csv 저장 완료 (v6.2)")


# ──────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Iran Risk × ABS/SM 원가 시뮬레이션 v6.2")
    print("리스크→WTI등가 전 원료 반영 | 실측vs이론 수급신호")
    print("=" * 65)

    setup_font()
    latest, df_all, hist8 = load_gsheet()
    if latest is None:
        print("구글시트 로드 실패"); exit(1)

    gs_date = pd.to_datetime(latest[hist8.columns[0]]).strftime('%Y-%m-%d')
    sens, r2, n_reg = calc_regression(df_all)
    wti_rt, wti_src = get_wti(fallback=float(latest[COL_MAP['wti']]))
    current = calc_costs(latest, wti_rt, sens, risk_premium=0)

    print(f"\n{'─' * 65}")
    print(f"  WTI          : ${current['WTI_RT']:.2f} (GS ${current['WTI_GS']:.2f}, Δ{current['WTI_Delta']:+.2f})")
    print(f"  NAP(환산기준): ${current['NAP']:.0f}/t  (sens={NAP_SENS_FOR_EQUIV:.2f})")
    print(f"  BZ Korea     : ${current['BZ_Actual']:.0f} → ${current['BZ']:.0f}")
    print(f"  ET           : ${current['ET_Actual']:.0f} → ${current['ET']:.0f}")
    print(f"  SM           : ${current['SM_Actual']:.0f} → ${current['SM_Market']:.0f}")
    print(f"  PR           : ${current['PR_Actual']:.0f} → ${current['PR']:.0f}")
    print(f"  AN           : ${current['AN_Actual']:.0f} → ${current['AN']:.0f}")
    print(f"  BD           : ${current['BD_Actual']:.0f} → ${current['BD']:.0f}")
    print(f"  {'─' * 61}")
    print(f"  SM Cost 실측 : ${current['SM_Cost_Actual']:.0f} → ${current['SM_Cost']:.0f}/t")
    print(f"  SM Cost 이론 : ${current['SM_Cost_Theory_Actual']:.0f} → ${current['SM_Cost_Theory']:.0f}/t (BZ×0.80+ET×0.30+150)")
    sig = current['SM_Cost_Gap_Signal']
    print(f"  수급신호 ★   : {sig:+.0f}/t  ({'수급타이트' if sig > 0 else '수급여유'})")
    print(f"  SM Margin 실측: ${current['SM_Margin_Actual']:+.0f} → ${current['SM_Margin']:+.0f}/t  [WTI 무상관]")
    print(f"  SM Margin 이론: ${current['SM_Margin_Theory_Actual']:+.0f} → ${current['SM_Margin_Theory']:+.0f}/t")
    print(f"  ABS Cost     : ${current['ABS_Cost_Actual']:.0f} → ${current['ABS_Cost']:.0f}/t")
    print(f"  ABS Market   : ${current['ABS_Mkt_Actual']:.0f} → ${current['ABS_Market']:.0f}/t")
    print(f"  ABS Gap ★    : ${current['ABS_Gap_Actual']:+.0f} → ${current['ABS_Gap']:+.0f}/t  "
          f"{'⚠경보!' if current['ABS_Gap'] < 150 else '✓'}")
    print(f"{'─' * 65}\n")

    print("[ 이란 리스크 시나리오 v6.2 - 전 원료 WTI 등가 반영 ]")
    print(f"  {'시나리오':20s} | WTI     | 리스크 | WTI등가     | ABS원가  | ABS Gap   | 수급신호")
    print(f"  {'─' * 80}")
    for s in SCENARIOS:
        r    = calc_scenario(latest, sens, s['wti'], s['risk'])
        flag = '🔴역마진' if r['ABS_Gap'] < 0 else ('⚠경보' if r['ABS_Gap'] < 150 else '✓')
        print(f"  {s['label'].replace(chr(10), ' '):20s} | "
              f"${s['wti']:5.0f}   | +${s['risk']:3.0f}  | "
              f"+${r['WTI_Risk_Equiv']:5.1f}/bbl  | "
              f"${r['ABS_Cost']:.0f}  | "
              f"${r['ABS_Gap']:+.0f}/t {flag} | "
              f"{r['SM_Cost_Gap_Signal']:+.0f}/t")

    generate_report(current, hist8, latest, sens, r2, n_reg, wti_src)
    save_csv(current, sens, r2, n_reg, wti_src, gs_date)
    print("\n시뮬레이션 완료 v6.2")
    print("=" * 65)
