# ============================================================
# pip install pandas matplotlib numpy yfinance requests
# ============================================================
# simulation.py - Iran Risk × ABS/SM 원가 대응 시뮬레이션
# v5.0 - Gap 직접 회귀 + 전품목 실측 기반 민감도 자동계산
# ============================================================
# 핵심 변경:
#   ABS Gap 민감도를 직접 회귀 → WTI↑ = Gap↓ 현실 반영
#   ABS Market도 역상관 반영 (-$4/bbl)
#   SM Margin R²=0.057 → 거의 무상관 (공급/수요 독립 변수)
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

# 구글시트 컬럼 헤더 매핑
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

# ABS 배합비 (실측)
ABS_RATIO = {'sm': 0.60, 'an': 0.25, 'bd': 0.15}

# 기본 민감도 (데이터 부족 시 폴백 - 9주 실측 기반)
DEFAULT_SENS = {
    'bz':      11.74,   # R²=0.790 ✓
    'et':      -3.25,   # R²=0.496 역상관
    'sm':       9.74,   # R²=0.529
    'an':      15.66,   # R²=0.878 ✓✓
    'bd':      25.69,   # R²=0.678 ✓
    'nap':      9.94,   # R²=0.947 ✓✓
    'pr':       9.83,   # R²=0.814 ✓
    'sm_cost':  8.25,   # R²=0.806 ✓
    'abs_cost': 13.61,  # R²=0.738 ✓
    'abs_mkt': -4.13,   # R²=0.320 역상관 ★
    'abs_gap': -17.75,  # R²=0.619 ✓ WTI↑=Gap↓ ★★★
    'sm_margin': 1.50,  # R²=0.057 거의 무상관
}

# ──────────────────────────────────────────────
# 1. 한글 폰트 설정
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

    # ABS Cost / Gap / SM Cost / SM Margin 파생 컬럼 생성
    df['_abs_cost']   = df[COL_MAP['sm_cn']]*0.60 + df[COL_MAP['an']]*0.25 + df[COL_MAP['bd']]*0.15
    df['_abs_gap']    = df[COL_MAP['abs_mkt']] - df['_abs_cost']
    df['_sm_cost']    = df[COL_MAP['bz']]*0.78 + df[COL_MAP['et']]*0.28 + 150
    df['_sm_margin']  = df[COL_MAP['sm_cn']] - df['_sm_cost']

    latest = df.iloc[-1]
    hist8  = df.tail(8).copy()

    print(f"[구글시트] {len(df)}주 데이터 | 최신: {latest[date_col].strftime('%Y-%m-%d')}")
    print(f"  WTI={latest[COL_MAP['wti']]:.2f} | BZ={latest[COL_MAP['bz']]:.1f} | "
          f"ET={latest[COL_MAP['et']]:.1f} | SM={latest[COL_MAP['sm_cn']]:.1f} | "
          f"AN={latest[COL_MAP['an']]:.1f} | BD={latest[COL_MAP['bd']]:.1f} | "
          f"ABS={latest[COL_MAP['abs_mkt']]:.1f}")
    print(f"  ABS Cost={latest['_abs_cost']:.0f} | ABS Gap={latest['_abs_gap']:.0f} | "
          f"SM Margin={latest['_sm_margin']:.0f}")
    return latest, df, hist8

# ──────────────────────────────────────────────
# 3. 자동 회귀계수 계산 (전품목 + Gap 직접 회귀)
# ──────────────────────────────────────────────
def calc_regression(df):
    """
    WTI 대비 전품목 + ABS Gap / SM Margin 직접 회귀.
    R² < 0.3 또는 4주 미만 → 기본값(DEFAULT_SENS) 사용.
    """
    wti_col = COL_MAP['wti']

    # 원료가 타깃
    raw_targets = {
        'bz':  COL_MAP['bz'],
        'et':  COL_MAP['et'],
        'sm':  COL_MAP['sm_cn'],
        'an':  COL_MAP['an'],
        'bd':  COL_MAP['bd'],
        'nap': COL_MAP['nap'],
        'pr':  COL_MAP['pr'],
    }
    # 파생 타깃 (계산된 컬럼)
    derived_targets = {
        'sm_cost':   '_sm_cost',
        'abs_cost':  '_abs_cost',
        'abs_mkt':   COL_MAP['abs_mkt'],
        'abs_gap':   '_abs_gap',
        'sm_margin': '_sm_margin',
    }

    sens, r2 = {}, {}
    n = 0

    print("\n[ 자동 회귀계수 계산 ]")
    all_targets = {**raw_targets, **derived_targets}

    for key, col in all_targets.items():
        valid = df[[wti_col, col]].dropna()
        n_pts = len(valid)

        if n_pts < 4:
            sens[key] = DEFAULT_SENS[key]
            r2[key]   = None
            print(f"  {key:12s}: 데이터 부족({n_pts}개) → 기본값 {DEFAULT_SENS[key]:+.2f}")
            continue

        x = valid[wti_col].values
        y = valid[col].values
        m, b = np.polyfit(x, y, 1)

        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_val = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else 0

        if r2_val < 0.3:
            sens[key] = DEFAULT_SENS[key]
            r2[key]   = r2_val
            flag = '(낮음→기본값)'
        else:
            sens[key] = round(m, 3)
            r2[key]   = r2_val
            flag = '✓'

        print(f"  {key:12s}: m={sens[key]:+7.2f}  R²={r2_val:.3f}  n={n_pts} {flag}")
        n = max(n, n_pts)

    return sens, r2, n

# ──────────────────────────────────────────────
# 4. WTI 실시간 취득
# ──────────────────────────────────────────────
def get_wti(fallback=67.02):
    try:
        import yfinance as yf
        h = yf.Ticker("CL=F").history(period="2d")
        if h.empty:
            raise ValueError("빈 데이터")
        wti = float(h['Close'].dropna().iloc[-1])
        if not (20 <= wti <= 200):
            raise ValueError(f"비정상값: {wti}")
        print(f"[WTI] 실시간 ${wti:.2f} (야후파이낸스 CL=F)")
        return wti, "야후파이낸스(실시간)"
    except Exception as e:
        print(f"[WTI] 폴백 ${fallback:.2f} ({e})")
        return fallback, f"폴백 ${fallback:.2f}"

# ──────────────────────────────────────────────
# 5. 원가 계산 (Gap 직접 회귀 방식)
# ──────────────────────────────────────────────
def calc_costs(latest, wti_rt, sens):
    """
    구글시트 실측값 기준 + WTI 델타 × 자동회귀 민감도 보정.

    핵심 변경 v5.0:
      ABS Gap = 실측 Gap + WTI 델타 × sens['abs_gap']
      → WTI↑ 시 Gap이 실제로 줄어드는 현실 반영
      ABS Market = 실측가 + WTI 델타 × sens['abs_mkt'] (역상관)
    """
    wti_gs  = float(latest[COL_MAP['wti']])
    bz_act  = float(latest[COL_MAP['bz']])
    et_act  = float(latest[COL_MAP['et']])
    nap_act = float(latest[COL_MAP['nap']])
    sm_act  = float(latest[COL_MAP['sm_cn']])
    bd_act  = float(latest[COL_MAP['bd']])
    an_act  = float(latest[COL_MAP['an']])
    abs_mkt_act = float(latest[COL_MAP['abs_mkt']])

    # 실측 파생값
    abs_cost_act  = float(latest['_abs_cost'])
    abs_gap_act   = float(latest['_abs_gap'])
    sm_cost_act   = float(latest['_sm_cost'])
    sm_margin_act = float(latest['_sm_margin'])

    delta = wti_rt - wti_gs  # WTI 실시간 - 구글시트 기준

    # 원료가 보정
    bz_rt  = bz_act  + delta * sens['bz']
    et_rt  = et_act  + delta * sens['et']
    sm_rt  = sm_act  + delta * sens['sm']
    bd_rt  = bd_act  + delta * sens['bd']
    an_rt  = an_act  + delta * sens['an']
    nap_rt = nap_act + delta * sens['nap']

    # SM Cost / Margin 보정
    sm_cost_rt   = sm_cost_act   + delta * sens['sm_cost']
    sm_margin_rt = sm_margin_act + delta * sens['sm_margin']

    # ★ ABS Cost / Market / Gap - 직접 회귀 보정
    abs_cost_rt  = abs_cost_act  + delta * sens['abs_cost']
    abs_mkt_rt   = abs_mkt_act   + delta * sens['abs_mkt']   # 역상관 반영
    abs_gap_rt   = abs_gap_act   + delta * sens['abs_gap']   # Gap 직접 보정 ★

    return {
        'WTI_RT':       round(wti_rt, 2),
        'WTI_GS':       round(wti_gs, 2),
        'WTI_Delta':    round(delta, 2),
        'NAP':          round(nap_rt, 1),
        'BZ':           round(bz_rt, 1),      'BZ_Actual':      round(bz_act, 1),
        'ET':           round(et_rt, 1),      'ET_Actual':      round(et_act, 1),
        'SM_Market':    round(sm_rt, 1),      'SM_Actual':      round(sm_act, 1),
        'SM_Cost':      round(sm_cost_rt, 1), 'SM_Cost_Actual': round(sm_cost_act, 1),
        'SM_Margin':    round(sm_margin_rt,1),'SM_Margin_Actual':round(sm_margin_act,1),
        'BD':           round(bd_rt, 1),      'BD_Actual':      round(bd_act, 1),
        'AN':           round(an_rt, 1),      'AN_Actual':      round(an_act, 1),
        'ABS_Market':   round(abs_mkt_rt, 1), 'ABS_Mkt_Actual': round(abs_mkt_act, 1),
        'ABS_Cost':     round(abs_cost_rt, 1),'ABS_Cost_Actual':round(abs_cost_act, 1),
        'ABS_Gap':      round(abs_gap_rt, 1), 'ABS_Gap_Actual': round(abs_gap_act, 1),
    }

# ──────────────────────────────────────────────
# 6. 시나리오 계산
# ──────────────────────────────────────────────
SCENARIOS = [
    {'label': 'Base\n($59)',     'wti': 59.44, 'risk':   0, 'color': '#2ecc71'},
    {'label': 'Mild\n($70)',     'wti': 70.00, 'risk':  50, 'color': '#f39c12'},
    {'label': 'Moderate\n($80)','wti': 80.00, 'risk': 100, 'color': '#e67e22'},
    {'label': 'Severe\n($90)',  'wti': 90.00, 'risk': 150, 'color': '#e74c3c'},
    {'label': 'Crisis\n($100)', 'wti':100.00, 'risk': 200, 'color': '#c0392b'},
]

def calc_scenario(latest, sens, wti_target, risk_premium):
    r = calc_costs(latest, wti_target, sens)
    # 리스크 프리미엄: ABS 시장가에 가산, Gap 재계산
    r['ABS_Market'] += risk_premium
    r['ABS_Gap']     = r['ABS_Market'] - r['ABS_Cost']
    r['Risk']        = risk_premium
    return r

# ──────────────────────────────────────────────
# 7. 차트 생성 (6패널)
# ──────────────────────────────────────────────
def generate_report(current, hist8, latest, sens, r2, n_reg, wti_source):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 13), facecolor='#0f172a')

    date_col = hist8.columns[0]
    gs_date  = pd.to_datetime(latest[date_col]).strftime('%Y-%m-%d')
    fig.suptitle(
        f'IRAN CONFLICT RISK DASHBOARD  |  WTI ${current["WTI_RT"]:.2f} (실시간)  |  '
        f'실측 앵커: {gs_date}  |  Gap 직접 회귀 v5.0  |  '
        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        fontsize=12, fontweight='bold', color='#fbbf24', y=0.98
    )

    dates = [pd.to_datetime(d).strftime('%m/%d') for d in hist8[date_col]]
    x     = range(len(dates))

    wti_h    = hist8[COL_MAP['wti']].tolist()
    bz_h     = hist8[COL_MAP['bz']].tolist()
    et_h     = hist8[COL_MAP['et']].tolist()
    sm_h     = hist8[COL_MAP['sm_cn']].tolist()
    bd_h     = hist8[COL_MAP['bd']].tolist()
    an_h     = hist8[COL_MAP['an']].tolist()
    abs_h    = hist8[COL_MAP['abs_mkt']].tolist()
    nap_h    = hist8[COL_MAP['nap']].tolist()
    abs_cost_h   = hist8['_abs_cost'].tolist()
    abs_gap_h    = hist8['_abs_gap'].tolist()
    sm_cost_h    = hist8['_sm_cost'].tolist()
    sm_margin_h  = hist8['_sm_margin'].tolist()

    def clean(lst): return [v if (v and not np.isnan(v)) else float('nan') for v in lst]

    # ── 차트1: ABS Gap 트렌드 (8주 실측) ★ 핵심
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#1e293b')
    gap_colors = ['#10b981' if g >= 150 else '#f59e0b' if g >= 0 else '#ef4444'
                  for g in abs_gap_h]
    bars = ax1.bar(list(x), abs_gap_h, color=gap_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for i, (bar, g) in enumerate(zip(bars, abs_gap_h)):
        if not np.isnan(g):
            ax1.text(i, g + (5 if g >= 0 else -20),
                     f'${g:.0f}', ha='center', fontsize=8, color='white', fontweight='bold')
    ax1.axhline(y=150, color='#fbbf24', linestyle='--', linewidth=1.5, label='경보 $150')
    ax1.axhline(y=0,   color='#ef4444', linestyle='-',  linewidth=1,   alpha=0.6)
    ax1.set_title(f'ABS Gap 8주 실측 | 현재 ${current["ABS_Gap_Actual"]:+.0f}/t → 보정 ${current["ABS_Gap"]:+.0f}/t',
                  color='#fbbf24', fontweight='bold', fontsize=9)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax1.set_ylabel('$/mt', color='#94a3b8'); ax1.tick_params(colors='#94a3b8')
    ax1.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax1.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트2: ABS Market vs Cost 트렌드
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#1e293b')
    abs_h_c = clean(abs_h)
    ax2.fill_between(x, abs_cost_h, abs_h_c, alpha=0.2, color='#10b981')
    ax2.plot(x, abs_h_c,    color='#3b82f6', linewidth=2, marker='o', markersize=5, label='ABS Market')
    ax2.plot(x, abs_cost_h, color='#ef4444', linewidth=2, marker='s', markersize=4, label='ABS Cost')
    ax2.set_title(f'ABS Market vs Cost (8W) | Gap ${current["ABS_Gap"]:+.0f}/t',
                  color='#10b981', fontweight='bold')
    ax2.set_xticks(list(x)); ax2.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax2.set_ylabel('$/mt', color='#94a3b8'); ax2.tick_params(colors='#94a3b8')
    ax2.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax2.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트3: SM Margin 트렌드
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#1e293b')
    sm_colors = ['#10b981' if m >= 0 else '#ef4444' for m in sm_margin_h]
    ax3.bar(list(x), sm_margin_h, color=sm_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for i, m in enumerate(sm_margin_h):
        if not np.isnan(m):
            ax3.text(i, m + (3 if m >= 0 else -15),
                     f'${m:.0f}', ha='center', fontsize=8, color='white', fontweight='bold')
    ax3.axhline(y=0, color='white', linewidth=1, alpha=0.5)
    sm_m_c = '#10b981' if current['SM_Margin'] >= 0 else '#ef4444'
    ax3.set_title(f'SM Margin 8주 | 현재 ${current["SM_Margin_Actual"]:+.0f} → 보정 ${current["SM_Margin"]:+.0f}/t',
                  color=sm_m_c, fontweight='bold', fontsize=9)
    ax3.set_xticks(list(x)); ax3.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax3.set_ylabel('$/mt', color='#94a3b8'); ax3.tick_params(colors='#94a3b8')
    for sp in ax3.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트4: SM Market vs Cost 트렌드
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#1e293b')
    ax4.fill_between(x, sm_cost_h, sm_h, alpha=0.2, color='#3b82f6')
    ax4.plot(x, sm_h,      color='#3b82f6', linewidth=2, marker='o', markersize=5, label='SM CFR China')
    ax4.plot(x, sm_cost_h, color='#ef4444', linewidth=2, marker='s', markersize=4, label='SM Cost')
    ax4.set_title('SM Market vs Cost Trend (8W)', color='#3b82f6', fontweight='bold')
    ax4.set_xticks(list(x)); ax4.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax4.set_ylabel('$/mt', color='#94a3b8'); ax4.tick_params(colors='#94a3b8')
    ax4.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax4.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트5: 원료가 트렌드 (BZ/ET/AN/BD + WTI)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#1e293b')
    ax5r = ax5.twinx()
    ax5r.plot(x, wti_h, color='#fbbf24', linewidth=2, marker='^', markersize=5,
              linestyle='--', label='WTI (R)')
    ax5r.set_ylabel('WTI $/bbl', color='#fbbf24', fontsize=9)
    ax5r.tick_params(axis='y', colors='#fbbf24')
    ax5r.set_ylim(40, 110)
    ax5.plot(x, bz_h,        color='#a855f7', linewidth=2, marker='o', markersize=5, label='BZ FOB KR')
    ax5.plot(x, et_h,        color='#10b981', linewidth=2, marker='s', markersize=4, label='ET CFR NEA')
    ax5.plot(x, clean(an_h), color='#f59e0b', linewidth=2, marker='D', markersize=4, label='AN CFR FEA')
    ax5.plot(x, clean(bd_h), color='#f97316', linewidth=2, marker='v', markersize=4, label='BD CFR CN')
    ax5.set_title('Raw Material Trend (8W)', color='#94a3b8', fontweight='bold')
    ax5.set_xticks(list(x)); ax5.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax5.set_ylabel('$/mt', color='#94a3b8'); ax5.tick_params(axis='y', colors='#94a3b8')
    ax5.set_ylim(400, 1600)
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5r.get_legend_handles_labels()
    ax5.legend(l1+l2, lb1+lb2, fontsize=7, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax5.spines.values(): sp.set_edgecolor('#334155')
    for sp in ax5r.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트6: 이란 시나리오 ABS Gap
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#1e293b')
    sc_costs, sc_mkts, sc_gaps, sc_cols = [], [], [], []
    for s in SCENARIOS:
        r = calc_scenario(latest, sens, s['wti'], s['risk'])
        sc_costs.append(r['ABS_Cost'])
        sc_mkts.append(r['ABS_Market'])
        sc_gaps.append(r['ABS_Gap'])
        sc_cols.append(s['color'])
    xp = np.arange(len(SCENARIOS))
    w  = 0.35
    ax6.bar(xp-w/2, sc_costs, w, label='ABS Cost',   color='#ef4444', alpha=0.85)
    ax6.bar(xp+w/2, sc_mkts,  w, label='ABS Market', color='#3b82f6', alpha=0.85)
    for i, (c, m, g, col) in enumerate(zip(sc_costs, sc_mkts, sc_gaps, sc_cols)):
        gc = '#10b981' if g >= 0 else '#ef4444'
        ax6.text(i, max(c, m)+15, f'${g:+.0f}',
                 ha='center', fontsize=8, color=gc, fontweight='bold')
    ax6.set_title('Iran Risk Scenario: ABS Gap', color='#fbbf24', fontweight='bold')
    ax6.set_xticks(xp)
    ax6.set_xticklabels([s['label'] for s in SCENARIOS], fontsize=8, color='white')
    ax6.set_ylabel('$/mt', color='#94a3b8'); ax6.tick_params(colors='#94a3b8')
    ax6.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155')
    ax6.set_ylim(0, max(sc_mkts+sc_costs)*1.2)
    for sp in ax6.spines.values(): sp.set_edgecolor('#334155')

    fig.text(0.5, 0.01,
             f'LAM Advanced Procurement  |  ABS Gap 직접 회귀 v5.0  |  {wti_source}',
             ha='center', fontsize=8, color='#475569')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('risk_simulation_report.png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print("[차트] risk_simulation_report.png 저장 완료")

# ──────────────────────────────────────────────
# 8. CSV 저장
# ──────────────────────────────────────────────
def save_csv(current, sens, r2, n_reg, wti_source, gs_date):
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    row = {
        'UpdateTime':       now,
        'WTI_Source':       wti_source,
        'GSheet_Date':      gs_date,
        'Reg_N':            n_reg,
        # 실시간 보정값
        'WTI':              current['WTI_RT'],
        'WTI_GSheet':       current['WTI_GS'],
        'WTI_Delta':        current['WTI_Delta'],
        'NAP':              current['NAP'],
        'BZ':               current['BZ'],          'BZ_Actual':        current['BZ_Actual'],
        'ET':               current['ET'],          'ET_Actual':        current['ET_Actual'],
        'SM_Market':        current['SM_Market'],   'SM_Actual':        current['SM_Actual'],
        'SM_Cost':          current['SM_Cost'],     'SM_Cost_Actual':   current['SM_Cost_Actual'],
        'SM_Margin':        current['SM_Margin'],   'SM_Margin_Actual': current['SM_Margin_Actual'],
        'BD':               current['BD'],          'BD_Actual':        current['BD_Actual'],
        'AN':               current['AN'],          'AN_Actual':        current['AN_Actual'],
        'ABS_Market':       current['ABS_Market'],  'ABS_Mkt_Actual':   current['ABS_Mkt_Actual'],
        'ABS_Cost':         current['ABS_Cost'],    'ABS_Cost_Actual':  current['ABS_Cost_Actual'],
        'ABS_Gap':          current['ABS_Gap'],     'ABS_Gap_Actual':   current['ABS_Gap_Actual'],
        # 자동 회귀계수
        'Sens_BZ':          sens['bz'],       'R2_BZ':        r2.get('bz',''),
        'Sens_ET':          sens['et'],       'R2_ET':        r2.get('et',''),
        'Sens_SM':          sens['sm'],       'R2_SM':        r2.get('sm',''),
        'Sens_AN':          sens['an'],       'R2_AN':        r2.get('an',''),
        'Sens_BD':          sens['bd'],       'R2_BD':        r2.get('bd',''),
        'Sens_NAP':         sens['nap'],      'R2_NAP':       r2.get('nap',''),
        'Sens_ABS_MKT':     sens['abs_mkt'],  'R2_ABS_MKT':   r2.get('abs_mkt',''),
        'Sens_ABS_GAP':     sens['abs_gap'],  'R2_ABS_GAP':   r2.get('abs_gap',''),
        'Sens_ABS_COST':    sens['abs_cost'], 'R2_ABS_COST':  r2.get('abs_cost',''),
        'Sens_SM_MARGIN':   sens['sm_margin'],'R2_SM_MARGIN': r2.get('sm_margin',''),
        # 구버전 호환
        'Margin':           current['SM_Margin'],
        'ABS_Landed':       current['ABS_Market'],
    }
    pd.DataFrame([row]).to_csv('simulation_result.csv', index=False)
    print("[CSV] simulation_result.csv 저장 완료")

# ──────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Iran Risk × ABS/SM 원가 시뮬레이션 v5.0")
    print("ABS Gap 직접 회귀 - WTI↑=Gap↓ 현실 반영")
    print("=" * 65)

    setup_font()

    latest, df_all, hist8 = load_gsheet()
    if latest is None:
        print("구글시트 로드 실패 - 종료")
        exit(1)

    gs_date = pd.to_datetime(latest[hist8.columns[0]]).strftime('%Y-%m-%d')

    # 자동 회귀계수 계산
    sens, r2, n_reg = calc_regression(df_all)

    # WTI 실시간
    wti_rt, wti_src = get_wti(fallback=float(latest[COL_MAP['wti']]))

    # 원가 계산
    current = calc_costs(latest, wti_rt, sens)

    # 결과 출력
    print(f"\n{'─'*55}")
    print(f"  WTI          : ${current['WTI_RT']:.2f}  (GS ${current['WTI_GS']:.2f}, Δ{current['WTI_Delta']:+.2f})")
    print(f"  BZ FOB KR    : 실측 ${current['BZ_Actual']:.0f} → 보정 ${current['BZ']:.0f}  (sens {sens['bz']:+.2f} R²={r2.get('bz','N/A')})")
    print(f"  ET CFR NEA   : 실측 ${current['ET_Actual']:.0f} → 보정 ${current['ET']:.0f}  (sens {sens['et']:+.2f} R²={r2.get('et','N/A')})")
    print(f"  SM CFR China : 실측 ${current['SM_Actual']:.0f} → 보정 ${current['SM_Market']:.0f}  (sens {sens['sm']:+.2f} R²={r2.get('sm','N/A')})")
    print(f"  AN CFR FEA   : 실측 ${current['AN_Actual']:.0f} → 보정 ${current['AN']:.0f}  (sens {sens['an']:+.2f} R²={r2.get('an','N/A')})")
    print(f"  BD CFR China : 실측 ${current['BD_Actual']:.0f} → 보정 ${current['BD']:.0f}  (sens {sens['bd']:+.2f} R²={r2.get('bd','N/A')})")
    print(f"  ─────────────────────────────────────────────")
    print(f"  SM Cost      : 실측 ${current['SM_Cost_Actual']:.0f} → 보정 ${current['SM_Cost']:.0f}/t")
    print(f"  SM Margin    : 실측 ${current['SM_Margin_Actual']:+.0f} → 보정 ${current['SM_Margin']:+.0f}/t  "
          f"{'⚠ 역마진!' if current['SM_Margin'] < 0 else '✓'}")
    print(f"  ABS Cost     : 실측 ${current['ABS_Cost_Actual']:.0f} → 보정 ${current['ABS_Cost']:.0f}/t")
    print(f"  ABS Market   : 실측 ${current['ABS_Mkt_Actual']:.0f} → 보정 ${current['ABS_Market']:.0f}/t  "
          f"(역상관 sens {sens['abs_mkt']:+.2f})")
    print(f"  ABS Gap ★    : 실측 ${current['ABS_Gap_Actual']:+.0f} → 보정 ${current['ABS_Gap']:+.0f}/t  "
          f"(sens {sens['abs_gap']:+.2f} R²={r2.get('abs_gap','N/A')})  "
          f"{'⚠ 경보!' if current['ABS_Gap'] < 150 else '✓'}")
    print(f"{'─'*55}\n")

    print("[ 이란 리스크 시나리오 ]")
    for s in SCENARIOS:
        r = calc_scenario(latest, sens, s['wti'], s['risk'])
        flag = '⚠ 역마진!' if r['ABS_Gap'] < 0 else ('⚠ 경보' if r['ABS_Gap'] < 150 else '✓')
        print(f"  {s['label'].replace(chr(10),' '):22s} | WTI ${s['wti']:5.0f} | "
              f"Risk +${s['risk']:3.0f} | ABS Gap ${r['ABS_Gap']:+.0f}/t {flag}")

    generate_report(current, hist8, latest, sens, r2, n_reg, wti_src)
    save_csv(current, sens, r2, n_reg, wti_src, gs_date)

    print("\n시뮬레이션 완료.")
    print("=" * 65)
