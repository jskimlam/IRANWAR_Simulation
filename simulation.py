# ============================================================
# pip install pandas matplotlib numpy yfinance requests scikit-learn
# ============================================================
# simulation.py - Iran Risk × ABS/SM 원가 대응 시뮬레이션
# v3.0 - 구글시트 실측가 + 자동 회귀계수 갱신 + WTI 실시간
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
import json

# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
GSHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQfp5e4ufXsgu7YvZ5EFEHatkJ7BdgW3vma78THWYn66wHUrau8hYB4q8TY2OXuc9PBguq-v09CkmKZ"
    "/pub?gid=0&single=true&output=csv"
)

# 컬럼 매핑 (구글시트 헤더 기준)
COL_MAP = {
    'wti':    'NYMEX Light Sweet Crude Settlement Mo01',
    'nap':    'Naphtha C+F Japan Cargo $/mt (NextGen MOC)',
    'sm_cn':  'Styrene CFR China Marker LC 90 days',
    'sm_fob': 'Styrene Monomer FOB China Marker',
    'et':     'Ethylene CFR NE Asia',
    'bz':     'Benzene FOB Korea Marker',
    'bz_ara': 'Benzene CIF ARA',
    'bz_usg': 'Benzene FOB USG Mo02 cts/gal',
    'pr':     'Propylene Poly Grade CFR China',
    'bd':     'Butadiene CFR China',
}

# 기본 민감도 (데이터 부족 시 폴백용)
DEFAULT_SENS = {
    'bz': 9.65,
    'et': -5.41,   # 공급과잉 역상관
    'sm': 20.07,
    'bd': 15.0,
    'pr': 5.0,
    'nap': 8.5,
}

# ──────────────────────────────────────────────
# 1. 한글 폰트 설정
# ──────────────────────────────────────────────
def setup_font():
    """서버 환경 한글 폰트 자동 탐색"""
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
    print("[폰트] DejaVu Sans (한글폰트 없음)")

# ──────────────────────────────────────────────
# 2. 구글시트 파싱
# ──────────────────────────────────────────────
def load_gsheet():
    """
    구글시트 CSV 파싱.
    반환: (latest행, 전체 DataFrame, 최근 8주 DataFrame)
    """
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

    wti_col = COL_MAP['wti']
    df = df[df[wti_col].notna()].copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    if df.empty:
        print("[구글시트] 유효 데이터 없음")
        return None, None, None

    latest = df.iloc[-1]
    hist8  = df.tail(8).copy()

    print(f"[구글시트] 총 {len(df)}주 데이터 | 최신: {latest[date_col].strftime('%Y-%m-%d')}")
    print(f"  WTI={latest[wti_col]:.2f} | BZ={latest[COL_MAP['bz']]:.1f} | "
          f"ET={latest[COL_MAP['et']]:.1f} | SM={latest[COL_MAP['sm_cn']]:.1f} | "
          f"BD={latest[COL_MAP['bd']]:.1f}")

    return latest, df, hist8

# ──────────────────────────────────────────────
# 3. 자동 회귀계수 계산 ★ 핵심 신규 기능
# ──────────────────────────────────────────────
def calc_regression(df):
    """
    구글시트 전체 데이터로 WTI 대비 각 원료 민감도(회귀계수) 자동 계산.
    데이터가 4주 미만이면 기본값 사용.
    R² 0.3 미만이면 해당 품목은 기본값 폴백.

    반환:
        sens (dict): 품목별 민감도
        r2   (dict): 품목별 R²
        n    (int) : 사용 데이터 수
    """
    from numpy.polynomial import polynomial as P

    wti_col = COL_MAP['wti']
    targets = {
        'bz':  COL_MAP['bz'],
        'et':  COL_MAP['et'],
        'sm':  COL_MAP['sm_cn'],
        'bd':  COL_MAP['bd'],
        'pr':  COL_MAP['pr'],
        'nap': COL_MAP['nap'],
    }

    sens = {}
    r2   = {}
    n    = 0

    for key, col in targets.items():
        # 유효 행만 (두 컬럼 모두 값 있는 행)
        valid = df[[wti_col, col]].dropna()
        n_pts = len(valid)

        if n_pts < 4:
            # 데이터 부족 → 기본값
            sens[key] = DEFAULT_SENS[key]
            r2[key]   = None
            print(f"  [{key}] 데이터 부족({n_pts}개) → 기본값 {DEFAULT_SENS[key]:.2f}")
            continue

        x = valid[wti_col].values
        y = valid[col].values

        # 선형 회귀 (numpy)
        m, b = np.polyfit(x, y, 1)

        # R² 계산
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_val = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # R² 0.3 미만이면 신뢰도 낮아서 기본값 사용
        if r2_val < 0.3:
            sens[key] = DEFAULT_SENS[key]
            r2[key]   = round(r2_val, 3)
            print(f"  [{key}] R²={r2_val:.3f} (낮음) → 기본값 {DEFAULT_SENS[key]:.2f} 사용")
        else:
            sens[key] = round(m, 3)
            r2[key]   = round(r2_val, 3)
            print(f"  [{key}] m={m:.3f} R²={r2_val:.3f} n={n_pts} ✓")

        n = max(n, n_pts)

    return sens, r2, n

# ──────────────────────────────────────────────
# 4. WTI 실시간 취득
# ──────────────────────────────────────────────
def get_wti(fallback=67.02):
    """야후파이낸스 CL=F 실시간 (롤오버 자동)"""
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
# 5. 원가 계산 (실측 + 자동회귀 보정)
# ──────────────────────────────────────────────
def calc_costs(latest, wti_rt, sens):
    """
    구글시트 최신 실측가 + WTI 실시간 델타 + 자동회귀 민감도로 원가 계산.
    """
    bz_act  = float(latest[COL_MAP['bz']])
    et_act  = float(latest[COL_MAP['et']])
    nap_act = float(latest[COL_MAP['nap']])
    sm_act  = float(latest[COL_MAP['sm_cn']])
    bd_act  = float(latest[COL_MAP['bd']])
    pr_act  = float(latest[COL_MAP['pr']])
    wti_gs  = float(latest[COL_MAP['wti']])

    delta = wti_rt - wti_gs   # WTI 실시간 - 구글시트 기준

    # 실측가 + 자동 민감도 × 델타 보정
    bz_rt  = bz_act  + delta * sens['bz']
    et_rt  = et_act  + delta * sens['et']
    sm_rt  = sm_act  + delta * sens['sm']
    bd_rt  = bd_act  + delta * sens['bd']
    pr_rt  = pr_act  + delta * sens['pr']
    nap_rt = nap_act + delta * sens['nap']

    # SM 제조원가: BZ 0.78t + ET 0.28t + 고정비 $45
    sm_cost   = bz_rt * 0.78 + et_rt * 0.28 + 45.0
    sm_margin = sm_rt - sm_cost

    # ABS 제조원가: SM 30% + BD 12% + AN(≈PR×1.15) 8% + 고정비 $680
    an_est  = pr_rt * 1.15
    abs_cost = sm_rt * 0.30 + bd_rt * 0.12 + an_est * 0.08 + 680.0

    # ABS 시장가: SM + 스프레드 $250
    abs_mkt = sm_rt + 250.0
    abs_gap = abs_mkt - abs_cost

    return {
        'WTI_RT':    round(wti_rt, 2),
        'WTI_GS':    round(wti_gs, 2),
        'WTI_Delta': round(delta, 2),
        'NAP':       round(nap_rt, 1),
        'BZ':        round(bz_rt, 1),   'BZ_Actual': round(bz_act, 1),
        'ET':        round(et_rt, 1),   'ET_Actual': round(et_act, 1),
        'BD':        round(bd_rt, 1),   'BD_Actual': round(bd_act, 1),
        'PR':        round(pr_rt, 1),
        'SM_Market': round(sm_rt, 1),   'SM_Actual': round(sm_act, 1),
        'SM_Cost':   round(sm_cost, 1),
        'SM_Margin': round(sm_margin, 1),
        'ABS_Market':round(abs_mkt, 1),
        'ABS_Cost':  round(abs_cost, 1),
        'ABS_Gap':   round(abs_gap, 1),
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

    gs_date = pd.to_datetime(latest[hist8.columns[0]]).strftime('%Y-%m-%d')
    fig.suptitle(
        f'IRAN CONFLICT RISK DASHBOARD  |  WTI ${current["WTI_RT"]:.2f} (실시간)  |  '
        f'실측 앵커: {gs_date}  |  회귀계수 자동갱신 ({n_reg}주 데이터)  |  '
        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        fontsize=12, fontweight='bold', color='#fbbf24', y=0.98
    )

    date_col = hist8.columns[0]
    dates    = [pd.to_datetime(d).strftime('%m/%d') for d in hist8[date_col]]
    x        = range(len(dates))

    wti_h = hist8[COL_MAP['wti']].tolist()
    bz_h  = hist8[COL_MAP['bz']].tolist()
    et_h  = hist8[COL_MAP['et']].tolist()
    sm_h  = hist8[COL_MAP['sm_cn']].tolist()
    bd_h  = hist8[COL_MAP['bd']].tolist()
    pr_h  = hist8[COL_MAP['pr']].tolist()
    nap_h = hist8[COL_MAP['nap']].tolist()

    # SM Cost 히스토리
    sm_cost_h = [
        bz_h[i] * 0.78 + et_h[i] * 0.28 + 45
        if bz_h[i] and et_h[i] else float('nan')
        for i in range(len(bz_h))
    ]
    sm_margin_h = [
        sm_h[i] - sm_cost_h[i]
        if sm_h[i] and not np.isnan(sm_cost_h[i]) else float('nan')
        for i in range(len(sm_h))
    ]

    # ABS Cost/Market 히스토리
    abs_cost_h, abs_mkt_h, abs_gap_h = [], [], []
    for i in range(len(sm_h)):
        if sm_h[i] and bd_h[i] and pr_h[i]:
            an_i  = pr_h[i] * 1.15
            ac    = sm_h[i]*0.30 + bd_h[i]*0.12 + an_i*0.08 + 680
            am    = sm_h[i] + 250
            abs_cost_h.append(ac)
            abs_mkt_h.append(am)
            abs_gap_h.append(am - ac)
        else:
            abs_cost_h.append(float('nan'))
            abs_mkt_h.append(float('nan'))
            abs_gap_h.append(float('nan'))

    # ── 차트1: SM 수익성 막대
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#1e293b')
    mc   = '#10b981' if current['SM_Margin'] >= 0 else '#ef4444'
    bars = ax1.bar(
        ['SM\nMarket', 'SM\nCost'],
        [current['SM_Market'], current['SM_Cost']],
        color=['#3b82f6', '#ef4444'], width=0.5, edgecolor='white', linewidth=0.5
    )
    for bar, v in zip(bars, [current['SM_Market'], current['SM_Cost']]):
        ax1.text(bar.get_x()+bar.get_width()/2, v+8, f'${v:,.0f}',
                 ha='center', fontweight='bold', color='white', fontsize=11)
    ax1.set_title(f'SM Profitability | Margin ${current["SM_Margin"]:+.0f}/t', color=mc, fontweight='bold')
    ax1.set_ylabel('$/mt', color='#94a3b8')
    ax1.tick_params(colors='#94a3b8')
    ax1.set_ylim(0, max(current['SM_Market'], current['SM_Cost']) * 1.2)
    for sp in ax1.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트2: ABS 마진 트렌드
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#1e293b')
    ax2.fill_between(x, abs_cost_h, abs_mkt_h, alpha=0.2, color='#10b981')
    ax2.plot(x, abs_mkt_h,  color='#3b82f6', linewidth=2, marker='o', markersize=5, label='ABS Market')
    ax2.plot(x, abs_cost_h, color='#ef4444', linewidth=2, marker='s', markersize=4, label='ABS Cost')
    for i, g in enumerate(abs_gap_h):
        if not np.isnan(g):
            gc = '#10b981' if g >= 0 else '#ef4444'
            ax2.annotate(f'${g:.0f}', (i, abs_cost_h[i] + abs(g)/2),
                         ha='center', fontsize=7, color=gc)
    ax2.axhline(y=1200, color='#fbbf24', linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_title(f'ABS Margin Trend (8W) | Gap ${current["ABS_Gap"]:+.0f}/t',
                  color='#10b981', fontweight='bold')
    ax2.set_xticks(list(x)); ax2.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax2.set_ylabel('$/mt', color='#94a3b8'); ax2.tick_params(colors='#94a3b8')
    ax2.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax2.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트3: 회귀계수 시각화 ★ 신규
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#1e293b')
    reg_items = [
        ('BZ', sens['bz'],  r2.get('bz'),  '#a855f7'),
        ('ET', sens['et'],  r2.get('et'),  '#10b981'),
        ('SM', sens['sm'],  r2.get('sm'),  '#3b82f6'),
        ('BD', sens['bd'],  r2.get('bd'),  '#f97316'),
        ('PR', sens['pr'],  r2.get('pr'),  '#f472b6'),
        ('NAP',sens['nap'], r2.get('nap'), '#94a3b8'),
    ]
    names  = [r[0] for r in reg_items]
    values = [r[1] for r in reg_items]
    colors = [r[3] for r in reg_items]
    r2vals = [r[2] for r in reg_items]
    xpos   = np.arange(len(names))
    bars3  = ax3.bar(xpos, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax3.axhline(y=0, color='white', linewidth=0.8, alpha=0.4)
    for i, (bar, v, r2v) in enumerate(zip(bars3, values, r2vals)):
        ypos = v + 0.3 if v >= 0 else v - 1.2
        r2_str = f'R²={r2v:.2f}' if r2v is not None else 'default'
        color  = 'white' if r2v and r2v >= 0.3 else '#f59e0b'
        ax3.text(i, ypos, f'{v:+.1f}\n{r2_str}',
                 ha='center', fontsize=7.5, color=color, fontweight='bold')
    ax3.set_title(f'Auto Regression Sensitivity ($/mt per $1/bbl WTI)\n{n_reg}주 실측 데이터 기반',
                  color='#fbbf24', fontweight='bold', fontsize=9)
    ax3.set_xticks(xpos); ax3.set_xticklabels(names, color='white', fontsize=10)
    ax3.set_ylabel('$/mt per $/bbl WTI', color='#94a3b8')
    ax3.tick_params(colors='#94a3b8')
    for sp in ax3.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트4: SM Market vs Cost 트렌드
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#1e293b')
    ax4.fill_between(x, sm_cost_h, sm_h, alpha=0.2, color='#3b82f6')
    ax4.plot(x, sm_h,      color='#3b82f6', linewidth=2, marker='o', markersize=5, label='SM CFR China')
    ax4.plot(x, sm_cost_h, color='#ef4444', linewidth=2, marker='s', markersize=4, label='SM Cost')
    for i, mg in enumerate(sm_margin_h):
        if not np.isnan(mg):
            gc = '#10b981' if mg >= 0 else '#ef4444'
            ax4.annotate(f'${mg:.0f}', (i, sm_cost_h[i] + mg/2),
                         ha='center', fontsize=7, color=gc)
    ax4.set_title('SM Market vs Cost Trend (8W)', color='#3b82f6', fontweight='bold')
    ax4.set_xticks(list(x)); ax4.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax4.set_ylabel('$/mt', color='#94a3b8'); ax4.tick_params(colors='#94a3b8')
    ax4.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax4.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트5: 원료가 트렌드 (WTI/BZ/ET/BD/Nap)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#1e293b')
    ax5r = ax5.twinx()
    ax5r.plot(x, wti_h, color='#fbbf24', linewidth=2, marker='^', markersize=5,
              linestyle='--', label='WTI (R)')
    ax5r.set_ylabel('WTI $/bbl', color='#fbbf24', fontsize=9)
    ax5r.tick_params(axis='y', colors='#fbbf24')
    ax5r.set_ylim(40, 110)
    ax5.plot(x, bz_h,  color='#a855f7', linewidth=2, marker='o', markersize=5, label='BZ FOB KR')
    ax5.plot(x, et_h,  color='#10b981', linewidth=2, marker='s', markersize=4, label='ET CFR NEA')
    ax5.plot(x, [v if v else float('nan') for v in bd_h],
             color='#f97316', linewidth=2, marker='D', markersize=4, label='BD CFR CN')
    ax5.plot(x, [v if v else float('nan') for v in nap_h],
             color='#94a3b8', linewidth=1.5, marker='v', markersize=4, linestyle=':', label='Naphtha')
    ax5.set_title('Raw Material Trend (8W)', color='#94a3b8', fontweight='bold')
    ax5.set_xticks(list(x)); ax5.set_xticklabels(dates, fontsize=8, color='#94a3b8', rotation=30)
    ax5.set_ylabel('$/mt', color='#94a3b8'); ax5.tick_params(axis='y', colors='#94a3b8')
    ax5.set_ylim(400, 1500)
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5r.get_legend_handles_labels()
    ax5.legend(l1+l2, lb1+lb2, fontsize=7, facecolor='#1e293b', edgecolor='#334155')
    for sp in ax5.spines.values(): sp.set_edgecolor('#334155')
    for sp in ax5r.spines.values(): sp.set_edgecolor('#334155')

    # ── 차트6: 이란 시나리오 ABS Gap
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#1e293b')
    sc_costs, sc_mkts, sc_gaps, sc_colors = [], [], [], []
    for s in SCENARIOS:
        r = calc_scenario(latest, sens, s['wti'], s['risk'])
        sc_costs.append(r['ABS_Cost'])
        sc_mkts.append(r['ABS_Market'])
        sc_gaps.append(r['ABS_Gap'])
        sc_colors.append(s['color'])
    xp = np.arange(len(SCENARIOS))
    w  = 0.35
    ax6.bar(xp - w/2, sc_costs, w, label='ABS Cost',   color='#ef4444', alpha=0.85)
    ax6.bar(xp + w/2, sc_mkts,  w, label='ABS Market', color='#3b82f6', alpha=0.85)
    for i, (c, m, g, col) in enumerate(zip(sc_costs, sc_mkts, sc_gaps, sc_colors)):
        gc = '#10b981' if g >= 0 else '#ef4444'
        ax6.text(i, max(c, m)+15, f'${g:+.0f}',
                 ha='center', fontsize=8, color=gc, fontweight='bold')
    ax6.set_title('Iran Risk Scenario: ABS Gap', color='#fbbf24', fontweight='bold')
    ax6.set_xticks(xp)
    ax6.set_xticklabels([s['label'] for s in SCENARIOS], fontsize=8, color='white')
    ax6.set_ylabel('$/mt', color='#94a3b8'); ax6.tick_params(colors='#94a3b8')
    ax6.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155')
    ax6.set_ylim(0, max(sc_mkts + sc_costs) * 1.2)
    for sp in ax6.spines.values(): sp.set_edgecolor('#334155')

    # 워터마크
    fig.text(0.5, 0.01,
             f'LAM Advanced Procurement  |  구글시트 실측가 + Yahoo Finance CL=F  |  {wti_source}',
             ha='center', fontsize=8, color='#475569')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('risk_simulation_report.png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print("[차트] risk_simulation_report.png 저장 완료")

# ──────────────────────────────────────────────
# 8. CSV 저장 (회귀계수 포함)
# ──────────────────────────────────────────────
def save_csv(current, sens, r2, n_reg, wti_source, gs_date):
    """simulation_result.csv 저장 - index.html이 읽어서 표시"""
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    row = {
        # 메타
        'UpdateTime':   now,
        'WTI_Source':   wti_source,
        'GSheet_Date':  gs_date,
        'Reg_N':        n_reg,
        # 실시간 가격
        'WTI':          current['WTI_RT'],
        'WTI_GSheet':   current['WTI_GS'],
        'WTI_Delta':    current['WTI_Delta'],
        'NAP':          current['NAP'],
        'BZ':           current['BZ'],        'BZ_Actual':  current['BZ_Actual'],
        'ET':           current['ET'],        'ET_Actual':  current['ET_Actual'],
        'BD':           current['BD'],        'BD_Actual':  current['BD_Actual'],
        'PR':           current['PR'],
        'SM_Market':    current['SM_Market'], 'SM_Actual':  current['SM_Actual'],
        'SM_Cost':      current['SM_Cost'],
        'SM_Margin':    current['SM_Margin'],
        'ABS_Market':   current['ABS_Market'],
        'ABS_Cost':     current['ABS_Cost'],
        'ABS_Gap':      current['ABS_Gap'],
        # 자동 회귀계수
        'Sens_BZ':      sens['bz'],   'R2_BZ':  r2.get('bz', ''),
        'Sens_ET':      sens['et'],   'R2_ET':  r2.get('et', ''),
        'Sens_SM':      sens['sm'],   'R2_SM':  r2.get('sm', ''),
        'Sens_BD':      sens['bd'],   'R2_BD':  r2.get('bd', ''),
        'Sens_PR':      sens['pr'],   'R2_PR':  r2.get('pr', ''),
        'Sens_NAP':     sens['nap'],  'R2_NAP': r2.get('nap', ''),
        # 구버전 호환
        'Margin':       current['SM_Margin'],
        'ABS_Landed':   current['ABS_Market'],
    }
    pd.DataFrame([row]).to_csv('simulation_result.csv', index=False)
    print("[CSV] simulation_result.csv 저장 완료")

# ──────────────────────────────────────────────
# 9. 메인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Iran Risk × ABS/SM 원가 시뮬레이션 v3.0 (자동 회귀계수)")
    print("=" * 65)

    setup_font()

    # 구글시트 로드
    latest, df_all, hist8 = load_gsheet()
    if latest is None:
        print("구글시트 로드 실패 - 종료")
        exit(1)

    gs_date = pd.to_datetime(latest[hist8.columns[0]]).strftime('%Y-%m-%d')

    # ★ 자동 회귀계수 계산
    print("\n[ 자동 회귀계수 계산 중... ]")
    sens, r2, n_reg = calc_regression(df_all)

    # WTI 실시간
    wti_rt, wti_src = get_wti(fallback=float(latest[COL_MAP['wti']]))

    # 원가 계산
    current = calc_costs(latest, wti_rt, sens)

    # 결과 출력
    print(f"\n{'─'*50}")
    print(f"  WTI 실시간  : ${current['WTI_RT']:.2f}  "
          f"(구글시트 ${current['WTI_GS']:.2f}, Δ{current['WTI_Delta']:+.2f})")
    print(f"  Naphtha     : ${current['NAP']:.0f}/t")
    print(f"  BZ FOB KR   : 실측 ${current['BZ_Actual']:.0f} → 보정 ${current['BZ']:.0f}  "
          f"(sens {sens['bz']:+.2f}, R²={r2.get('bz','N/A')})")
    print(f"  ET CFR NEA  : 실측 ${current['ET_Actual']:.0f} → 보정 ${current['ET']:.0f}  "
          f"(sens {sens['et']:+.2f}, R²={r2.get('et','N/A')})")
    print(f"  BD CFR CN   : 실측 ${current['BD_Actual']:.0f} → 보정 ${current['BD']:.0f}  "
          f"(sens {sens['bd']:+.2f}, R²={r2.get('bd','N/A')})")
    print(f"  SM Market   : 실측 ${current['SM_Actual']:.0f} → 보정 ${current['SM_Market']:.0f}  "
          f"(sens {sens['sm']:+.2f}, R²={r2.get('sm','N/A')})")
    print(f"  SM Cost     : ${current['SM_Cost']:.0f}/t")
    print(f"  SM Margin   : ${current['SM_Margin']:+.0f}/t  "
          f"{'⚠ 역마진!' if current['SM_Margin'] < 0 else '✓'}")
    print(f"  ABS Market  : ${current['ABS_Market']:.0f}/t")
    print(f"  ABS Cost    : ${current['ABS_Cost']:.0f}/t")
    print(f"  ABS Gap     : ${current['ABS_Gap']:+.0f}/t  "
          f"{'⚠ 경보!' if current['ABS_Gap'] < 150 else '✓'}")
    print(f"{'─'*50}\n")

    print("[ 이란 리스크 시나리오 ]")
    for s in SCENARIOS:
        r = calc_scenario(latest, sens, s['wti'], s['risk'])
        flag = '⚠ 역마진!' if r['ABS_Gap'] < 0 else ('⚠ 경보' if r['ABS_Gap'] < 150 else '✓')
        print(f"  {s['label'].replace(chr(10),' '):22s} | "
              f"WTI ${s['wti']:5.0f} | Risk +${s['risk']:3.0f} | "
              f"ABS Gap ${r['ABS_Gap']:+.0f}/t {flag}")

    generate_report(current, hist8, latest, sens, r2, n_reg, wti_src)
    save_csv(current, sens, r2, n_reg, wti_src, gs_date)

    print("\n시뮬레이션 완료.")
    print("=" * 65)
