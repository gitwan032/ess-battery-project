"""피처 계산 (모든 피처는 cycle 100 이전 데이터만 사용)

DAY1 전략서의 피처 정의를 그대로 구현
- 기본 : dq_var   = log10(Var[Q100(V) - Q10(V)])     (1,000점 전압 격자)
- 비교 : fade_10_100 = cycle 10~100 QD 선형 기울기 (mAh/cycle)
- 대체 : dq_min, dq_mean = log10|min ΔQ|, log10|mean ΔQ|
- 참고(제외 피처, 비교 · 해석용) : QD_2, IR_2, Tavg_100, chargetime_5, avgC

실행 : python src/features.py   ->  data/features.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_summary, load_cells

POLICY = re.compile(r'([\d.]+)C\((\d+)%\)-([\d.]+)C')


def delta_q(cell, a=100, b=10):
    return cell['curves'][a]['Qdlin'] - cell['curves'][b]['Qdlin']


def avg_c_rate(policy):
    """0~80% 평균 C-rate = 0.8 / (SOC1/C1 + (0.8-SOC1)/C2)"""
    m = POLICY.search(policy)
    if not m:
        return np.nan
    c1, q1, c2 = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
    return 0.8 / (q1 / c1 + max(0.8 - q1, 0) / c2)


def cell_features(cell):
    dq = delta_q(cell)
    s = clean_summary(cell)
    e = s[s['cycle'].between(2, 100)]
    w = s[s['cycle'].between(10, 100)]
    fade = np.polyfit(w['cycle'], w['QD'], 1)[0] * 1000 if len(w) > 10 else np.nan
    return {
        'cell_id': cell['cell_id'], 'batch': cell['batch'], 'policy': cell['policy'],
        'cycle_life': cell['cycle_life'],
        # 전략 피처
        'dq_var': np.log10(np.var(dq)),
        'dq_min': np.log10(abs(dq.min()) + 1e-12),
        'dq_mean': np.log10(abs(dq.mean()) + 1e-12),
        'fade_10_100': fade,
        # 제외 피처 (EDA 근거 확인 · 비교용)
        'QD_2': e['QD'].iloc[0] if len(e) else np.nan,
        'IR_2': e['IR'].dropna().iloc[0] if e['IR'].notna().any() else np.nan,
        'Tavg_100': e['Tavg'].mean(),
        'chargetime_5': e['chargetime'].dropna().iloc[:5].mean(),
        'avgC': avg_c_rate(cell['policy']),
    }


def build_features(cache_dir='data/cache', out='data/features.csv'):
    cells = load_cells(cache_dir)
    df = pd.DataFrame([cell_features(c) for c in cells])
    df['log_life'] = np.log10(df['cycle_life'])
    df.to_csv(out, index=False)
    print(df.groupby('batch').size(), f'-> {out}')
    return df


if __name__ == '__main__':
    build_features()
