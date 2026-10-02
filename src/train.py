"""학습 · 평가

DAY1 전략 구현
- Target : log10(cycle_life) 로 학습, 평가는 사이클 단위로 복원 (10^y)
- 학습 : B1 (36셀) / 평가 : B2 (1차, 39셀), B3 (2차, 40셀)  -> 배치 단위 분할
- 튜닝 : B1 내부 충전 정책 단위 GroupKFold (같은 정책 셀이 학습/검증에 나뉘지 않게)
- 전처리 : 결측 대체 · 표준화는 Pipeline 안에서 (CV fold마다 학습 fold로만 적합)
- 평가 지표 : MAPE(주) · MAE · RMSE,  GAP = B2 MAPE - 논문 9.1 (%p)
- 외삽 점검 : 평가 셀을 학습 수명 범위 안 / 밖으로 나눠 MAPE 비교
- B2 · B3 결과는 보고만 하고 모델 선택 · 튜닝에 사용하지 않음

실행 : python src/train.py --features data/features.csv
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
SEED = 42
PAPER_MAPE = 9.1   # 원논문 테스트 오차 (%), 수업 자료 기준

# 피처 세트 : DAY1 전략서 '입력 변수' 표
FEATURE_SETS = {
    'A_dq_var':        ['dq_var'],                         # 기본
    'B_dq_var+fade':   ['dq_var', 'fade_10_100'],          # 비교 (B1 rho 0.50)
    'C_dq_min':        ['dq_min'],                         # 대체
    'D_with_excluded': ['dq_var', 'fade_10_100', 'QD_2', 'IR_2', 'Tavg_100', 'chargetime_5'],
    # D : EDA에서 '제외'로 판단한 피처를 넣으면 정말 나빠지는지 확인하는 대조군
}

MODELS = {
    'Dummy':      (DummyRegressor(strategy='mean'), {}),
    'ElasticNet': (ElasticNet(max_iter=50000, random_state=SEED),
                   {'m__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0], 'm__l1_ratio': [0.1, 0.5, 0.9, 1.0]}),
    'Ridge':      (Ridge(), {'m__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100]}),
    'RandomForest': (RandomForestRegressor(n_estimators=300, random_state=SEED),
                     {'m__max_depth': [2, 3, None], 'm__min_samples_leaf': [1, 3, 5]}),
    'GBM':        (GradientBoostingRegressor(random_state=SEED, learning_rate=0.05),
                   {'m__n_estimators': [100, 300], 'm__max_depth': [1, 2, 3]}),
}


def policy_folds(groups, n_splits=5):
    """충전 정책 단위 fold 고정 (정책 이름순으로 0~4 fold 배정, 버전 무관 재현)"""
    pol = sorted(groups.unique())
    fold_of = {p: i % n_splits for i, p in enumerate(pol)}
    f = groups.map(fold_of).values
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def metrics(y_true, y_pred):
    e = y_pred - y_true
    return {'MAPE': np.mean(np.abs(e) / y_true) * 100,
            'MAE': np.mean(np.abs(e)),
            'RMSE': np.sqrt(np.mean(e ** 2)),
            'n': len(y_true)}


def run(features_csv, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(features_csv)
    tr = df[df['batch'] == 'b1'].reset_index(drop=True)
    tests = {'B2': df[df['batch'] == 'b2'], 'B3': df[df['batch'] == 'b3']}
    lo, hi = tr['cycle_life'].min(), tr['cycle_life'].max()
    groups = tr['policy']
    cv = policy_folds(groups, n_splits=5)
    print(f'train B1 n={len(tr)}, policies={groups.nunique()}, life range {lo:.0f}~{hi:.0f}')

    rows, preds, extra = [], [], []
    for fs_name, cols in FEATURE_SETS.items():
        X, y = tr[cols], np.log10(tr['cycle_life'])
        for m_name, (model, grid) in MODELS.items():
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc', StandardScaler()), ('m', model)])
            gs = GridSearchCV(pipe, grid or {'m__strategy': ['mean']}, cv=cv,
                              scoring='neg_mean_squared_error')
            gs.fit(X, y)
            best = gs.best_estimator_

            # B1 CV 성능 (선택된 하이퍼파라미터로 그룹 CV 예측)
            oof = 10 ** cross_val_predict(best, X, y, cv=cv)
            rows.append({'feature_set': fs_name, 'model': m_name, 'split': 'CV_B1',
                         **metrics(tr['cycle_life'].values, oof), 'params': str(gs.best_params_)})

            for t_name, te in tests.items():
                p = 10 ** best.predict(te[cols])
                yt = te['cycle_life'].values
                rows.append({'feature_set': fs_name, 'model': m_name, 'split': t_name,
                             **metrics(yt, p), 'params': str(gs.best_params_)})
                inside = (yt >= lo) & (yt <= hi)
                for tag, mask in [('inside', inside), ('outside_below', yt < lo), ('outside_above', yt > hi)]:
                    if mask.sum():
                        extra.append({'feature_set': fs_name, 'model': m_name, 'split': t_name,
                                      'range': tag, **metrics(yt[mask], p[mask])})
                preds.append(pd.DataFrame({'feature_set': fs_name, 'model': m_name, 'split': t_name,
                                           'cell_id': te['cell_id'].values, 'policy': te['policy'].values,
                                           'true': yt, 'pred': p}))

    res = pd.DataFrame(rows)
    res.loc[res['split'] == 'B2', 'GAP_vs_paper(%p)'] = res['MAPE'] - PAPER_MAPE
    res.round(2).to_csv(f'{out_dir}/model_performance.csv', index=False)
    pd.DataFrame(extra).round(2).to_csv(f'{out_dir}/extrapolation.csv', index=False)
    pr = pd.concat(preds)
    pr['ape(%)'] = (pr['pred'] - pr['true']).abs() / pr['true'] * 100
    pr.round(2).to_csv(f'{out_dir}/predictions.csv', index=False)
    plot_main(pr, lo, hi, out_dir)
    plot_relation(df, out_dir)
    return res


def plot_relation(df, out_dir):
    """B1에서 배운 직선(log 공간)을 범위 밖까지 연장해 B2 · B3와 비교"""
    tr = df[df['batch'] == 'b1']
    k, b = np.polyfit(tr['dq_var'], np.log10(tr['cycle_life']), 1)
    xs = np.linspace(df['dq_var'].min() - 0.1, df['dq_var'].max() + 0.1, 50)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for bt, col, lab in [('b1', '#1f77b4', 'B1 (train)'), ('b2', '#ff7f0e', 'B2'), ('b3', '#2ca02c', 'B3')]:
        d = df[df['batch'] == bt]
        ax.scatter(d['dq_var'], d['cycle_life'], s=18, color=col, label=lab)
    ax.plot(xs, 10 ** (k * xs + b), 'k--', lw=1.2, label='B1 fit, extended')
    ax.axvspan(tr['dq_var'].min(), tr['dq_var'].max(), color='#1f77b4', alpha=0.07)
    ax.set(yscale='log', xlabel='log10 Var(dQ)', ylabel='Cycle life (log scale)',
           title='B1 relation extended to B2 / B3')
    ax.legend(); plt.tight_layout(); plt.savefig(f'{out_dir}/relation_extended.png', dpi=120); plt.close()


def plot_main(pr, lo, hi, out_dir, fs='A_dq_var'):
    models = ['Dummy', 'ElasticNet', 'Ridge', 'RandomForest', 'GBM']
    fig, ax = plt.subplots(1, len(models), figsize=(22, 4.6), sharex=True, sharey=True)
    for a, m in zip(ax, models):
        for t, col in [('B2', '#ff7f0e'), ('B3', '#2ca02c')]:
            d = pr[(pr['feature_set'] == fs) & (pr['model'] == m) & (pr['split'] == t)]
            a.scatter(d['true'], d['pred'], s=16, color=col, label=t)
        a.plot([300, 2000], [300, 2000], 'k--', lw=1)
        a.axvspan(lo, hi, color='#1f77b4', alpha=0.08, label='B1 train range')
        a.set(title=m, xlabel='True cycle life', xlim=(300, 2000), ylim=(300, 2000))
    ax[0].set_ylabel('Predicted cycle life'); ax[0].legend(fontsize=8)
    plt.suptitle(f'Predicted vs true (feature set {fs})'); plt.tight_layout()
    plt.savefig(f'{out_dir}/pred_vs_true.png', dpi=120); plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='data/features.csv')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    r = run(a.features, a.out)
    show = r[r['split'] != 'CV_B1'].pivot_table(index=['feature_set', 'model'], columns='split', values='MAPE')
    show['CV_B1'] = r[r['split'] == 'CV_B1'].set_index(['feature_set', 'model'])['MAPE']
    print(show.round(1))
