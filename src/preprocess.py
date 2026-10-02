"""데이터 추출 · 정제

- .mat(HDF5)에서 필요한 필드만 h5py로 읽어 배치별 pickle 캐시로 저장
- 원저자 로딩 코드 기준 제외 셀 처리
- summary 정제 규칙 적용

실행 : python src/preprocess.py --data_dir <.mat 폴더>
"""
import argparse
import gc
import os
import pickle

import h5py
import numpy as np
import pandas as pd

BATCH_FILES = {
    'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',  # 학습
    'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',  # 1차 평가
    'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',  # 2차 평가
}
KEEP_CYCLES = [2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EOL_AH = 0.88  # 공칭 1.1Ah의 80%

# 원저자 로딩 코드(rdbraatz/data-driven-prediction...) 기준 제외 셀
EXCLUDE_REASON = {
    **{c: 'EOL(80%) 도달 전 테스트 종료' for c in ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']},
    **{c: '다음 배치로 이어진 셀, 수명 불완전' for c in ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']},
    **{c: '노이즈 채널' for c in ['b3c2', 'b3c23', 'b3c32', 'b3c37', 'b3c42', 'b3c43']},
}


# ---------------------------------------------------------------- 추출
def _arr(f, ref):
    return np.asarray(f[ref][()], dtype=float).ravel()


def _is_empty(f, ref):
    d = f[ref]
    return bool(d.attrs.get('MATLAB_empty', 0)) or d.size < 10


def _str(f, ref):
    return ''.join(chr(int(c)) for c in f[ref][()].ravel())


def extract_batch(path, name):
    """필요한 필드만 골라 읽기 (파일 전체를 메모리에 올리지 않음)"""
    out = []
    with h5py.File(path, 'r') as f:
        b = f['batch']
        for i in range(b['summary'].shape[0]):
            try:
                s = f[b['summary'][i, 0]]
                summ = {k: np.asarray(s[k][()], dtype=float).ravel() for k in s.keys()}
                cyc = f[b['cycles'][i, 0]]
                n_cyc = cyc['Qdlin'].shape[0]
                # 사이클 번호 -> 인덱스 (summary['cycle'] 기준, B1은 cycle 1 미포함)
                cyc_no = summ.get('cycle', np.arange(1, n_cyc + 1))
                idx_of = ({int(v): j for j, v in enumerate(cyc_no[:n_cyc])}
                          if len(cyc_no) >= n_cyc else {j + 1: j for j in range(n_cyc)})
                curves = {}
                for c in KEEP_CYCLES:
                    j = idx_of.get(c)
                    if j is not None and not _is_empty(f, cyc['Qdlin'][j, 0]):
                        curves[c] = {'Qdlin': _arr(f, cyc['Qdlin'][j, 0])}
                life = _arr(f, b['cycle_life'][i, 0])
                out.append({
                    'batch': name, 'cell_id': f'{name}c{i}',
                    'policy': _str(f, b['policy_readable'][i, 0]),
                    'cycle_life': float(life[0]) if life.size else np.nan,
                    'n_cycles': n_cyc, 'summary': summ, 'curves': curves,
                    'Vdlin': _arr(f, b['Vdlin'][i, 0]),
                })
            except Exception as e:  # 셀 하나가 실패해도 전체는 계속
                print(f'  {name}c{i} skip : {e}')
    gc.collect()
    return out


def build_cache(data_dir, cache_dir='data/cache'):
    os.makedirs(cache_dir, exist_ok=True)
    for name, fname in BATCH_FILES.items():
        pkl = os.path.join(cache_dir, f'{name}.pkl')
        if os.path.exists(pkl):
            print(f'{name}: cache exists -> skip')
            continue
        data = extract_batch(os.path.join(data_dir, fname), name)
        with open(pkl, 'wb') as fp:
            pickle.dump(data, fp)
        print(f'{name}: {len(data)} cells saved')


# ---------------------------------------------------------------- 로드 · 정제
def load_cells(cache_dir='data/cache'):
    """캐시 로드 후 분석 대상 셀만 반환 (제외 셀, 수명 없음, ΔQ 계산 불가 셀 제거)"""
    cells = []
    for name in BATCH_FILES:
        with open(os.path.join(cache_dir, f'{name}.pkl'), 'rb') as fp:
            cells += pickle.load(fp)
    keep = []
    for c in cells:
        if c['cell_id'] in EXCLUDE_REASON:
            continue
        if not np.isfinite(c['cycle_life']):          # B2 VarCharge / SLOWCYCLE 등
            continue
        if 10 not in c['curves'] or 100 not in c['curves']:
            continue
        keep.append(c)
    return keep


def clean_summary(cell):
    """사이클별 summary를 DataFrame으로 만들고 정제 규칙 적용"""
    s = cell['summary']
    n = len(s['QDischarge'])
    get = lambda k: s.get(k, np.full(n, np.nan))
    d = pd.DataFrame({
        'cycle': s['cycle'] if len(s.get('cycle', [])) == n else np.arange(1, n + 1),
        'QD': s['QDischarge'], 'QC': get('QCharge'), 'IR': get('IR'),
        'Tavg': get('Tavg'), 'Tmax': get('Tmax'), 'Tmin': get('Tmin'),
        'chargetime': get('chargetime'),
    })
    d = d[(d['QD'] > 0) & d['QD'].between(0.8, 1.3)].copy()          # 0값 · 용량 스파이크
    d.loc[~d['chargetime'].between(5, 60), 'chargetime'] = np.nan     # 충전시간 이상값
    d.loc[d['IR'] <= 0, 'IR'] = np.nan
    return d.sort_values('cycle')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--cache_dir', default='data/cache')
    a = ap.parse_args()
    build_cache(a.data_dir, a.cache_dir)
    cells = load_cells(a.cache_dir)
    print(pd.Series([c['batch'] for c in cells]).value_counts().sort_index())
