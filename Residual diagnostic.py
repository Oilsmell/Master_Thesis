# -*- coding: utf-8 -*-
"""
Residual Magnitude Diagnostic
------------------------------------------------------------
DeepONet 학습에 들어가는 잔차의 크기를 (구조 × 위치 × 레벨) 별로 측정.
목적:
  (1) 2nd_location 잔차가 1st_location보다 체계적으로 큰지 확인
  (2) 특정 구조/레벨이 잔차를 키우는 주범인지 확인

* 학습 스크립트와 '완전히 동일한' 정규화/잔차 정의를 씁니다:
    res = (scaler_N.transform(damage) - scaler_N.transform(healthy)) * RESIDUAL_SCALE
"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === CFG (DeepONet_Unified.py와 동일하게 맞출 것) ===
BASE_A           = r"E:\git\benchmarktu1402-master\benchmarktu1402-master"
LOCATIONS        = ["state_1", "state_4"]
HEALTHY_LOCATION = "1st_location"
STRUCT_IDS       = list(range(0, 9))     # A0~A8
DAMAGE_LEVELS    = list(range(1, 11))    # D1~D10 = 10%~100%
WINDOW_SIZE      = 128
CHANNELS         = 8
SELECTED_NODES   = [3, 21, 39, 57, 63, 81, 99, 117]
RESIDUAL_SCALE   = 0.9


def _find_file(folder, must_contain, exts=(".dat", ".data")):
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if must_contain in f and f.lower().endswith(exts):
            return os.path.join(folder, f)
    return None

def _struct_folder(location, n):
    base = os.path.join(BASE_A, location)
    for name in (f"Case {n}", f"Case A{n}", f"CaseA{n}", f"Case_{n}"):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    return os.path.join(base, f"Case {n}")

def load_A(location, n, level=None):
    folder = _struct_folder(location, n)
    tag = "_H_" if level is None else f"_D{level}_"
    path = _find_file(folder, tag)
    if path is None:
        return None
    data = np.loadtxt(path)
    data = data[:, SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[SELECTED_NODES, :]
    ns = data.shape[1] // WINDOW_SIZE
    return data[:, :ns * WINDOW_SIZE].astype(np.float32)

def reshape_for_scaler(data):
    ns = data.shape[1] // WINDOW_SIZE
    return data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1), ns


def main():
    # 1) 구조별 healthy + 스케일러
    scalers, norm_h = {}, {}
    for n in STRUCT_IDS:
        raw = load_A(HEALTHY_LOCATION, n, None)
        if raw is None:
            print(f"[!] A{n} healthy 없음 -> 스킵"); continue
        resh, _ = reshape_for_scaler(raw)
        sc = MinMaxScaler((0, 1)).fit(resh)
        scalers[n], norm_h[n] = sc, sc.transform(resh)
    valid = list(norm_h.keys())

    # 2) (구조, 위치, 레벨)별 잔차 RMS
    table = {}                 # (loc, M) -> list of rms over structures
    by_loc = {loc: [] for loc in LOCATIONS}
    by_struct_loc = {}         # (n, loc) -> list of rms over levels
    print(f"\n{'struct':>6} {'loc':>13} {'level':>5} {'%':>4} {'rms_residual':>13} {'win':>6}")
    for n in valid:
        hN, sc = norm_h[n], scalers[n]
        ns_h = hN.shape[0]
        for loc in LOCATIONS:
            for M in DAMAGE_LEVELS:
                raw_d = load_A(loc, n, M)
                if raw_d is None:
                    continue
                resh_d, ns_d = reshape_for_scaler(raw_d)
                norm_d = sc.transform(resh_d)
                m = min(ns_h, ns_d)
                res = (norm_d[:m] - hN[:m]) * RESIDUAL_SCALE
                rms = float(np.sqrt(np.mean(res ** 2)))
                print(f"A{n:<5d} {loc:>13} {M:>5d} {M*10:>4d} {rms:>13.6f} {m:>6d}")
                table.setdefault((loc, M), []).append(rms)
                by_loc[loc].append(rms)
                by_struct_loc.setdefault((n, loc), []).append(rms)

    # 3) 위치별 평균 잔차 (핵심: 2nd가 1st보다 큰가?)
    print("\n=== 위치별 평균 잔차 RMS ===")
    for loc in LOCATIONS:
        v = by_loc[loc]
        if v:
            print(f"  {loc:>13}: mean={np.mean(v):.6f}  median={np.median(v):.6f}  max={np.max(v):.6f}  (n={len(v)})")
    if all(by_loc[l] for l in LOCATIONS) and len(LOCATIONS) == 2:
        r = np.mean(by_loc[LOCATIONS[1]]) / (np.mean(by_loc[LOCATIONS[0]]) + 1e-12)
        print(f"  -> 2nd/1st 평균 비율 = {r:.2f}배")

    # 4) 레벨×위치 평균 (어느 레벨이 큰가?)
    print("\n=== 레벨별 평균 잔차 RMS (위치별) ===")
    print(f"{'%':>4}", *[f"{loc:>13}" for loc in LOCATIONS])
    for M in DAMAGE_LEVELS:
        cells = []
        for loc in LOCATIONS:
            v = table.get((loc, M), [])
            cells.append(f"{np.mean(v):>13.6f}" if v else f"{'-':>13}")
        print(f"{M*10:>4}", *cells)

    # 5) 구조×위치 평균 (어느 구조가 큰가?)
    print("\n=== 구조별 평균 잔차 RMS (위치별) ===")
    print(f"{'struct':>6}", *[f"{loc:>13}" for loc in LOCATIONS])
    for n in valid:
        cells = []
        for loc in LOCATIONS:
            v = by_struct_loc.get((n, loc), [])
            cells.append(f"{np.mean(v):>13.6f}" if v else f"{'-':>13}")
        print(f"A{n:<5d}", *cells)


if __name__ == "__main__":
    main()