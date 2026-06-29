# -*- coding: utf-8 -*-
"""
Unified DeepONet Training Pipeline
------------------------------------------------------------
- 기존: Case A1~A8 각각에 대해 DeepONet을 따로 학습.
- 변경: 9개 구조(A0~A8) × 2개 손상 위치(1st/2nd) × 10개 강성저감 레벨(D1~D10)
        의 잔차를 하나의 풀로 모아 '단일 DeepONet' 1회 학습.

핵심 설계 (피드백 반영):
  * 조건 변수 = Stiffness Reduction 하나로 통일.  D{M} -> M*10%  (0.1 ~ 1.0)
  * 위치(1st/2nd)는 입력 차원에 넣지 않음. 같은 강성저감 라벨을 공유하며
    학습 다양성으로만 섞임.
  * healthy는 위치와 무관하므로 1st_location에서만 로드.
  * 구조물 B의 실제 손상(D3_*)은 '검증용'이라 학습 풀에 넣지 않음.
  * 샘플 subsample 없음(요청). 필요 시 MAX_SAMPLES_PER_BLOCK로 제어.
"""

import os, gc, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# 0. Config  (★ 경로/하이퍼파라미터는 여기서만 수정)
# =========================================================
class CFG:
    # --- A 구조 경로 ---
    BASE_A           = r"E:\git\benchmarktu1402-master\benchmarktu1402-master\compare"
    LOCATIONS        = ["state_1", "state_4"]   # 손상 위치 2곳
    HEALTHY_LOCATION = "1st_location"                     # healthy는 여기서만 로드
    STRUCT_IDS       = list(range(0, 5))                  # A0 ~ A8 (총 9개)
    DAMAGE_LEVELS    = list(range(1, 11))                 # D1~D10 = 10%~100%

    # --- B 구조 경로 ---
    DIR_B    = r"E:\2ndstructuredata\raw data"
    FILE_B_H = "healthyclean.txt"

    # --- 출력 ---
    SAVE_DIR     = r"E:\2ndstructuredata\Unified_DeepONet1"
    SYNTH_SUBDIR = "Synthetic_B_Data"

    # --- 신호 형식 ---
    WINDOW_SIZE    = 128
    CHANNELS       = 8
    SELECTED_NODES = [3, 21, 39, 57, 63, 81, 99, 117]    # A .dat 에서 뽑는 8개 노드

    # --- 학습 ---
    LATENT_DIM     = 8
    AE_EPOCHS      = 500
    DON_EPOCHS     = 500
    BATCH_SIZE     = 128
    LR             = 1e-3
    LAMBDA_MMD     = 0.3
    RESIDUAL_SCALE = 0.9

    # --- 생성 (조건=강성저감 0~1, 0.02 간격) ---
    SYNTHETIC_STEPS = np.round(np.arange(0.00, 1.01, 0.02), 2)

    # --- 블록(구조×위치×레벨)별 샘플 상한. None=전부 사용 ---
    MAX_SAMPLES_PER_BLOCK = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()


# =========================================================
# 1. Models  (기존 DeepONet.py 정의 그대로)
# =========================================================
def rbf_kernel(x, y, gamma=1.0):
    x = x.unsqueeze(1); y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss(x, y, gamma=1.0):
    return rbf_kernel(x, x, gamma).mean() + rbf_kernel(y, y, gamma).mean() \
           - 2 * rbf_kernel(x, y, gamma).mean()

class Autoencoder(nn.Module):
    def __init__(self, input_dim=cfg.WINDOW_SIZE * cfg.CHANNELS, latent_dim=cfg.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 256), nn.SiLU(),
            nn.Linear(256, 512), nn.SiLU(),
            nn.Linear(512, input_dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class FourierFeature(nn.Module):
    def __init__(self, input_dim, mapping_size=128, scale=10):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    def forward(self, x):
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class DeepONet(nn.Module):
    def __init__(self, branch_dim=cfg.LATENT_DIM + 1, trunk_dim=2, hidden_dim=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.fourier = FourierFeature(trunk_dim)
        self.trunk = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, branch_in, trunk_in):
        Bm = self.branch(branch_in)
        Tm = self.trunk(self.fourier(trunk_in))
        return torch.matmul(Bm, Tm.T) + self.bias


# =========================================================
# 2. Data loading  (폴더명/확장자에 강건한 glob 방식)
# =========================================================
def _find_file(folder, must_contain, exts=(".dat", ".data")):
    """folder 안에서 must_contain 문자열을 포함하는 첫 파일 경로 반환."""
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if must_contain in f and f.lower().endswith(exts):
            return os.path.join(folder, f)
    return None

def _struct_folder(location, n):
    """구조 N 폴더. 'Case N' / 'Case A{N}' 등 후보를 시도."""
    base = os.path.join(cfg.BASE_A, location)
    for name in (f"Case {n}", f"Case A{n}", f"CaseA{n}", f"Case_{n}"):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    return os.path.join(base, f"Case {n}")  # 기본값(없으면 이후 경고)

def load_A(location, n, level=None):
    """구조 A{n}, 위치 location 에서 healthy(level=None) 또는 D{level} 로드 -> (8, T)."""
    folder = _struct_folder(location, n)
    tag = "_H_" if level is None else f"_D{level}_"
    path = _find_file(folder, tag)
    if path is None:
        print(f"  [!] 파일 없음: {folder}  (tag={tag})")
        return None
    data = np.loadtxt(path)
    data = data[:, cfg.SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[cfg.SELECTED_NODES, :]
    ns = data.shape[1] // cfg.WINDOW_SIZE
    return data[:, :ns * cfg.WINDOW_SIZE].astype(np.float32)

def load_B_healthy():
    path = os.path.join(cfg.DIR_B, cfg.FILE_B_H)
    raw = [float(l.split()[1]) for l in open(path) if len(l.split()) >= 2]
    data = np.array(raw, dtype=np.float32).reshape(cfg.CHANNELS, -1)
    ns = data.shape[1] // cfg.WINDOW_SIZE
    return data[:, :ns * cfg.WINDOW_SIZE]

def reshape_for_scaler(data):
    ns = data.shape[1] // cfg.WINDOW_SIZE
    return data.reshape(cfg.CHANNELS, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1), ns


# =========================================================
# 3. Main pipeline
# =========================================================
def main():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    synth_dir = os.path.join(cfg.SAVE_DIR, cfg.SYNTH_SUBDIR)
    os.makedirs(synth_dir, exist_ok=True)
    print(f"Device: {cfg.device}")

    # ---- Phase 1: 모든 healthy 로드 + 구조별 스케일러 ----
    print("\n=== Phase 1: Healthy 로드 & 정규화 (구조별 스케일러) ===")
    scalers, norm_h = {}, {}
    for n in cfg.STRUCT_IDS:
        raw = load_A(cfg.HEALTHY_LOCATION, n, None)
        if raw is None:
            print(f"  [!] A{n} healthy 없음 -> 이 구조 스킵")
            continue
        resh, ns = reshape_for_scaler(raw)
        sc = MinMaxScaler((0, 1)).fit(resh)
        scalers[n], norm_h[n] = sc, sc.transform(resh)
        print(f"  A{n}: healthy windows = {ns}")

    raw_hB = load_B_healthy()
    resh_B, nsB = reshape_for_scaler(raw_hB)
    scaler_B = MinMaxScaler((0, 1)).fit(resh_B)
    norm_hB = scaler_B.transform(resh_B)
    print(f"  B : healthy windows = {nsB}")

    valid_ids = list(norm_h.keys())
    if not valid_ids:
        print("유효한 A 구조가 없습니다. 경로/폴더명 확인 필요."); return

    # ---- Phase 2: 단일 Autoencoder (전 구조 healthy + B, MMD: pooled-A vs B) ----
    print("\n=== Phase 2: Autoencoder 학습 (전 구조 + B 공통 잠재공간) ===")
    all_A = np.vstack([norm_h[n] for n in valid_ids]).astype(np.float32)
    ae = Autoencoder().to(cfg.device)
    opt_ae = optim.Adam(ae.parameters(), lr=cfg.LR)
    recon = nn.MSELoss()
    loader_A = DataLoader(TensorDataset(torch.FloatTensor(all_A)),
                          batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    loader_B = DataLoader(TensorDataset(torch.FloatTensor(norm_hB)),
                          batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    for ep in range(cfg.AE_EPOCHS):
        ae.train(); tot = 0.0
        itB = iter(loader_B)
        for (bA,) in loader_A:
            try:
                (bB,) = next(itB)
            except StopIteration:
                itB = iter(loader_B); (bB,) = next(itB)
            bA, bB = bA.to(cfg.device), bB.to(cfg.device)
            opt_ae.zero_grad()
            rA, zA = ae(bA); rB, zB = ae(bB)
            loss = recon(rA, bA) + recon(rB, bB) + cfg.LAMBDA_MMD * mmd_loss(zA, zB)
            loss.backward(); opt_ae.step(); tot += loss.item()
        if (ep + 1) % 100 == 0:
            print(f"   AE {ep+1}/{cfg.AE_EPOCHS} | loss {tot/len(loader_A):.6f}")
    ae.eval()

    # ---- Phase 3: 통합 학습 풀 구성 (9구조 × 2위치 × 10레벨) ----
    print("\n=== Phase 3: DeepONet 학습 풀 구성 ===")
    with torch.no_grad():
        z_cache = {n: ae.encoder(torch.FloatTensor(norm_h[n]).to(cfg.device)).cpu().numpy()
                   for n in valid_ids}

    X_branch, Y_target, n_blocks = [], [], 0
    for n in valid_ids:
        zN, hN, sc = z_cache[n], norm_h[n], scalers[n]
        ns_h = hN.shape[0]
        for loc in cfg.LOCATIONS:
            for M in cfg.DAMAGE_LEVELS:
                raw_d = load_A(loc, n, M)
                if raw_d is None:
                    continue
                resh_d, ns_d = reshape_for_scaler(raw_d)
                norm_d = sc.transform(resh_d)                 # 동일 구조 스케일러로 잔차 계산
                m = min(ns_h, ns_d)
                res  = (norm_d[:m] - hN[:m]) * cfg.RESIDUAL_SCALE
                cond = np.full((m, 1), M * 0.1, dtype=np.float32)   # ★ Stiffness Reduction
                bin_ = np.hstack([zN[:m], cond]).astype(np.float32)
                if cfg.MAX_SAMPLES_PER_BLOCK and m > cfg.MAX_SAMPLES_PER_BLOCK:
                    r = np.random.RandomState(42).choice(m, cfg.MAX_SAMPLES_PER_BLOCK, replace=False)
                    bin_, res = bin_[r], res[r]
                X_branch.append(bin_); Y_target.append(res.astype(np.float32))
                n_blocks += 1
    if not X_branch:
        print("학습 샘플이 없습니다. 손상 파일 경로 확인 필요."); return
    X_branch = np.vstack(X_branch); Y_target = np.vstack(Y_target)
    print(f"  블록 수: {n_blocks} (이상적 {len(valid_ids)*len(cfg.LOCATIONS)*len(cfg.DAMAGE_LEVELS)})")
    print(f"  총 샘플: {X_branch.shape[0]:,} | target dim: {Y_target.shape[1]}")

    # ---- Phase 4: 단일 DeepONet 학습 ----
    print("\n=== Phase 4: 단일 DeepONet 학습 ===")
    T_grid, X_grid = np.meshgrid(np.linspace(0, 1, cfg.WINDOW_SIZE),
                                 np.linspace(0, 1, cfg.CHANNELS))
    trunk_in = torch.FloatTensor(np.stack([T_grid.flatten(), X_grid.flatten()], 1)).to(cfg.device)

    don = DeepONet(branch_dim=cfg.LATENT_DIM + 1, trunk_dim=2).to(cfg.device)
    opt = optim.Adam(don.parameters(), lr=cfg.LR)
    sch = optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_branch), torch.FloatTensor(Y_target)),
                        batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for ep in range(cfg.DON_EPOCHS):
        don.train(); tot = 0.0
        for bin_, tgt in loader:
            opt.zero_grad()
            pred = don(bin_.to(cfg.device), trunk_in)
            loss = nn.MSELoss()(pred, tgt.to(cfg.device))
            loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        if (ep + 1) % 50 == 0:
            print(f"   DON {ep+1}/{cfg.DON_EPOCHS} | loss {tot/len(loader):.6f} | {time.time()-t0:.0f}s")
    print(f"  총 학습 시간: {time.time()-t0:.1f}s")

    torch.save(don.state_dict(), os.path.join(cfg.SAVE_DIR, "deeponet_unified.pt"))
    torch.save(ae.state_dict(),  os.path.join(cfg.SAVE_DIR, "autoencoder_unified.pt"))

    # ---- Phase 5: 구조물 B 합성 데이터 생성 ----
    print("\n=== Phase 5: B 합성 데이터 생성 ===")
    with torch.no_grad():
        z_B = ae.encoder(torch.FloatTensor(norm_hB).to(cfg.device)).cpu().numpy()
    don.eval()
    for s in cfg.SYNTHETIC_STEPS:
        if s == 0.0:
            # 강성저감 0% = healthy 그대로 (다운스트림 baseline용 깨끗한 기준)
            gen_phys = scaler_B.inverse_transform(norm_hB)
        else:
            bin_ = torch.FloatTensor(np.hstack([z_B, np.full((len(norm_hB), 1), s)])).to(cfg.device)
            with torch.no_grad():
                pred_res = don(bin_, trunk_in).cpu().numpy() / cfg.RESIDUAL_SCALE
            gen_phys = scaler_B.inverse_transform(norm_hB + pred_res)
        gen = gen_phys.reshape(len(norm_hB), cfg.CHANNELS, cfg.WINDOW_SIZE) \
                      .transpose(1, 0, 2).reshape(cfg.CHANNELS, -1)
        np.savetxt(os.path.join(synth_dir, f"Synthetic_B_DI_{s:.2f}.txt"),
                   gen.T, fmt='%.6e', delimiter='\t')
    print(f"  저장 완료: {synth_dir} ({len(cfg.SYNTHETIC_STEPS)}개 파일)")
    gc.collect()


if __name__ == "__main__":
    main()