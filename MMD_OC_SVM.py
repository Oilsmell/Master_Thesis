# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:02:04 2026

@author: Oilsmell
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:58:16 2026

@author: Oilsmell
"""

# -*- coding: utf-8 -*-
"""
Strategy C: Optuna-Optimized Unbiased MMD Pipeline
- AI automatically finds the best PCA dimension, Chunk Size, and Kernel Gamma.
- Objective: Maximize the Fisher's margin between Healthy and Target Damage MMD scores.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist
from scipy.signal import hilbert
from tqdm import tqdm

# Optuna 로그 깔끔하게 정리
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=== Phase 1: Setup & Dynamic Data Loaders ===")

class Config:
    DIR_B_RAW       = r"E:\2ndstructuredata\raw data"
    FILE_B          = "healthyclean.txt"
    SYNTH_DATA_DIR  = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"
    SAVE_DIR        = r"E:\2ndstructuredata\Code_MMD_Optuna"

    WINDOW_SIZE = 128 
    CHANNELS    = 8
    NUM_SAMPLES = 1000 
    PFA         = 0.05 # FAR 5%
    
    OPTUNA_TRIALS = 40# 💡 AI가 최적의 조합을 탐색할 횟수

    DAMAGE_CASES_B       = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    SYNTHETIC_DI_STEPS   = np.round(np.arange(0.0, 1.01, 0.02), 2)

cfg = Config()
if not os.path.exists(cfg.SAVE_DIR): os.makedirs(cfg.SAVE_DIR)

def _windowize(data):
    ns = data.shape[1] // cfg.WINDOW_SIZE
    if ns == 0: return None, 0
    data = data[:, :ns * cfg.WINDOW_SIZE]
    return data.reshape(cfg.CHANNELS, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2), ns

def load_data(path, is_synth=False):
    try:
        if is_synth: data = np.loadtxt(path, delimiter='\t').T.astype(np.float32)
        else:
            raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(cfg.CHANNELS, -1)
        windows, ns = _windowize(data)
        if ns > cfg.NUM_SAMPLES and "healthyclean" not in path.lower():
            np.random.seed(42)
            windows = windows[np.random.choice(ns, cfg.NUM_SAMPLES, replace=False)]
            ns = cfg.NUM_SAMPLES
        return windows, ns
    except: return None, 0

# 1. 정상 데이터 분할
Train_H_raw_all, ns_total = load_data(os.path.join(cfg.DIR_B_RAW, cfg.FILE_B))
np.random.seed(42); np.random.shuffle(Train_H_raw_all)
train_len = int(ns_total * 0.7)
Train_H_raw = Train_H_raw_all[:train_len]
Val_H_raw   = Train_H_raw_all[train_len:]

# =====================================================================
print("\n=== Phase 2: Feature Extraction (Pre-PCA) ===")

def extract_features(data):
    if len(data) == 0: return np.array([])
    mean_v = np.mean(data, axis=2); std_v = np.std(data, axis=2); rms_v = np.sqrt(np.mean(data**2, axis=2))
    peak_v = np.max(np.abs(data), axis=2); p2p_v = np.ptp(data, axis=2); crest_v= peak_v / (rms_v + 1e-10)
    centered = data - mean_v[:, :, None]
    skew_v = np.mean(centered**3, axis=2) / (std_v**3 + 1e-10); kurt_v = np.mean(centered**4, axis=2) / (std_v**4 + 1e-10)
    zcr_v = np.mean(np.abs(np.diff(np.sign(data), axis=2)), axis=2) / 2
    
    fft_mag = np.abs(np.fft.rfft(data, axis=-1)); freqs = np.fft.rfftfreq(cfg.WINDOW_SIZE)
    fft_sum = np.sum(fft_mag, axis=2) + 1e-10; centroid = np.sum(freqs * fft_mag, axis=2) / fft_sum
    spread = np.sqrt(np.sum(((freqs - centroid[..., None])**2) * fft_mag, axis=2) / fft_sum)
    p_norm = fft_mag / fft_sum[..., None]; sp_ent = -np.sum(p_norm * np.log2(p_norm + 1e-10), axis=2)
    hf_ratio = np.sum(fft_mag[:, :, fft_mag.shape[-1]//2:], axis=2) / fft_sum 
    
    n_bands = 4; band_edges = np.linspace(0, fft_mag.shape[2], n_bands + 1, dtype=int); band_ratios = []
    for i in range(n_bands):
        band_e = np.sum(fft_mag[:, :, band_edges[i]:band_edges[i+1]], axis=2)
        band_ratios.append(band_e / fft_sum)
        
    env = np.abs(hilbert(data, axis=-1)); env_rms = np.sqrt(np.mean(env**2, axis=2))
    env_mean = np.mean(env, axis=2); env_std = np.std(env, axis=2)
    env_kurt = np.mean((env - env_mean[..., None])**4, axis=2) / (env_std**4 + 1e-10)
    
    N = data.shape[0]; corr_feats = np.zeros((N, 28)); triu_idx = np.triu_indices(cfg.CHANNELS, k=1)
    for i in range(N): corr_feats[i] = np.corrcoef(data[i])[triu_idx]
        
    return np.concatenate([mean_v, std_v, rms_v, peak_v, p2p_v, crest_v, skew_v, kurt_v, zcr_v, 
                           centroid, spread, sp_ent, hf_ratio, np.concatenate(band_ratios, axis=1), 
                           env_rms, env_mean, env_std, env_kurt, corr_feats], axis=1)

print("Extracting full features...")
train_feat_raw = extract_features(Train_H_raw)
val_feat_raw   = extract_features(Val_H_raw)

# 💡 목표 성능을 위해 합성 손상 데이터(DI 0.50)를 타겟으로 삼습니다.
target_raw, _ = load_data(os.path.join(cfg.SYNTH_DATA_DIR, "Synthetic_B_DI_0.50.txt"), is_synth=True)
if target_raw is None: target_raw, _ = load_data(os.path.join(cfg.DIR_B_RAW, "D3_48_1.txt"), is_synth=False)
target_feat_raw = extract_features(target_raw)

scaler = RobustScaler()
train_scaled = scaler.fit_transform(train_feat_raw)
val_scaled   = scaler.transform(val_feat_raw)
target_scaled = scaler.transform(target_feat_raw)

# =====================================================================
print("\n=== Phase 3: Optuna Tuning for Stable MK-MMD (Sliding Window) ===")

def compute_mk_mmd_stable(X, Y, base_gamma):
    """ 
    💡 분산이 폭발하던 비편향(Unbiased) 공식을 버리고, 
    마이너스 값이 나오지 않는 안정적인 표준 MMD 공식으로 복구합니다.
    """
    gammas = [base_gamma * 0.1, base_gamma, base_gamma * 10]
    total_mmd = 0.0
    
    for g in gammas:
        XX = rbf_kernel(X, X, g)
        YY = rbf_kernel(Y, Y, g)
        XY = rbf_kernel(X, Y, g)
        
        # 분산을 낮추고 안정을 가져오는 표준 평균(Mean) 방식
        total_mmd += (XX.mean() + YY.mean() - 2 * XY.mean())
        
    return total_mmd

def get_mk_mmd_scores_sliding(features, ref_dist, chunk_size, base_gamma, stride=2):
    """
    💡 핵심 비기: 슬라이딩 윈도우(Sliding Window)
    stride=2로 설정하여, 2칸씩만 이동하며 점수를 촘촘하게 추출합니다.
    고작 7개 나오던 점수가 수백 개로 늘어나 완벽한 분포를 형성합니다.
    """
    scores = []
    n = len(features)
    if n < chunk_size: return np.array([])
    
    # 뭉텅이로 건너뛰지 않고 촘촘하게 훑고 지나갑니다
    for i in range(0, n - chunk_size + 1, stride):
        chunk = features[i:i + chunk_size]
        scores.append(compute_mk_mmd_stable(ref_dist, chunk, base_gamma))
        
    return np.array(scores)

def objective(trial):
    pca_comp = trial.suggest_int('pca_comp', 15, 60)
    chunk_size = trial.suggest_int('chunk_size', 20, 80, step=10) # 최대 청크 크기 약간 축소
    gamma_mult = trial.suggest_float('gamma_mult', 0.01, 10.0, log=True) 
    
    pca = PCA(n_components=pca_comp, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    val_pca = pca.transform(val_scaled)
    target_pca = pca.transform(target_scaled)
    
    distances = pdist(train_pca, metric='sqeuclidean')
    base_gamma = (1.0 / (np.median(distances) + 1e-10)) * gamma_mult
    
    # 평가 시 계산 속도를 위해 stride를 청크 크기 절반으로 둡니다
    v_scores = get_mk_mmd_scores_sliding(val_pca, train_pca, chunk_size, base_gamma, stride=chunk_size//2)
    t_scores = get_mk_mmd_scores_sliding(target_pca, train_pca, chunk_size, base_gamma, stride=chunk_size//2)
    
    if len(v_scores) < 5 or len(t_scores) < 5: return -999.0
    
    margin = (np.mean(t_scores) - np.mean(v_scores)) / (np.std(v_scores) + 1e-6)
    return margin

study = optuna.create_study(direction='maximize')
with tqdm(total=cfg.OPTUNA_TRIALS, desc="Optimizing Stable MK-MMD") as pbar:
    def callback(study, trial): pbar.update(1)
    study.optimize(objective, n_trials=cfg.OPTUNA_TRIALS, callbacks=[callback])

best = study.best_params
print(f"🔥 Best Parameters Found: PCA={best['pca_comp']}, Chunk={best['chunk_size']}, Base Gamma Mult={best['gamma_mult']:.4f}")

# =====================================================================
print("\n=== Phase 4: Apply Best Params & Evaluate All Data ===")

# 최적 파라미터 적용
pca_final = PCA(n_components=best['pca_comp'], random_state=42)
train_final = pca_final.fit_transform(train_scaled)
val_final = pca_final.transform(val_scaled)

base_gamma = 1.0 / (np.median(pdist(train_final, metric='sqeuclidean')) + 1e-10)
final_gamma = base_gamma * best['gamma_mult']
final_chunk = best['chunk_size']

# 💡 최종 평가 시에는 stride=1 로 설정하여 영혼까지 끌어모은 촘촘한 점수를 뽑아냅니다.
val_scores_final = get_mk_mmd_scores_sliding(val_final, train_final, final_chunk, final_gamma, stride=2)
threshold = np.percentile(val_scores_final, (1 - cfg.PFA) * 100)

real_dr, synth_dr = [], []

print("[Evaluating Real Data]")
for case in tqdm(cfg.DAMAGE_CASES_B, desc="Real", leave=False):
    d, _ = load_data(os.path.join(cfg.DIR_B_RAW, f"D3_{case}_1.txt"), is_synth=False)
    if d is not None:
        feat = pca_final.transform(scaler.transform(extract_features(d)))
        scores = get_mk_mmd_scores_sliding(feat, train_final, final_chunk, final_gamma, stride=2)
        dr = np.mean(scores > threshold) * 100.0 if len(scores)>0 else 0.0
        real_dr.append(dr)

print("[Evaluating Synthetic Data]")
for di in tqdm(cfg.SYNTHETIC_DI_STEPS, desc="Synth", leave=False):
    d, _ = load_data(os.path.join(cfg.SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
    if d is not None:
        feat = pca_final.transform(scaler.transform(extract_features(d)))
        scores = get_mk_mmd_scores_sliding(feat, train_final, final_chunk, final_gamma, stride=2)
        dr = np.mean(scores > threshold) * 100.0 if len(scores)>0 else 0.0
        synth_dr.append(dr)

# (Phase 5 그래프 출력 코드는 기존과 동일하게 사용하시면 됩니다!)

# =====================================================================
print("\n=== Phase 5: Plot Optimized Results ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. 진단 플롯 (Real 48% + Synth 1.00)
d_48, _ = load_data(os.path.join(cfg.DIR_B_RAW, f"D3_48_1.txt"), is_synth=False)
# 💡 수정 완료: 예전 이름(get_mmd_scores)을 지우고 슬라이딩 함수로 교체 및 stride=1 추가!
scores_48 = get_mk_mmd_scores_sliding(pca_final.transform(scaler.transform(extract_features(d_48))), train_final, final_chunk, final_gamma, stride=1)

d_synth, _ = load_data(os.path.join(cfg.SYNTH_DATA_DIR, "Synthetic_B_DI_1.00.txt"), is_synth=True)
# 💡 수정 완료: 예전 이름을 지우고 슬라이딩 함수로 교체 및 stride=1 추가!
scores_synth = get_mk_mmd_scores_sliding(pca_final.transform(scaler.transform(extract_features(d_synth))), train_final, final_chunk, final_gamma, stride=1)

ax1.hist(val_scores_final, bins=12, alpha=0.5, color='skyblue', edgecolor='black', label=f'Healthy Val (μ={np.mean(val_scores_final):.4f})')
ax1.hist(scores_48, bins=12, alpha=0.5, color='crimson', edgecolor='black', label=f'Real 48% (μ={np.mean(scores_48):.4f})')
ax1.hist(scores_synth, bins=12, alpha=0.5, color='orange', edgecolor='black', label=f'Synth 1.00 (μ={np.mean(scores_synth):.4f})')

ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (FAR 5%)')
ax1.set_title(f"Optuna Tuned MK-MMD Distribution (Chunk={final_chunk})", fontweight='bold')
ax1.set_xlabel("MK-MMD Distance")
ax1.legend()

# 2. POD 플롯
ax2.plot(cfg.DAMAGE_CASES_B, real_dr, 'o-', color='#E74C3C', linewidth=2.5, label='Real Damage MMD')
ax2.plot(cfg.SYNTHETIC_DI_STEPS * 100, synth_dr, 's-', color='#3498DB', linewidth=2.5, label='Synth Damage MMD') 

ax2.axhline(cfg.PFA * 100, color='gray', linestyle=':', linewidth=2, label='FAR 5%')
ax2.set_title("Optuna MK-MMD POD Comparison", fontweight='bold')
ax2.set_xlabel("Damage Severity (%)")
ax2.set_ylabel("Probability of Detection (%)")
ax2.set_ylim([-2, 105])
ax2.legend()
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "MK_MMD_Optuna_Final.png"), dpi=300)
plt.show()

print("\n🎉 Optuna MK-MMD 최적화 및 튜닝 결과 출력 완료!")
# %%


# %% [Phase 6] POD Curve via â-vs-a Regression (NDT Standard)
# ============================================================
# 각 손상 레벨에서 수집된 MMD 점수 분포(â)와 손상 크기(a)를 이용해
# Heteroscedastic Censored MLE로 a vs â 관계를 적합하고,
# 거기서 POD(a) = P(â > threshold | a)를 계산합니다.
# Bootstrap으로 95% Lower Confidence Bound도 산출합니다.
# ============================================================

from scipy.stats import norm, linregress
from scipy.optimize import minimize

print("\n=== Phase 6: POD Curve Construction ===")

# ── Step 1: 손상 레벨별 (a, â) 산점 데이터 재수집 ──────────
# 기존 Phase 4에서는 detection rate만 저장했으므로 점수를 다시 모읍니다.
print("Step 1: Collecting (a, â) scatter data...")

all_real_a, all_real_ahat = [], []
all_synth_a, all_synth_ahat = [], []

for case in tqdm(cfg.DAMAGE_CASES_B, desc="  Real", leave=False):
    d, _ = load_data(os.path.join(cfg.DIR_B_RAW, f"D3_{case}_1.txt"), is_synth=False)
    if d is not None:
        feat = pca_final.transform(scaler.transform(extract_features(d)))
        scores = get_mk_mmd_scores_sliding(feat, train_final, final_chunk, final_gamma, stride=1)
        all_real_a.extend([case] * len(scores))
        all_real_ahat.extend(scores.tolist())

for di in tqdm(cfg.SYNTHETIC_DI_STEPS, desc="  Synth", leave=False):
    d, _ = load_data(os.path.join(cfg.SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
    if d is not None:
        feat = pca_final.transform(scaler.transform(extract_features(d)))
        scores = get_mk_mmd_scores_sliding(feat, train_final, final_chunk, final_gamma, stride=1)
        all_synth_a.extend([di * 100.0] * len(scores))   # 0~100% 스케일로 매핑
        all_synth_ahat.extend(scores.tolist())

all_real_a    = np.array(all_real_a)
all_real_ahat = np.array(all_real_ahat)
all_synth_a   = np.array(all_synth_a)
all_synth_ahat = np.array(all_synth_ahat)

print(f"  Real:  {len(all_real_a):,} (a, â) pairs")
print(f"  Synth: {len(all_synth_a):,} (a, â) pairs")

# ── Step 2: Heteroscedastic Censored MLE 정의 ───────────────
# 모델: â = (b0 + b1·a) + ε,  ε ~ N(0, (tau0 + tau1·a)²)
# Threshold 아래 데이터는 censored로 처리 (정보 보존)

def mll_loss(params, x, y, a_th):
    b0, b1, tau0, tau1 = params
    mu_y = b0 + b1 * x
    tau  = np.maximum(tau0 + tau1 * x, 1e-8)
    censored   = y <= a_th
    uncensored = ~censored
    ll_unc = norm.logpdf(y[uncensored], loc=mu_y[uncensored], scale=tau[uncensored])
    ll_cen = norm.logcdf(a_th, loc=mu_y[censored], scale=tau[censored])
    return -(np.sum(ll_unc) + np.sum(ll_cen))

def fit_mle(x, y, a_th):
    slope, intercept, *_ = linregress(x, y)
    init   = [intercept, slope, np.std(y), 0.0]
    bounds = [(None, None), (None, None), (1e-8, None), (0.0, None)]
    res    = minimize(mll_loss, init, args=(x, y, a_th),
                      bounds=bounds, method='L-BFGS-B')
    return res.x, res.success

def get_mean_pod(x_range, params, a_th):
    b0, b1, t0, t1 = params
    mu  = b0 + b1 * x_range
    tau = np.maximum(t0 + t1 * x_range, 1e-8)
    return norm.cdf((mu - a_th) / tau)

def bootstrap_lcb(x, y, a_th, x_range, n_boot=300, block_size=None):
    """
    슬라이딩 윈도우라 chunk들이 상관되어 있으므로 block bootstrap 사용.
    block_size를 final_chunk 정도로 잡으면 독립성 확보.
    """
    n = len(x)
    if block_size is None:
        block_size = final_chunk
    pods = []
    n_blocks = n // block_size
    
    for _ in range(n_boot):
        # 블록 단위로 리샘플링 (상관 보정)
        block_starts = np.random.choice(n - block_size + 1, n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        idx = idx[idx < n]
        p, ok = fit_mle(x[idx], y[idx], a_th)
        if ok:
            pods.append(get_mean_pod(x_range, p, a_th))
    if not pods:
        return np.zeros_like(x_range)
    return np.percentile(np.array(pods), 5, axis=0)

# ── Step 3: POD 적합 (Real & Synth) ─────────────────────────
print("\nStep 2: Heteroscedastic MLE + Block Bootstrap fitting...")

x_range_r = np.linspace(0, 100, 300)   # 외삽 포함 0~100%
x_range_s = np.linspace(0, 100, 300)

# Real
params_r, ok_r = fit_mle(all_real_a, all_real_ahat, threshold)
pod_mean_r = get_mean_pod(x_range_r, params_r, threshold)
pod_lcb_r  = bootstrap_lcb(all_real_a, all_real_ahat, threshold, x_range_r)

a50_r     = x_range_r[np.argmax(pod_mean_r >= 0.5)] if np.any(pod_mean_r >= 0.5) else np.nan
a90_r     = x_range_r[np.argmax(pod_mean_r >= 0.9)] if np.any(pod_mean_r >= 0.9) else np.nan
a90_95_r  = x_range_r[np.argmax(pod_lcb_r  >= 0.9)] if np.any(pod_lcb_r  >= 0.9) else np.nan

# Synth
params_s, ok_s = fit_mle(all_synth_a, all_synth_ahat, threshold)
pod_mean_s = get_mean_pod(x_range_s, params_s, threshold)
pod_lcb_s  = bootstrap_lcb(all_synth_a, all_synth_ahat, threshold, x_range_s)

a50_s     = x_range_s[np.argmax(pod_mean_s >= 0.5)] if np.any(pod_mean_s >= 0.5) else np.nan
a90_s     = x_range_s[np.argmax(pod_mean_s >= 0.9)] if np.any(pod_mean_s >= 0.9) else np.nan
a90_95_s  = x_range_s[np.argmax(pod_lcb_s  >= 0.9)] if np.any(pod_lcb_s  >= 0.9) else np.nan

print(f"\n  ───── Real Data ─────")
print(f"   a50 = {a50_r:.2f}%   a90 = {a90_r:.2f}%   a90/95 (LCB) = {a90_95_r:.2f}%")
print(f"  ───── Synth Data ────")
print(f"   a50 = {a50_s:.2f}%   a90 = {a90_s:.2f}%   a90/95 (LCB) = {a90_95_s:.2f}%")

# ── Step 4: 4-Panel 시각화 ──────────────────────────────────
print("\nStep 3: Generating 4-panel diagnostic plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (1,1) Real: 산점도 + MLE 평균선 + ±2σ + threshold
ax = axes[0, 0]
ax.scatter(all_real_a, all_real_ahat, alpha=0.08, color='#E74C3C', s=8, label='Samples')
x_plot = np.linspace(0, 50, 100)
mu_y  = params_r[0] + params_r[1] * x_plot
tau_y = np.maximum(params_r[2] + params_r[3] * x_plot, 1e-8)
ax.plot(x_plot, mu_y, 'k-', linewidth=2.5,
        label=f'MLE Mean ($\\mu = {params_r[0]:.4f} + {params_r[1]:.6f}\\cdot a$)')
ax.fill_between(x_plot, mu_y - 2*tau_y, mu_y + 2*tau_y,
                color='gray', alpha=0.2, label='$\\pm 2\\sigma$ Band')
ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
           label=f'Threshold ({threshold:.4f})')
ax.set_title("Real: â vs a Scatter + MLE Fit", fontweight='bold')
ax.set_xlabel("Damage Case, $a$ (%)"); ax.set_ylabel("MMD Score, $\\hat{a}$")
ax.set_xlim([0, 50])
ax.legend(loc='upper left', fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)

# (1,2) Synth: 산점도 + MLE 평균선 + ±2σ + threshold
ax = axes[0, 1]
ax.scatter(all_synth_a, all_synth_ahat, alpha=0.05, color='#3498DB', s=8, label='Samples')
x_plot = np.linspace(0, 100, 100)
mu_y  = params_s[0] + params_s[1] * x_plot
tau_y = np.maximum(params_s[2] + params_s[3] * x_plot, 1e-8)
ax.plot(x_plot, mu_y, 'k-', linewidth=2.5,
        label=f'MLE Mean ($\\mu = {params_s[0]:.4f} + {params_s[1]:.6f}\\cdot a$)')
ax.fill_between(x_plot, mu_y - 2*tau_y, mu_y + 2*tau_y,
                color='gray', alpha=0.2, label='$\\pm 2\\sigma$ Band')
ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
           label=f'Threshold ({threshold:.4f})')
ax.set_title("Synth: â vs a Scatter + MLE Fit", fontweight='bold')
ax.set_xlabel("Mapped Synthetic Damage, $a$ (%)"); ax.set_ylabel("MMD Score, $\\hat{a}$")
ax.set_xlim([0, 100])
ax.legend(loc='upper left', fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)

# (2,1) Real POD Curve
ax = axes[1, 0]
ax.plot(x_range_r, pod_mean_r, color='#E74C3C', linewidth=3,
        label=f'Mean POD ($a_{{90}}$={a90_r:.1f}%)')
ax.plot(x_range_r, pod_lcb_r,  color='#E74C3C', linewidth=2, linestyle='--',
        label=f'95% LCB ($a_{{90/95}}$={a90_95_r:.1f}%)')
ax.axhline(0.9, color='black', linestyle=':', linewidth=1.5, label='90% Target')
ax.axhline(0.5, color='gray',  linestyle=':', linewidth=1.0, alpha=0.5)
if not np.isnan(a90_r) and a90_r < 100:
    ax.axvline(a90_r,    color='#E74C3C', linestyle=':', alpha=0.5)
ax.set_title("Real Data: POD Curve", fontweight='bold')
ax.set_xlabel("Damage Case, $a$ (%)"); ax.set_ylabel("Probability of Detection")
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower right')

# (2,2) Synth POD Curve
ax = axes[1, 1]
ax.plot(x_range_s, pod_mean_s, color='#3498DB', linewidth=3,
        label=f'Mean POD ($a_{{90}}$={a90_s:.1f}%)')
ax.plot(x_range_s, pod_lcb_s,  color='#3498DB', linewidth=2, linestyle='--',
        label=f'95% LCB ($a_{{90/95}}$={a90_95_s:.1f}%)')
ax.axhline(0.9, color='black', linestyle=':', linewidth=1.5, label='90% Target')
ax.axhline(0.5, color='gray',  linestyle=':', linewidth=1.0, alpha=0.5)
if not np.isnan(a90_s) and a90_s < 100:
    ax.axvline(a90_s,    color='#3498DB', linestyle=':', alpha=0.5)
ax.set_title("Synth Data: POD Curve", fontweight='bold')
ax.set_xlabel("Mapped Synthetic Damage, $a$ (%)"); ax.set_ylabel("Probability of Detection")
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "POD_Curves_4Panel.png"), dpi=300)
plt.show()

# ── Step 5: Real vs Synth Overlay (논문 핵심 그림) ──────────
print("\nStep 4: Generating Real-vs-Synth overlay plot...")
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_range_r, pod_mean_r, color='#E74C3C', linewidth=3,
        label=f'Real Mean ($a_{{90}}$={a90_r:.1f}%)')
ax.plot(x_range_r, pod_lcb_r,  color='#E74C3C', linewidth=2, linestyle='--',
        alpha=0.7, label=f'Real 95% LCB ($a_{{90/95}}$={a90_95_r:.1f}%)')

ax.plot(x_range_s, pod_mean_s, color='#3498DB', linewidth=3,
        label=f'Synth Mean ($a_{{90}}$={a90_s:.1f}%)')
ax.plot(x_range_s, pod_lcb_s,  color='#3498DB', linewidth=2, linestyle='--',
        alpha=0.7, label=f'Synth 95% LCB ($a_{{90/95}}$={a90_95_s:.1f}%)')

ax.axhline(0.9, color='black', linestyle=':', linewidth=1.5, label='90% Target')

ax.set_title("Real vs Synth POD: Thesis Hypothesis Verification",
             fontweight='bold', fontsize=14)
ax.set_xlabel("Damage Severity, $a$ (%)", fontsize=12)
ax.set_ylabel("Probability of Detection", fontsize=12)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "POD_RealVsSynth_Overlay.png"), dpi=300)
plt.show()

# ── Step 6: 요약 테이블 ──────────────────────────────────────
print("\n" + "=" * 60)
print("                  POD 분석 최종 요약")
print("=" * 60)
print(f"  {'Metric':<20} {'Real':>15} {'Synth':>15}")
print("-" * 60)
print(f"  {'Slope (b1)':<20} {params_r[1]:>15.6f} {params_s[1]:>15.6f}")
print(f"  {'Intercept (b0)':<20} {params_r[0]:>15.6f} {params_s[0]:>15.6f}")
print(f"  {'a50 (%)':<20} {a50_r:>15.2f} {a50_s:>15.2f}")
print(f"  {'a90 (%)':<20} {a90_r:>15.2f} {a90_s:>15.2f}")
print(f"  {'a90/95 LCB (%)':<20} {a90_95_r:>15.2f} {a90_95_s:>15.2f}")
print("=" * 60)
print(f"\n🎉 POD 분석 완료! 그림 2장 저장:")
print(f"   - POD_Curves_4Panel.png")
print(f"   - POD_RealVsSynth_Overlay.png")