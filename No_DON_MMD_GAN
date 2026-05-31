# ================================================================
# MMD-GAN Standalone (5 Runs DR Averaging) for Manual Residual Data
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from scipy.optimize import curve_fit
from tqdm import tqdm

# ================================================================
# [0] 공통 설정
# ================================================================
DIR_B_RAW       = r"E:\2ndstructuredata\raw data"
FILE_B          = "healthyclean.txt"

# 💡 방금 Code_37 스크립트가 수동 잔차 합성 데이터를 만들어놓은 폴더 경로입니다!
SYNTH_DATA_DIR  = r"E:\2ndstructuredata\Code_37_Manual_Residual\Synthetic_B_Data"

# 결과를 저장할 새로운 폴더
SAVE_DIR_GAN    = r"E:\2ndstructuredata\Code_MMD_GAN_Manual_Residual"

if not os.path.exists(SAVE_DIR_GAN):
    os.makedirs(SAVE_DIR_GAN)

WINDOW_SIZE        = 128
CHANNELS           = 8
DAMAGE_CASES_B     = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
SYNTHETIC_DI_STEPS = np.round(np.arange(0.1, 1.01, 0.1), 2) # DI 0.1 ~ 1.0 (10단계)
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_range_r = np.linspace(0, 50, 200)
x_range_s = np.linspace(0, 100, 200)

N_RUNS = 5  
SEED = 42 

print("=" * 70)
print(f"  MMD-GAN (5-Seed DR Avg) on Manual Residual Data")
print(f"  Device: {device}")
print("=" * 70)


# ================================================================
# [1] 데이터 로더 (MinMaxScaler, [-1,1] 정규화)
# ================================================================
TRAIN_SIZE        = 1000
VAL_SIZE          = 500
NUM_SAMPLES_GANAE = 500

scaler_B = None
Train_H  = None
Val_H    = None

def load_health_data_ganae():
    global scaler_B, Train_H, Val_H
    path = os.path.join(DIR_B_RAW, FILE_B)
    raw  = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
    data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
    ns   = data.shape[1] // WINDOW_SIZE
    data = data[:, :ns * WINDOW_SIZE]
    reshaped  = data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
    scaler_B  = MinMaxScaler(feature_range=(0, 1)).fit(reshaped)
    norm_data = (scaler_B.transform(reshaped) * 2.0) - 1.0
    norm_data = norm_data.reshape(-1, CHANNELS, WINDOW_SIZE)
    np.random.seed(42)
    idx       = np.random.permutation(ns)
    Train_H   = norm_data[idx[:TRAIN_SIZE]]
    Val_H     = norm_data[idx[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]]
    print(f"✅ GAN 건강 데이터: Train {Train_H.shape}, Val {Val_H.shape}")

def load_damage_ganae(path, is_synth=False):
    try:
        if is_synth:
            data = np.loadtxt(path, delimiter='\t').T.astype(np.float32)
        else:
            raw  = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
        ns = data.shape[1] // WINDOW_SIZE
        if ns == 0: return None
        data = data[:, :ns * WINDOW_SIZE]
        reshaped  = data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
        norm_data = (scaler_B.transform(reshaped) * 2.0) - 1.0
        norm_data = norm_data.reshape(-1, CHANNELS, WINDOW_SIZE)
        if ns > NUM_SAMPLES_GANAE:
            np.random.seed(42)
            sel = np.random.choice(ns, NUM_SAMPLES_GANAE, replace=False)
            norm_data = norm_data[sel]
        return norm_data
    except Exception as e:
        print(f"   Error loading {path}: {e}")
        return None

load_health_data_ganae()


# ================================================================
# [2] MMD-GAN 모델 정의 및 Probit 함수
# ================================================================
def probit_func(x, mu, sigma):
    return norm.cdf(x, loc=mu, scale=sigma)

GAN_N1, GAN_N2     = 35, 19
GAN_KERNEL_SIZE    = 6
GAN_ALPHA          = 0.3799
GAN_STRIDE         = 3
GAN_LR             = 0.001926
GAN_EPOCHS         = 112
GAN_BATCH          = 96
GAN_LATENT_DIM     = 100
GAN_FEAT_DIM       = 64

def rbf_kernel_torch(x, y, gamma=1.0):
    x = x.unsqueeze(1); y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss_calc(x, y, gamma=1.0):
    xx = rbf_kernel_torch(x, x, gamma)
    yy = rbf_kernel_torch(y, y, gamma)
    xy = rbf_kernel_torch(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

class GAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(GAN_LATENT_DIM, 32 * GAN_N1), nn.LeakyReLU(GAN_ALPHA))
        self.conv1 = nn.ConvTranspose1d(GAN_N1, GAN_N1, GAN_KERNEL_SIZE, stride=2, padding=GAN_KERNEL_SIZE//2, output_padding=1)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2 = nn.ConvTranspose1d(GAN_N1, CHANNELS, GAN_KERNEL_SIZE, stride=2, padding=GAN_KERNEL_SIZE//2, output_padding=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.fc(x).view(-1, GAN_N1, 32)
        x = self.lrelu1(self.conv1(x))
        x = self.tanh(self.conv2(x))
        if x.size(2) < 128: x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

class GAN_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, GAN_N2, GAN_KERNEL_SIZE, stride=GAN_STRIDE, padding=GAN_KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2 = nn.Conv1d(GAN_N2, GAN_N2, GAN_KERNEL_SIZE, stride=GAN_STRIDE, padding=GAN_KERNEL_SIZE//2)
        self.lrelu2 = nn.LeakyReLU(GAN_ALPHA)
        self.flatten = nn.Flatten()
        dummy = torch.zeros(1, CHANNELS, 128)
        out = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.fc = nn.Linear(out.view(1, -1).size(1), GAN_FEAT_DIM)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(self.flatten(x))


# ================================================================
# [3] 5회 반복 학습 및 개별 탐지율(DR) 추출
# ================================================================
print("\n" + "=" * 70)
print(f"  [3] MMD-GAN: {N_RUNS}회 반복 학습 및 탐지율(DR) 평균화")
print("=" * 70)

gan_dr_real_runs = {case: [] for case in DAMAGE_CASES_B}
gan_dr_synth_runs = {di: [] for di in SYNTHETIC_DI_STEPS}

for run in range(N_RUNS):
    print(f"\n  [Run {run+1}/{N_RUNS}] 모델 초기화 및 학습 중...")
    
    torch.manual_seed(SEED + run)
    
    gan_gen    = GAN_Generator().to(device)
    gan_critic = GAN_Critic().to(device)
    opt_g = optim.Adam(gan_gen.parameters(), lr=GAN_LR)
    opt_c = optim.Adam(gan_critic.parameters(), lr=GAN_LR)

    gan_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)),
        batch_size=GAN_BATCH, shuffle=True, drop_last=True
    )

    for epoch in tqdm(range(GAN_EPOCHS), desc=f"Run {run+1} Epochs", leave=False):
        for (real_data,) in gan_loader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)
            
            opt_c.zero_grad()
            z = torch.randn(b_size, GAN_LATENT_DIM).to(device)
            fake_data = gan_gen(z)
            feat_real = gan_critic(real_data)
            feat_fake = gan_critic(fake_data.detach())
            c_loss = -mmd_loss_calc(feat_real, feat_fake)
            c_loss.backward()
            opt_c.step()
            
            opt_g.zero_grad()
            feat_fake_g = gan_critic(gan_gen(z))
            g_loss = mmd_loss_calc(feat_real.detach(), feat_fake_g)
            g_loss.backward()
            opt_g.step()

    # Mahalanobis 통계 추출
    gan_critic.eval()
    with torch.no_grad():
        gan_feat_train = gan_critic(torch.FloatTensor(Train_H).to(device)).cpu().numpy()
        gan_mu_h    = np.mean(gan_feat_train, axis=0)
        gan_cov_h   = np.cov(gan_feat_train, rowvar=False)
        gan_cov_inv = np.linalg.inv(gan_cov_h + 1e-5 * np.eye(GAN_FEAT_DIM))

    def gan_score(data):
        with torch.no_grad():
            feat = gan_critic(torch.FloatTensor(data).to(device)).detach().cpu().numpy()
            diff = feat - gan_mu_h
            mahal_sq = np.sum(np.dot(diff, gan_cov_inv) * diff, axis=1)
            return np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))

    gan_val_scores = gan_score(Val_H)
    thr_real = np.mean(gan_val_scores) + 2 * np.std(gan_val_scores)
    
    data_s_0 = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), is_synth=True)
    if data_s_0 is not None:
        gan_synth_0_scores = gan_score(data_s_0)
        thr_synth = np.mean(gan_synth_0_scores) + 2 * np.std(gan_synth_0_scores)
    else:
        thr_synth = thr_real

    # 1. Real Data DR 계산
    for case in DAMAGE_CASES_B:
        d = load_damage_ganae(os.path.join(DIR_B_RAW, f"D3_{case}_1.txt"))
        if d is not None:
            scores = gan_score(d)
            dr = np.mean(scores > thr_real) 
            gan_dr_real_runs[case].append(dr)

    # 2. Synthetic Data DR 계산 (Manual Residual Data)
    for di in SYNTHETIC_DI_STEPS:
        d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
        if d is not None:
            scores = gan_score(d)
            dr = np.mean(scores > thr_synth)
            gan_dr_synth_runs[di].append(dr)

# (주의: 파일 맨 위 import 부분에 from scipy.optimize import minimize 가 있어야 합니다!)
from scipy.optimize import minimize

# ================================================================
# [4] Hit/Miss MLE 기반 세밀한 POD 커브 피팅 및 부트스트랩 (LCB)
# ================================================================
print("\n" + "=" * 70)
print("  [4] Hit/Miss MLE 피팅 및 95% LCB(신뢰하한선) 추출")
print("=" * 70)

def fit_hit_miss_mle(a_vals, dr_vals):
    """
    💡 단순 최소제곱법(curve_fit)이 아닌, 
    이항 분포(Binomial)의 교차 엔트로피(Cross-Entropy)를 최소화하는 엄밀한 MLE 함수입니다.
    """
    def neg_log_lik(params):
        mu, sigma = params
        p = norm.cdf(a_vals, loc=mu, scale=sigma)
        p = np.clip(p, 1e-10, 1 - 1e-10) # Log(0) 방지
        # Fractional Binomial Log-Likelihood
        ll = dr_vals * np.log(p) + (1 - dr_vals) * np.log(1 - p)
        return -np.sum(ll)
    
    init_mu = np.median(a_vals)
    init_sigma = np.std(a_vals) + 1e-5
    res = minimize(neg_log_lik, x0=[init_mu, init_sigma], bounds=[(1e-3, None), (1e-3, None)], method='L-BFGS-B')
    return res.x, res.success

def get_mle_pod_curve(x_range, params):
    mu, sigma = params
    return norm.cdf(x_range, loc=mu, scale=sigma)

def bootstrap_hit_miss_lcb(a_vals, dr_runs_dict, x_range, n_boot=200):
    """5번의 Run 결과를 무작위 복원 추출하여 95% 신뢰하한선(LCB)을 계산합니다."""
    boot_pods = []
    cases = list(dr_runs_dict.keys())
    n_runs = len(dr_runs_dict[cases[0]])
    
    for _ in tqdm(range(n_boot), desc="  Bootstrapping LCB", leave=False):
        # 5번의 Run 중 무작위로 복원 추출 (예: [0, 2, 2, 3, 4])
        idx = np.random.choice(n_runs, n_runs, replace=True)
        boot_avg_dr = []
        for c in a_vals:
            sampled_drs = [dr_runs_dict[c][i] for i in idx]
            boot_avg_dr.append(np.mean(sampled_drs))
            
        params, success = fit_hit_miss_mle(a_vals, np.array(boot_avg_dr))
        if success:
            boot_pods.append(get_mle_pod_curve(x_range, params))
            
    if not boot_pods: return np.zeros_like(x_range)
    return np.percentile(np.array(boot_pods), 5, axis=0) # 하위 5% -> 95% LCB

# 💡 Real Data MLE 피팅
a_vals_r = np.array(DAMAGE_CASES_B)
avg_dr_real = np.array([np.mean(gan_dr_real_runs[case]) for case in DAMAGE_CASES_B])
params_r, ok_r = fit_hit_miss_mle(a_vals_r, avg_dr_real)
gan_pod_mean_r = get_mle_pod_curve(x_range_r, params_r) if ok_r else np.zeros_like(x_range_r)
gan_pod_lcb_r  = bootstrap_hit_miss_lcb(a_vals_r, gan_dr_real_runs, x_range_r)
gan_a90_r = x_range_r[np.argmax(gan_pod_mean_r >= 0.9)] if np.any(gan_pod_mean_r >= 0.9) else np.nan

# 💡 Synthetic Data MLE 피팅
a_vals_s = np.array(SYNTHETIC_DI_STEPS * 100)
avg_dr_synth = np.array([np.mean(gan_dr_synth_runs[di]) for di in SYNTHETIC_DI_STEPS])
params_s, ok_s = fit_hit_miss_mle(a_vals_s, avg_dr_synth)
gan_pod_mean_s = get_mle_pod_curve(x_range_s, params_s) if ok_s else np.zeros_like(x_range_s)
gan_pod_lcb_s  = bootstrap_hit_miss_lcb(SYNTHETIC_DI_STEPS, gan_dr_synth_runs, x_range_s)
gan_a90_s = x_range_s[np.argmax(gan_pod_mean_s >= 0.9)] if np.any(gan_pod_mean_s >= 0.9) else np.nan

print(f"✅ MLE-Based POD | Real a90: {gan_a90_r:.1f}%, Synth a90: {gan_a90_s:.1f}%")


# ================================================================
# [5] 시각화 (MLE 곡선 및 95% LCB 포함)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Real
ax = axes[0]
lbl_r = f'Mean POD (MLE) ($a_{{90}}$={gan_a90_r:.1f}%)' if not np.isnan(gan_a90_r) else 'Mean POD (N/A)'
ax.plot(x_range_r, gan_pod_mean_r, color='#E74C3C', linewidth=3, label=lbl_r)
ax.plot(x_range_r, gan_pod_lcb_r, color='#E74C3C', linewidth=2, linestyle='--', label='95% LCB')
ax.plot(DAMAGE_CASES_B, avg_dr_real, 'ko', alpha=0.5, label='Averaged DR Points')
ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
ax.fill_between(x_range_r, 0, 0.9, color='gray', alpha=0.04)
ax.set_title("Real Damage: Detailed MLE POD", fontweight='bold', fontsize=13)
ax.set_xlabel("Damage Case, $a$ (%)", fontsize=11)
ax.set_ylabel("Probability of Detection", fontsize=11)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 50])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=10)

# Synth
ax = axes[1]
lbl_s = f'Mean POD (MLE) ($a_{{90}}$={gan_a90_s:.1f}%)' if not np.isnan(gan_a90_s) else 'Mean POD (N/A)'
ax.plot(x_range_s, gan_pod_mean_s, color='#E74C3C', linewidth=3, label=lbl_s)
ax.plot(x_range_s, gan_pod_lcb_s, color='#E74C3C', linewidth=2, linestyle='--', label='95% LCB')
ax.plot(SYNTHETIC_DI_STEPS * 100, avg_dr_synth, 'ko', alpha=0.5, label='Averaged DR Points')
ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
ax.fill_between(x_range_s, 0, 0.9, color='gray', alpha=0.04)
ax.set_title("Manual Residual Synthetic: Detailed MLE POD", fontweight='bold', fontsize=13)
ax.set_xlabel("Mapped Synthetic Damage, $a$ (%)", fontsize=11)
ax.set_ylabel("Probability of Detection", fontsize=11)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_GAN, "MMD_GAN_Detailed_MLE_POD.png"), dpi=300)
plt.show()

print("\n🎉 MLE 기반 MMD-GAN 정밀 평가 완료!")