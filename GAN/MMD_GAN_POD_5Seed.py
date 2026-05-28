# ================================================================
# MMD-GAN Standalone POD Pipeline (5 Runs DR Averaging & Curve Fitting)
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
SYNTH_DATA_DIR  = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"
SAVE_DIR_GAN    = r"E:\2ndstructuredata\Code_MMD_GAN_Standalone"

if not os.path.exists(SAVE_DIR_GAN):
    os.makedirs(SAVE_DIR_GAN)

WINDOW_SIZE        = 128
CHANNELS           = 8
DAMAGE_CASES_B     = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
SYNTHETIC_DI_STEPS = np.round(np.arange(0.0, 1.01, 0.02), 2)
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_range_r = np.linspace(0, 50, 200)
x_range_s = np.linspace(0, 100, 200)

N_RUNS = 5  
SEED = 42 # 💡 누락되었던 SEED 변수 추가!

print("=" * 70)
print(f"  MMD-GAN Standalone POD Pipeline (Runs: {N_RUNS})")
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
    
    # 딥러닝 초기화 시드 분산 적용
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

    # 해당 Run의 개별 임계값 계산 (동적 임계값)
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

    # 2. Synthetic Data DR 계산
    for di in SYNTHETIC_DI_STEPS:
        d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
        if d is not None:
            scores = gan_score(d)
            dr = np.mean(scores > thr_synth)
            gan_dr_synth_runs[di].append(dr)

# ================================================================
# [4] 평균 탐지율(DR) 도출 및 Probit 커브 피팅
# ================================================================
print("\n" + "=" * 70)
print("  [4] 평균 탐지율 피팅 및 그래프 출력")
print("=" * 70)

# Real Data 피팅
gan_avg_dr_real = [np.mean(gan_dr_real_runs[case]) for case in DAMAGE_CASES_B]
try:
    popt_r, _ = curve_fit(probit_func, DAMAGE_CASES_B, gan_avg_dr_real, bounds=([0, 0], [100, 50]))
    gan_pod_mean_r = probit_func(x_range_r, *popt_r)
    gan_a90_r = x_range_r[np.argmax(gan_pod_mean_r >= 0.9)] if np.any(gan_pod_mean_r >= 0.9) else np.nan
except Exception as e:
    print("Real Curve Fit Failed:", e)
    gan_pod_mean_r = np.zeros_like(x_range_r); gan_a90_r = np.nan

# Synthetic Data 피팅
gan_avg_dr_synth = [np.mean(gan_dr_synth_runs[di]) for di in SYNTHETIC_DI_STEPS]
try:
    popt_s, _ = curve_fit(probit_func, SYNTHETIC_DI_STEPS * 100, gan_avg_dr_synth, bounds=([0, 0], [150, 50]))
    gan_pod_mean_s = probit_func(x_range_s, *popt_s)
    gan_a90_s = x_range_s[np.argmax(gan_pod_mean_s >= 0.9)] if np.any(gan_pod_mean_s >= 0.9) else np.nan
except Exception as e:
    print("Synth Curve Fit Failed:", e)
    gan_pod_mean_s = np.zeros_like(x_range_s); gan_a90_s = np.nan

print(f"✅ MMD-GAN (DR Averaged) | Real a90: {gan_a90_r:.1f}%, Synth a90: {gan_a90_s:.1f}%")

# ================================================================
# [5] 시각화 (Real vs Synth)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Real
ax = axes[0]
lbl_r = f'MMD-GAN ($a_{{90}}$={gan_a90_r:.1f}%)' if not np.isnan(gan_a90_r) else 'MMD-GAN (N/A)'
ax.plot(x_range_r, gan_pod_mean_r, color='#E74C3C', linewidth=3, label=lbl_r)
ax.plot(DAMAGE_CASES_B, gan_avg_dr_real, 'ko', alpha=0.5, label='Averaged DR Points')
ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
ax.fill_between(x_range_r, 0, 0.9, color='gray', alpha=0.04)
ax.set_title("Real Damage: POD (DR Averaged)", fontweight='bold', fontsize=13)
ax.set_xlabel("Damage Case, $a$ (%)", fontsize=11)
ax.set_ylabel("Probability of Detection", fontsize=11)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 50])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=10)

# Synth
ax = axes[1]
lbl_s = f'MMD-GAN ($a_{{90}}$={gan_a90_s:.1f}%)' if not np.isnan(gan_a90_s) else 'MMD-GAN (N/A)'
ax.plot(x_range_s, gan_pod_mean_s, color='#E74C3C', linewidth=3, label=lbl_s)
ax.plot(SYNTHETIC_DI_STEPS * 100, gan_avg_dr_synth, 'ko', alpha=0.5, label='Averaged DR Points')
ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
ax.fill_between(x_range_s, 0, 0.9, color='gray', alpha=0.04)
ax.set_title("Synthetic Damage: POD (DR Averaged)", fontweight='bold', fontsize=13)
ax.set_xlabel("Mapped Synthetic Damage, $a$ (%)", fontsize=11)
ax.set_ylabel("Probability of Detection", fontsize=11)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_GAN, "MMD_GAN_Standalone_POD.png"), dpi=300)
plt.show()

print("\n🎉 MMD-GAN Standalone 실행 완료!")