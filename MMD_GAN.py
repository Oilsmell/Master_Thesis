# %% [Cell 1] Configuration & Data Loading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

class Config:
    # 경로 설정
    DIR_B_RAW = r"E:\2ndstructuredata\raw data"
    FILE_B = "healthyclean.txt"
    SYNTH_DATA_DIR = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"
    SAVE_DIR = r"E:\2ndstructuredata\Code_22_Optimized_MMDGAN"
    
    # 기본 설정
    WINDOW_SIZE = 128
    CHANNELS = 8
    TRAIN_SIZE = 1000
    VAL_SIZE = 500
    NUM_SAMPLES = 500
    
    # 💡 NSGA-II 최적 하이퍼파라미터 적용 (MMD-GAN 용)
    N1 = 35
    N2 = 19
    KERNEL_SIZE = 6
    ALPHA = 0.3799
    STRIDE = 3
    LR = 0.001926
    EPOCHS = 112
    BATCH_SIZE = 96
    
    LATENT_DIM = 100
    FEAT_DIM = 64 # MMD-GAN의 핵심: 64차원 특징 공간
    
    DAMAGE_CASES_B = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    SYNTHETIC_DI_STEPS = np.arange(0.0, 1.01, 0.02)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
if not os.path.exists(cfg.SAVE_DIR): os.makedirs(cfg.SAVE_DIR)

# 전역(Global) 스케일러 및 데이터 저장 변수
scaler_B = None
Train_H, Val_H = None, None

def load_health_data():
    global scaler_B, Train_H, Val_H
    path = os.path.join(cfg.DIR_B_RAW, cfg.FILE_B)
    raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
    data = np.array(raw, dtype=np.float32).reshape(cfg.CHANNELS, -1)
    ns = data.shape[1] // cfg.WINDOW_SIZE
    data = data[:, :ns * cfg.WINDOW_SIZE]
    
    reshaped = data.reshape(cfg.CHANNELS, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
    scaler_B = MinMaxScaler(feature_range=(0, 1)).fit(reshaped)
    norm_data = (scaler_B.transform(reshaped) * 2.0) - 1.0 
    norm_data = norm_data.reshape(-1, cfg.CHANNELS, cfg.WINDOW_SIZE)
    
    np.random.seed(42)
    indices = np.random.permutation(ns)
    Train_H = norm_data[indices[:cfg.TRAIN_SIZE]]
    Val_H = norm_data[indices[cfg.TRAIN_SIZE:cfg.TRAIN_SIZE + cfg.VAL_SIZE]]
    print(f"✅ Healthy Data Loaded: Train {Train_H.shape}, Val {Val_H.shape}")

def load_damage_data(path, is_synth=False):
    try:
        if is_synth: data = np.loadtxt(path, delimiter='\t').T 
        else:
            raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(cfg.CHANNELS, -1)
            
        ns = data.shape[1] // cfg.WINDOW_SIZE
        if ns == 0: return None
        data = data[:, :ns * cfg.WINDOW_SIZE]
        reshaped = data.reshape(cfg.CHANNELS, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
        norm_data = (scaler_B.transform(reshaped) * 2.0) - 1.0 
        norm_data = norm_data.reshape(-1, cfg.CHANNELS, cfg.WINDOW_SIZE)
        
        if ns > cfg.NUM_SAMPLES:
            np.random.seed(42) 
            indices = np.random.choice(ns, cfg.NUM_SAMPLES, replace=False)
            norm_data = norm_data[indices]
        return norm_data
    except Exception: return None

load_health_data()

# %% [Cell 2] MMD-GAN Architecture & Loss Functions
def rbf_kernel(x, y, gamma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss_calc(x, y, gamma=1.0):
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(cfg.LATENT_DIM, 32 * cfg.N1), nn.LeakyReLU(cfg.ALPHA))
        self.conv1 = nn.ConvTranspose1d(cfg.N1, cfg.N1, cfg.KERNEL_SIZE, stride=2, padding=cfg.KERNEL_SIZE//2, output_padding=1)
        self.lrelu1 = nn.LeakyReLU(cfg.ALPHA)
        self.conv2 = nn.ConvTranspose1d(cfg.N1, cfg.CHANNELS, cfg.KERNEL_SIZE, stride=2, padding=cfg.KERNEL_SIZE//2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x).view(-1, cfg.N1, 32)
        x = self.lrelu1(self.conv1(x))
        x = self.tanh(self.conv2(x))
        
        # 💡 [핵심 방어 코드] 
        # 만약 수학적 연산 오차로 인해 길이가 128보다 짧게 만들어지면, 빈 공간을 0으로 채워(Padding) 강제로 맞춥니다.
        if x.size(2) < 128:
            x = F.pad(x, (0, 128 - x.size(2)))
            
        return x[:, :, :128] # 128보다 길면 잘라내고 반환

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.CHANNELS, cfg.N2, cfg.KERNEL_SIZE, stride=cfg.STRIDE, padding=cfg.KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(cfg.ALPHA)
        self.conv2 = nn.Conv1d(cfg.N2, cfg.N2, cfg.KERNEL_SIZE, stride=cfg.STRIDE, padding=cfg.KERNEL_SIZE//2)
        self.lrelu2 = nn.LeakyReLU(cfg.ALPHA)
        self.flatten = nn.Flatten()
        
        dummy = torch.zeros(1, cfg.CHANNELS, 128)
        out = self.conv2(self.lrelu1(self.conv1(dummy)))
        flat_size = out.view(1, -1).size(1)
        
        self.fc = nn.Linear(flat_size, cfg.FEAT_DIM) # 64차원 출력

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)

print("✅ MMD-GAN Architecture Loaded!")
# %% [Cell 3] Training Optimized MMD-GAN
# 최적의 파라미터로 모델을 단 1회 집중적으로 학습합니다.
print("=== Starting MMD-GAN Training ===")
gen = Generator().to(cfg.device)
critic = Critic().to(cfg.device)

opt_g = optim.Adam(gen.parameters(), lr=cfg.LR)
opt_c = optim.Adam(critic.parameters(), lr=cfg.LR)

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)), 
    batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True
)

for epoch in range(cfg.EPOCHS):
    for (real_data,) in dataloader:
        real_data = real_data.to(cfg.device)
        b_size = real_data.size(0)
        
        # 1. Critic 학습 (진짜와 가짜 사이의 MMD 거리 최대화)
        opt_c.zero_grad()
        z = torch.randn(b_size, cfg.LATENT_DIM).to(cfg.device)
        fake_data = gen(z)
        
        feat_real = critic(real_data)
        feat_fake = critic(fake_data.detach())
        c_loss = -mmd_loss_calc(feat_real, feat_fake) 
        c_loss.backward()
        opt_c.step()
        
        # 2. Generator 학습 (진짜와 가짜 사이의 MMD 거리 최소화)
        opt_g.zero_grad()
        feat_fake_g = critic(gen(z))
        g_loss = mmd_loss_calc(feat_real.detach(), feat_fake_g)
        g_loss.backward()
        opt_g.step()
        
    if (epoch+1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{cfg.EPOCHS} | Critic Loss: {c_loss.item():.6f} | Gen Loss: {g_loss.item():.6f}")

print("✅ MMD-GAN Training Complete!")

# %% [Cell 4] Mahalanobis Distance Evaluation & Plotting (Dual Baseline 적용)
print("=== Calculating Mahalanobis Distances (Dual Baseline) ===")
critic.eval()

# 1. 💡 두 개의 베이스라인(중심점/공분산) 설정
with torch.no_grad():
    # Baseline 1: 실제 데이터 기준 (Train_H)
    feat_h = critic(torch.FloatTensor(Train_H).to(cfg.device)).cpu().numpy()
    mu_real = np.mean(feat_h, axis=0)
    cov_real = np.cov(feat_h, rowvar=False)
    cov_inv_real = np.linalg.inv(cov_real + 1e-5 * np.eye(cfg.FEAT_DIM))
    
    # Baseline 2: 합성 데이터 기준 (DI 0.00)
    data_s_0 = load_damage_data(os.path.join(cfg.SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), is_synth=True)
    feat_s0 = critic(torch.FloatTensor(data_s_0).to(cfg.device)).cpu().numpy()
    mu_synth = np.mean(feat_s0, axis=0)
    cov_synth = np.cov(feat_s0, rowvar=False)
    cov_inv_synth = np.linalg.inv(cov_synth + 1e-5 * np.eye(cfg.FEAT_DIM))

# 2. 💡 중심점(mu, cov_inv)을 인자로 받는 유연한 거리 함수
def get_mahalanobis_scores(data_array, mu, cov_inv):
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        feat = critic(tensor_data).detach().cpu().numpy()
        diff = feat - mu
        mahal_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        return np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))

# 3. 💡 독립적 Threshold 설정 (PFA=0.02)
val_scores = get_mahalanobis_scores(Val_H, mu_real, cov_inv_real)
threshold_real = np.mean(val_scores) + 2 * np.std(val_scores)
print(f"   -> 🔴 Real Threshold (PFA=0.02): {threshold_real:.4f}")

synth_0_scores = get_mahalanobis_scores(data_s_0, mu_synth, cov_inv_synth)
threshold_synth = np.mean(synth_0_scores) + 2 * np.std(synth_0_scores)
print(f"   -> 🔵 Synth Threshold (PFA=0.02): {threshold_synth:.4f}")

# 4. 각 케이스별 검출률(%) 계산
rates_real, rates_synth = [], []

print("Evaluating Real Damage Data...")
for case in cfg.DAMAGE_CASES_B:
    data_d = load_damage_data(os.path.join(cfg.DIR_B_RAW, f"D3_{case}_1.txt"), is_synth=False)
    if data_d is not None:
        rates_real.append(np.mean(get_mahalanobis_scores(data_d, mu_real, cov_inv_real) > threshold_real) * 100.0)

print("Evaluating Synthetic Damage Data...")
for di in cfg.SYNTHETIC_DI_STEPS:
    data_s = load_damage_data(os.path.join(cfg.SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
    if data_s is not None:
        rates_synth.append(np.mean(get_mahalanobis_scores(data_s, mu_synth, cov_inv_synth) > threshold_synth) * 100.0)
    else: rates_synth.append(0.0)

# 시각화 (기존과 동일)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(cfg.DAMAGE_CASES_B, rates_real, 'r-o', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.axhline(y=2.0, color='gray', linestyle='--', label='False Alarm Rate (2%)')
ax1.set_title("Optimized MMD-GAN (Real): % Classified as Damage", fontweight='bold')
ax1.set_xlabel("Damage Case (%)"); ax1.set_ylabel("Detection Rate (%)")
ax1.set_xticks(cfg.DAMAGE_CASES_B); ax1.set_ylim([0, max(max(rates_real), 10) + 5])
ax1.grid(True, linestyle=':', alpha=0.7); ax1.legend()

ax2.plot(cfg.SYNTHETIC_DI_STEPS, rates_synth, 'b-s', linewidth=2.5, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
ax2.axhline(y=2.0, color='gray', linestyle='--', label='False Alarm Rate (2%)')
ax2.set_title("Optimized MMD-GAN (Synthetic): % Classified as Damage", fontweight='bold')
ax2.set_xlabel("Input Synthetic DI (0.0 to 1.0)"); ax2.set_ylabel("Detection Rate (%)")
ax2.set_xticks(np.arange(0.0, 1.1, 0.2)); ax2.set_ylim([0, max(max(rates_synth), 10) + 5])
ax2.grid(True, linestyle=':', alpha=0.7); ax2.legend()

plt.tight_layout()
plot_path = os.path.join(cfg.SAVE_DIR, "optimized_mmd_mahalanobis_trend_dual.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"\n✅ 분석 완료! 깔끔하게 분리된 이중 베이스라인 파이프라인이 정상 작동했습니다.")

# %% [Cell 5] Export Model & Mahalanobis Parameters (Dual Baseline)
import pickle
import os
import torch

print("=== Phase 4: Exporting Models and Parameters ===")

# 1. PyTorch 딥러닝 모델 가중치 저장 (.pth)
gen_path = os.path.join(cfg.SAVE_DIR, "MMD_Generator_Optimized.pth")
critic_path = os.path.join(cfg.SAVE_DIR, "MMD_Critic_Optimized.pth")

torch.save(gen.state_dict(), gen_path)
torch.save(critic.state_dict(), critic_path)
print(f"✅ Models saved:\n - {gen_path}\n - {critic_path}")

# 2. 💡 [수정] 이중 베이스라인을 위한 두 세트의 필수 통계 파라미터 모두 저장 (.pkl)
mahalanobis_params = {
    'mu_real': mu_real,
    'cov_inv_real': cov_inv_real,
    'threshold_real': threshold_real,
    
    'mu_synth': mu_synth,
    'cov_inv_synth': cov_inv_synth,
    'threshold_synth': threshold_synth,
    
    'feat_dim': cfg.FEAT_DIM
}

# 혹시 헷갈리지 않도록 저장 파일 이름도 Dual로 살짝 바꿔줍니다
params_path = os.path.join(cfg.SAVE_DIR, "Mahalanobis_Params_Dual.pkl")
with open(params_path, 'wb') as f:
    pickle.dump(mahalanobis_params, f)

print(f"✅ Mahalanobis Dual Parameters saved:\n - {params_path}")

# 3. 데이터 스케일러 저장 (나중에 새로운 생 데이터가 들어왔을 때 전처리용)
scaler_path = os.path.join(cfg.SAVE_DIR, "Scaler_B_Optimized.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler_B, f)
print(f"✅ Data Scaler saved:\n - {scaler_path}")

print("\n🎉 모든 학습된 모델과 이중 기준(Dual) 파라미터 추출이 완료되었습니다! 이제 에러 없이 잘 저장되었습니다.")

# %% [Cell 6] POD Preprocessing: Normality Check & Linear Trend (Dual Baseline)
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.stats import norm, linregress

print("=== Phase 1: Preparing Data for POD Preprocessing ===")

# 중심점을 인자로 받는 함수 재선언
def get_mahalanobis_scores(data_array, mu, cov_inv):
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        feat = critic(tensor_data).detach().cpu().numpy()
        diff = feat - mu
        mahal_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        return np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))

target_real_cases = [4, 32, 48]
target_synth_dis = [0.0, 0.2, 0.5, 0.8]

real_scores_dict, synth_scores_dict = {}, {}
all_real_x, all_real_y = [], []
all_synth_x, all_synth_y = [], []

print("Extracting Scores for Real Data...")
for case in cfg.DAMAGE_CASES_B:
    data = load_damage_data(os.path.join(cfg.DIR_B_RAW, f"D3_{case}_1.txt"), is_synth=False)
    if data is not None:
        scores = get_mahalanobis_scores(data, mu_real, cov_inv_real) # 💡 Real 중심 사용
        if case in target_real_cases:
            real_scores_dict[case] = scores
        all_real_x.extend([case] * len(scores))
        all_real_y.extend(scores)

print("Extracting Scores for Synthetic Data...")
for di in cfg.SYNTHETIC_DI_STEPS:
    data = load_damage_data(os.path.join(cfg.SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
    if data is not None:
        scores = get_mahalanobis_scores(data, mu_synth, cov_inv_synth) # 💡 Synth 중심 사용
        for target in target_synth_dis:
            if abs(di - target) < 0.01:
                synth_scores_dict[target] = scores
        all_synth_x.extend([di] * len(scores))
        all_synth_y.extend(scores)

# --- 아래 그래프 그리기 코드는 원본과 동일하게 두시면 됩니다 ---
# (공간 절약을 위해 시각화 코드는 생략하지만, 기존 Cell 6의 '=== Phase 2: Plotting ~' 
# 부터 끝까지 그대로 붙여넣으시면 됩니다.)

# %% [Cell 7] Final POD Pipeline: Scatter MLE & Bootstrapped POD Curves (Dual Baseline)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, linregress
import os
import torch
from tqdm import tqdm

print("=== Phase 1: Independent Thresholds & X-Axis Mapping ===")

# 합성 데이터 스케일 매핑 (DI 0~1.0 -> 0~100%)
all_synth_x_mapped = np.array(all_synth_x) * 100.0

print("\n=== Phase 2: Heteroscedastic MLE Optimization ===")
def mll_loss(params, x, y, a_th):
    b0, b1, tau0, tau1 = params
    mu = b0 + b1 * x
    tau = np.maximum(tau0 + tau1 * x, 1e-5)
    censored = y <= a_th
    uncensored = y > a_th
    
    ll_uncensored = norm.logpdf(y[uncensored], loc=mu[uncensored], scale=tau[uncensored])
    ll_censored = norm.logcdf(a_th, loc=mu[censored], scale=tau[censored])
    return -(np.sum(ll_uncensored) + np.sum(ll_censored))

def fit_mle_pod(x_data, y_data, a_th):
    slope, intercept, _, _, _ = linregress(x_data, y_data)
    init_params = [intercept, slope, np.std(y_data), 0.0]
    bounds = [(None, None), (None, None), (1e-3, None), (0.0, None)]
    res = minimize(mll_loss, init_params, args=(x_data, y_data, a_th), bounds=bounds, method='L-BFGS-B')
    return res.x, res.success

# 💡 [핵심] Cell 4에서 계산된 각각의 독립된 Threshold 사용
params_r, success_r = fit_mle_pod(np.array(all_real_x), np.array(all_real_y), threshold_real)
params_s, success_s = fit_mle_pod(all_synth_x_mapped, np.array(all_synth_y), threshold_synth)

print("\n=== Phase 2.5: Plotting MLE Trend, Scatter, and Thresholds ===")
fig_scatter, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(16, 6))

def plot_mle_scatter(ax, x_data, y_data, params, a_th, title, xlabel, x_limit):
    b0, b1, t0, t1 = params
    ax.scatter(x_data, y_data, alpha=0.15, color='gray', s=15, label='Actual Data Points')
    
    x_range = np.linspace(0, x_limit, 100)
    mu_y = b0 + b1 * x_range
    tau_y = np.maximum(t0 + t1 * x_range, 1e-5)
    
    ax.plot(x_range, mu_y, 'k-', linewidth=2.5, label=f'MLE Mean ($\mu = {b0:.2f} + {b1:.4f}a$)')
    ax.plot(x_range, mu_y + 2*tau_y, 'k--', linewidth=1.5, alpha=0.7, label='$\pm 2\sigma$ Boundary')
    ax.plot(x_range, mu_y - 2*tau_y, 'k--', linewidth=1.5, alpha=0.7)
    ax.fill_between(x_range, mu_y - 2*tau_y, mu_y + 2*tau_y, color='gray', alpha=0.1)
    
    ax.axhline(y=a_th, color='red', linestyle='-', linewidth=2.5, label=f'Threshold ({a_th:.2f})')
    ax.fill_between(x_range, ax.get_ylim()[0], a_th, color='red', alpha=0.05, label='Miss Zone')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel("Damage Index, $\hat{a}$")
    ax.set_xlim([0, x_limit]); ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='upper left')

# 각각의 산점도에 맞는 threshold를 넘겨줍니다.
plot_mle_scatter(ax_s1, np.array(all_real_x), np.array(all_real_y), params_r, threshold_real, 
                 "Real Data: MLE & Threshold", "Damage Case, $a$ (%)", 50)
plot_mle_scatter(ax_s2, all_synth_x_mapped, np.array(all_synth_y), params_s, threshold_synth, 
                 "Synthetic Data: MLE & Threshold", "Mapped Synthetic Damage, $a$ (%)", 100)

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "Scatter_MLE_Dual_Threshold.png"), dpi=300)
plt.show()

print("\n=== Phase 3: Robust LCB Calculation via Bootstrapping ===")
def calculate_pod_bootstrap(x_data, y_data, a_th, x_range, n_boot=300):
    n = len(x_data)
    boot_pods = []
    for _ in tqdm(range(n_boot), desc="Bootstrapping LCB"):
        idx = np.random.choice(n, n, replace=True)
        params_b, success_b = fit_mle_pod(x_data[idx], y_data[idx], a_th)
        if success_b:
            b0, b1, t0, t1 = params_b
            mu = b0 + b1 * x_range
            tau = np.maximum(t0 + t1 * x_range, 1e-5)
            boot_pods.append(norm.cdf((mu - a_th) / tau))
    return np.percentile(np.array(boot_pods), 5, axis=0) 

def get_mean_pod(x_range, params, a_th):
    b0, b1, t0, t1 = params
    mu = b0 + b1 * x_range
    tau = np.maximum(t0 + t1 * x_range, 1e-5)
    return norm.cdf((mu - a_th) / tau)

x_range_r = np.linspace(0, 50, 200)   
x_range_s = np.linspace(0, 100, 200)  

# 💡 [핵심] 부트스트랩 계산 시에도 독립된 Threshold 적용
pod_mean_r = get_mean_pod(x_range_r, params_r, threshold_real)
pod_lcb_r = calculate_pod_bootstrap(np.array(all_real_x), np.array(all_real_y), threshold_real, x_range_r)

pod_mean_s = get_mean_pod(x_range_s, params_s, threshold_synth)
pod_lcb_s = calculate_pod_bootstrap(all_synth_x_mapped, np.array(all_synth_y), threshold_synth, x_range_s)

a_90_r = x_range_r[np.argmax(pod_mean_r >= 0.9)] if np.any(pod_mean_r >= 0.9) else np.nan
a_90_s = x_range_s[np.argmax(pod_mean_s >= 0.9)] if np.any(pod_mean_s >= 0.9) else np.nan

print("\n=== Phase 4: Plotting Separated POD Curves ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(x_range_r, pod_mean_r, 'r-', linewidth=3, label=f'Mean POD ($a_{{r,90}}$={a_90_r:.1f}%)')
ax1.plot(x_range_r, pod_lcb_r, 'r--', linewidth=2, label='95% Lower Confidence Bound')
ax1.axhline(y=0.9, color='k', linestyle=':', linewidth=2, label='90% Target')
ax1.set_title("Real Data: POD Curve (Actual Damage)", fontweight='bold')
ax1.set_xlabel("Damage Case, $a$ (%)")
ax1.set_ylabel("Probability of Detection (POD)")
ax1.set_ylim([0, 1.05])
ax1.set_xlim([0, 50])
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='lower right')

ax2.plot(x_range_s, pod_mean_s, 'b-', linewidth=3, label=f'Mean POD ($a_{{g,90}}$={a_90_s:.1f}%)')
ax2.plot(x_range_s, pod_lcb_s, 'b--', linewidth=2, label='95% Lower Confidence Bound')
ax2.axhline(y=0.9, color='k', linestyle=':', linewidth=2, label='90% Target')
ax2.set_title("Synthetic Data: POD Curve (Generated Damage)", fontweight='bold')
ax2.set_xlabel("Mapped Synthetic Damage, $a$ (%)")
ax2.set_ylabel("Probability of Detection (POD)")
ax2.set_ylim([0, 1.05])
ax2.set_xlim([0, 50]) # 애플 투 애플 비교를 위해 50%로 고정
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "POD_Separated_Dual_Bootstrap.png"), dpi=300)
plt.show()

print(f"✅ 연산 완료! 스케일과 독립 임계값이 완벽하게 맞춰진 상태에서 a_r_90 = {a_90_r:.2f}%, a_g_90 = {a_90_s:.2f}% 입니다.")
