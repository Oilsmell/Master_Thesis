# %% [Cell 1] Configuration & Data Loading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from scipy.stats import norm, linregress

# 💡 [VS Code 블로킹 방지 설정] 창을 띄우지 않고 백그라운드에서 즉시 이미지 파일로 저장
# %% [Cell 1] Configuration & Data Loading (일부 수정)
# ... (앞선 기본 import 문은 동일)

# 💡 [VS Code 블로킹 방지 설정] 창을 띄우지 않고 백그라운드에서 즉시 이미지 파일로 저장
plt.ioff()

class Config:
    DIR_B_RAW = r"E:\2ndstructuredata\raw data"
    FILE_B = "healthyclean.txt"
    SYNTH_DATA_DIR = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"
    SAVE_DIR = r"E:\git\Master_Thesis\MMD_GAN"
    
    WINDOW_SIZE = 128
    CHANNELS = 8
    TRAIN_SIZE = 1000
    VAL_SIZE = 500
    NUM_SAMPLES = 500
    
    N1 = 35; N2 = 19; KERNEL_SIZE = 6; ALPHA = 0.3799; STRIDE = 3
    LR = 0.001926; EPOCHS = 112; BATCH_SIZE = 96
    LATENT_DIM = 100; FEAT_DIM = 64 
    
    DAMAGE_CASES_B = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    SYNTHETIC_DI_STEPS = np.arange(0.0, 1.01, 0.02)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 💡 [변경] 완전히 독립적이고 분산된 10개의 랜덤 시드 자동 확보
    np.random.seed(999) # 시드 생성을 위한 마스터 시드 고정
    SEEDS = np.random.choice(range(100, 10000), size=10, replace=False).tolist()

cfg = Config()
print(f"🎲 설정된 10개의 분산 무작위 시드: {cfg.SEEDS}")
if not os.path.exists(cfg.SAVE_DIR): os.makedirs(cfg.SAVE_DIR)

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
        if x.size(2) < 128:
            x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

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
        self.fc = nn.Linear(flat_size, cfg.FEAT_DIM)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)

print("✅ MMD-GAN Architecture Loaded!")

# %% [Cell 3 & 4] 5-Seed Ensemble Training & Feature Parameter Extraction
# 💡 5개 모델의 가중치와 마할라노비스 통계량을 저장할 딕셔너리 리스트
ensemble_models = []

print(f"=== Starting 5-Seed MMD-GAN Ensemble Training ===")

for run, seed in enumerate(cfg.SEEDS):
    print(f"\n▶ [Run {run+1}/5] Training with Seed {seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    gen = Generator().to(cfg.device)
    critic = Critic().to(cfg.device)
    
    opt_g = optim.Adam(gen.parameters(), lr=cfg.LR)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.LR)
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)), 
        batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True
    )
    
    # 모델 학습
    for epoch in range(cfg.EPOCHS):
        for (real_data,) in dataloader:
            real_data = real_data.to(cfg.device)
            b_size = real_data.size(0)
            
            opt_c.zero_grad()
            z = torch.randn(b_size, cfg.LATENT_DIM).to(cfg.device)
            fake_data = gen(z)
            feat_real = critic(real_data)
            feat_fake = critic(fake_data.detach())
            c_loss = -mmd_loss_calc(feat_real, feat_fake) 
            c_loss.backward()
            opt_c.step()
            
            opt_g.zero_grad()
            feat_fake_g = critic(gen(z))
            g_loss = mmd_loss_calc(feat_real.detach(), feat_fake_g)
            g_loss.backward()
            opt_g.step()
            
    print(f"  ↳ Run {run+1} Complete. Extracting Mahalanobis baseline parameters...")
    critic.eval()
    
    # 현재 시드 모델의 마할라노비스 기준점 계산
    with torch.no_grad():
        feat_h = critic(torch.FloatTensor(Train_H).to(cfg.device)).cpu().numpy()
        mu_healthy = np.mean(feat_h, axis=0)
        cov_healthy = np.cov(feat_h, rowvar=False)
        cov_inv = np.linalg.inv(cov_healthy + 1e-5 * np.eye(cfg.FEAT_DIM))
        
    # 앙상블 리스트에 추가
    ensemble_models.append({
        'critic_state': critic.state_dict(),
        'mu_healthy': mu_healthy,
        'cov_inv': cov_inv
    })

print("\n✅ All 5 Models Successfully Trained and Extracted!")

# 💡 [앙상블 전용 점수 계산 함수] 5개 모델의 예측치 평균값 반환
def get_ensemble_mahalanobis_scores(data_array):
    total_scores = np.zeros(len(data_array))
    
    # 임시 모델 뼈대 생성
    temp_critic = Critic().to(cfg.device)
    temp_critic.eval()
    
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        for model_meta in ensemble_models:
            temp_critic.load_state_dict(model_meta['critic_state'])
            feat = temp_critic(tensor_data).detach().cpu().numpy()
            diff = feat - model_meta['mu_healthy']
            mahal_sq = np.sum(np.dot(diff, model_meta['cov_inv']) * diff, axis=1)
            run_score = np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))
            total_scores += run_score
            
    return total_scores / len(ensemble_models) # 5개 점수의 산술 평균

# %% [Cell 5] 앙상블 스코어 검증 시각화 (Trend Verification)
print("\n=== Evaluating 10-Seed Ensemble Mahalanobis Distances ===")

# 5개 모델 평균 계산 함수는 이전과 동일한 메커니즘으로 10개를 돕니다.
def get_ensemble_mahalanobis_scores(data_array):
    total_scores = np.zeros(len(data_array))
    temp_critic = Critic().to(cfg.device)
    temp_critic.eval()
    
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        for model_meta in ensemble_models:
            temp_critic.load_state_dict(model_meta['critic_state'])
            feat = temp_critic(tensor_data).detach().cpu().numpy()
            diff = feat - model_meta['mu_healthy']
            mahal_sq = np.sum(np.dot(diff, model_meta['cov_inv']) * diff, axis=1)
            run_score = np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))
            total_scores += run_score
            
    return total_scores / len(ensemble_models)

val_scores = get_ensemble_mahalanobis_scores(Val_H)

# 💡 [변경] PFA = 0.04 (상위 4% 커트라인) 설정을 위해 96백분위수 사용
threshold_real = np.percentile(val_scores, 98)
print(f" -> 🟢 10-Seed Ensemble Mahalanobis Threshold (PFA=0.02): {threshold_real:.4f}")

# 독립 임계값을 위한 합성 데이터 Baseline (Synthetic DI 0.00) 적용
data_s_0 = load_damage_data(os.path.join(cfg.SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), is_synth=True)
synth_0_scores = get_ensemble_mahalanobis_scores(data_s_0)
threshold_synth = np.percentile(synth_0_scores, 98) # 💡 여기도 동일하게 4% 커트라인 적용
print(f" -> 🔵 10-Seed Ensemble Synthetic Threshold (PFA=0.02): {threshold_synth:.4f}")

# ... [데이터 아카이빙 로직 및 Plotting 구조는 이전과 동일하나 threshold 변수 반영]

# 데이터 가공 및 전체 추세 저장을 위한 배열 초기화
all_real_x, all_real_y = [], []
all_synth_x, all_synth_y = [], []

for case in cfg.DAMAGE_CASES_B:
    data_d = load_damage_data(os.path.join(cfg.DIR_B_RAW, f"D3_{case}_1.txt"), is_synth=False)
    if data_d is not None:
        scores = get_ensemble_mahalanobis_scores(data_d)
        all_real_x.extend([case] * len(scores))
        all_real_y.extend(scores)

for di in cfg.SYNTHETIC_DI_STEPS:
    data_s = load_damage_data(os.path.join(cfg.SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
    if data_s is not None:
        scores = get_ensemble_mahalanobis_scores(data_s)
        all_synth_x.extend([di] * len(scores))
        all_synth_y.extend(scores)

all_real_x, all_real_y = np.array(all_real_x), np.array(all_real_y)
all_synth_x, all_synth_y = np.array(all_synth_x), np.array(all_synth_y)

# 시각화 후 저장
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
slope_r, intercept_r, r_val_r, _, _ = linregress(all_real_x, all_real_y)
ax1.scatter(all_real_x, all_real_y, alpha=0.1, color='red', s=10, label='Samples')
x_vals_r = np.array([min(all_real_x), max(all_real_x)])
ax1.plot(x_vals_r, slope_r * x_vals_r + intercept_r, 'k--', linewidth=2.5, label=fr'Trend ($R^2$={r_val_r**2:.2f})')
ax1.axhline(y=threshold_real, color='darkred', linestyle='-', linewidth=2, label=f'Threshold ({threshold_real:.2f})')
ax1.set_title("Ensemble Real Data: â vs a", fontweight='bold')
ax1.set_xlabel("Damage Case, a (%)"); ax1.set_ylabel("Damage Index, â")
ax1.grid(True, linestyle=':', alpha=0.7); ax1.legend()

slope_s, intercept_s, r_val_s, _, _ = linregress(all_synth_x, all_synth_y)
ax2.scatter(all_synth_x, all_synth_y, alpha=0.05, color='blue', s=10, label='Samples')
x_vals_s = np.array([0.0, 1.0])
ax2.plot(x_vals_s, slope_s * x_vals_s + intercept_s, 'k--', linewidth=2.5, label=fr'Trend ($R^2$={r_val_s**2:.2f})')
ax2.axhline(y=threshold_synth, color='darkblue', linestyle='-', linewidth=2, label=f'Threshold ({threshold_synth:.2f})')
ax2.set_title("Ensemble Synthetic Data: â vs a", fontweight='bold')
ax2.set_xlabel("Input Synthetic DI, a"); ax2.set_ylabel("Damage Index, â")
ax2.grid(True, linestyle=':', alpha=0.7); ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "Ensemble_Scatter_Independent_Thresholds.png"), dpi=300)
plt.close() # 💡 창을 열지 않고 즉시 닫아 멈춤 방지

# %% [Cell 6] Final POD Pipeline: 5-Seed Ensemble Mean POD Curves
print("\n=== Phase 3: Plotting Final Ensemble POD Curves ===")

def calculate_pod_curve(x_data, y_data, a_th, x_range):
    n = len(x_data)
    slope, intercept, _, _, _ = linregress(x_data, y_data)
    
    y_pred = slope * x_data + intercept
    sigma_res = np.sqrt(np.sum((y_data - y_pred)**2) / (n - 2))
    
    mu_y_vals = slope * x_range + intercept
    z_vals = (mu_y_vals - a_th) / sigma_res
    mean_pod = norm.cdf(z_vals) 
    
    # 95% 신뢰 하한선(LCB) 계산 
    mean_x = np.mean(x_data)
    s_xx = np.sum((x_data - mean_x)**2)
    var_z = (1/n) + ((x_range - mean_x)**2 / s_xx) + (z_vals**2 / (2 * (n - 1)))
    se_z = np.sqrt(var_z)
    
    z_lcb = z_vals - norm.ppf(0.95) * se_z 
    pod_95_lcb = norm.cdf(z_lcb)
    
    return mean_pod, pod_95_lcb

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Real Data - Ensemble POD Curve
x_range_r = np.linspace(0, 50, 200)
pod_mean_r, pod_lcb_r = calculate_pod_curve(all_real_x, all_real_y, threshold_real, x_range_r)
a_90_r = x_range_r[np.argmax(pod_mean_r >= 0.9)] if np.any(pod_mean_r >= 0.9) else np.nan

ax3.plot(x_range_r, pod_mean_r, 'r-', linewidth=3, label=fr'Mean POD ($a_{{r,90}}$={a_90_r:.1f}%)')
ax3.plot(x_range_r, pod_lcb_r, 'r--', linewidth=2, label='95% Lower Confidence Bound')
ax3.axhline(y=0.9, color='gray', linestyle=':', label='90% POD Target')
ax3.set_title("Ensemble Mean POD Curve (Real Data)", fontweight='bold')
ax3.set_xlabel("Damage Case, a (%)"); ax3.set_ylabel("Probability of Detection (POD)")
ax3.set_ylim([0, 1.05]); ax3.grid(True, linestyle=':', alpha=0.7); ax3.legend(loc='lower right')

# 2. Synthetic Data - Ensemble POD Curve
x_range_s = np.linspace(0, 1.0, 200)
pod_mean_s, pod_lcb_s = calculate_pod_curve(all_synth_x, all_synth_y, threshold_synth, x_range_s)
a_90_s = x_range_s[np.argmax(pod_mean_s >= 0.9)] if np.any(pod_mean_s >= 0.9) else np.nan

ax4.plot(x_range_s, pod_mean_s, 'b-', linewidth=3, label=fr'Mean POD ($a_{{g,90}}$={a_90_s:.2f})')
ax4.plot(x_range_s, pod_lcb_s, 'b--', linewidth=2, label='95% Lower Confidence Bound')
ax4.axhline(y=0.9, color='gray', linestyle=':', label='90% POD Target')
ax4.set_title("Ensemble Mean POD Curve (Synthetic Data)", fontweight='bold')
ax4.set_xlabel("Input Synthetic DI, a"); ax4.set_ylabel("Probability of Detection (POD)")
ax4.set_ylim([0, 1.05]); ax4.grid(True, linestyle=':', alpha=0.7); ax4.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "Final_Ensemble_POD_Curves.png"), dpi=300)
plt.close()

print(f"🎉 5-Seed 앙상블 평균 POD 분석 완료!")
print(fr"   - Real 90% Detection Size ($a_{{r,90}}$): {a_90_r:.2f}%")
print(fr"   - Synthetic 90% Detection Size ($a_{{g,90}}$): {a_90_s:.4f}")
print(f"   - 생성된 모든 그래프 파일은 다음 경로에 안전하게 저장되었습니다: {cfg.SAVE_DIR}")