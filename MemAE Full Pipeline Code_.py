# %% [Cell 0] MemAE Full Pipeline (Code_34 → POD)
# ============================================================
# 데이터 로드 → MemAE 학습 → Detection rate → MLE Scatter → POD
# 한 번에 실행되는 통합 코드
# 사용 폴더: E:\2ndstructuredata\Code_34_MemAE_Baseline_v2_shrink002
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, norm, linregress
from scipy.optimize import minimize


# %% [Cell 1] Configuration
class Config:
    DIR_B_RAW = r"E:\2ndstructuredata\raw data"
    FILE_B = "healthyclean.txt"
    SYNTH_DATA_DIR = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"

    # 💡 결과 저장 폴더
    SAVE_DIR = r"E:\2ndstructuredata\Code_34_MemAE_Baseline_v2_shrink002"

    # 데이터 설정
    WINDOW_SIZE = 128
    CHANNELS = 8
    TRAIN_SIZE = 1000
    VAL_SIZE = 500
    NUM_SAMPLES = 500

    # AE 백본 (MMD-GAN Critic 구조 차용)
    N_FEAT = 19
    KERNEL_SIZE = 6
    STRIDE = 3
    ALPHA = 0.3799
    LATENT_DIM = 64

    # 💡 MemAE 핵심 하이퍼파라미터 (v2: shrink 0.02로 sparse 강제)
    N_MEMORY_SLOTS = 50
    MEMORY_SHRINK_THRESHOLD = 0.02      # ← 이 값이 핵심! (이전 0.0025 → 0.02)
    LAMBDA_ENTROPY = 0.0002
    MEMORY_TEMP = 1.0

    # 학습
    LR = 0.001
    EPOCHS = 112        # MMD-GAN과 동일
    BATCH_SIZE = 96     # MMD-GAN과 동일
    SEED = 42

    # 평가 케이스
    DAMAGE_CASES_B = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    SYNTHETIC_DI_STEPS = np.arange(0.0, 1.01, 0.02)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
if not os.path.exists(cfg.SAVE_DIR):
    os.makedirs(cfg.SAVE_DIR)

# Seed 고정
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.SEED)

print("=" * 70)
print(f"  MemAE Full Pipeline (seed={cfg.SEED})")
print(f"  Save dir: {cfg.SAVE_DIR}")
print(f"  Device: {cfg.device}")
print(f"  Shrink threshold: {cfg.MEMORY_SHRINK_THRESHOLD} (1/N_SLOTS = {1/cfg.N_MEMORY_SLOTS})")
print("=" * 70)


# %% [Cell 2] Data Loading
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
    print(f"\n✅ Healthy: Train {Train_H.shape}, Val {Val_H.shape}")

def load_damage_data(path, is_synth=False):
    try:
        if is_synth:
            data = np.loadtxt(path, delimiter='\t').T
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
    except Exception as e:
        print(f"   Error loading {path}: {e}")
        return None

load_health_data()


# %% [Cell 3] MemAE Architecture
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.CHANNELS, cfg.N_FEAT, cfg.KERNEL_SIZE,
                               stride=cfg.STRIDE, padding=cfg.KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(cfg.ALPHA)
        self.conv2 = nn.Conv1d(cfg.N_FEAT, cfg.N_FEAT, cfg.KERNEL_SIZE,
                               stride=cfg.STRIDE, padding=cfg.KERNEL_SIZE//2)
        self.lrelu2 = nn.LeakyReLU(cfg.ALPHA)

        dummy = torch.zeros(1, cfg.CHANNELS, 128)
        out = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.conv_out_len = out.size(2)
        self.flat_size = out.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_size, cfg.LATENT_DIM)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(x.view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, conv_out_len, flat_size):
        super().__init__()
        self.conv_out_len = conv_out_len
        self.fc = nn.Linear(cfg.LATENT_DIM, flat_size)
        self.lrelu0 = nn.LeakyReLU(cfg.ALPHA)
        self.deconv1 = nn.ConvTranspose1d(cfg.N_FEAT, cfg.N_FEAT, cfg.KERNEL_SIZE,
                                          stride=cfg.STRIDE, padding=cfg.KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(cfg.ALPHA)
        self.deconv2 = nn.ConvTranspose1d(cfg.N_FEAT, cfg.CHANNELS, cfg.KERNEL_SIZE,
                                          stride=cfg.STRIDE, padding=cfg.KERNEL_SIZE//2)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.lrelu0(self.fc(z))
        x = x.view(-1, cfg.N_FEAT, self.conv_out_len)
        x = self.lrelu1(self.deconv1(x))
        x = self.tanh(self.deconv2(x))
        if x.size(2) < 128:
            x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]


class MemoryModule(nn.Module):
    """
    💡 핵심: 학습 가능한 메모리 뱅크
    - Cosine similarity로 attention
    - Hard shrinkage로 작은 attention 제거 (sparse 강제)
    """
    def __init__(self, n_slots, dim, shrink_threshold=0.02, temp=1.0):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.shrink_threshold = shrink_threshold
        self.temp = temp
        self.memory = nn.Parameter(torch.empty(n_slots, dim))
        nn.init.xavier_normal_(self.memory)

    def forward(self, z):
        z_norm = F.normalize(z, p=2, dim=1)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        sim = torch.matmul(z_norm, mem_norm.t()) / self.temp
        attention = F.softmax(sim, dim=1)

        if self.shrink_threshold > 0:
            attention = (F.relu(attention - self.shrink_threshold) * attention) / \
                        (torch.abs(attention - self.shrink_threshold) + 1e-12)
            attention = F.normalize(attention, p=1, dim=1)

        z_mem = torch.matmul(attention, self.memory)
        return z_mem, attention


class MemAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(self.encoder.conv_out_len, self.encoder.flat_size)
        self.memory = MemoryModule(
            n_slots=cfg.N_MEMORY_SLOTS,
            dim=cfg.LATENT_DIM,
            shrink_threshold=cfg.MEMORY_SHRINK_THRESHOLD,
            temp=cfg.MEMORY_TEMP
        )

    def forward(self, x):
        z = self.encoder(x)
        z_mem, attention = self.memory(z)
        recon = self.decoder(z_mem)
        return recon, z, z_mem, attention


def memory_entropy_loss(attention, eps=1e-12):
    return torch.mean(-torch.sum(attention * torch.log(attention + eps), dim=1))


print("✅ MemAE Architecture Loaded!")
print(f"   Memory slots: {cfg.N_MEMORY_SLOTS}")
print(f"   Shrink threshold: {cfg.MEMORY_SHRINK_THRESHOLD}")


# %% [Cell 4] Training
print("\n=== Starting MemAE Training ===")
memae = MemAE().to(cfg.device)
optimizer = optim.Adam(memae.parameters(), lr=cfg.LR)
mse_loss = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)),
    batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True
)
val_tensor = torch.FloatTensor(Val_H).to(cfg.device)

best_val_loss = float('inf')
patience = 25
patience_counter = 0
history = {'recon_loss': [], 'entropy_loss': [], 'val_loss': [], 'attention_sparsity': []}

for epoch in range(cfg.EPOCHS):
    memae.train()
    epoch_recon, epoch_entropy, n_batches = 0.0, 0.0, 0
    epoch_active_slots = 0.0

    for (real_data,) in train_loader:
        real_data = real_data.to(cfg.device)

        optimizer.zero_grad()
        recon, z, z_mem, attention = memae(real_data)
        r_loss = mse_loss(recon, real_data)
        e_loss = memory_entropy_loss(attention)
        total_loss = r_loss + cfg.LAMBDA_ENTROPY * e_loss
        total_loss.backward()
        optimizer.step()

        epoch_recon += r_loss.item()
        epoch_entropy += e_loss.item()

        with torch.no_grad():
            active = (attention > 1.0 / (2 * cfg.N_MEMORY_SLOTS)).float().sum(dim=1).mean().item()
            epoch_active_slots += active
        n_batches += 1

    history['recon_loss'].append(epoch_recon / n_batches)
    history['entropy_loss'].append(epoch_entropy / n_batches)
    history['attention_sparsity'].append(epoch_active_slots / n_batches)

    # Validation
    memae.eval()
    with torch.no_grad():
        recon_v, _, _, _ = memae(val_tensor)
        v_loss = mse_loss(recon_v, val_tensor).item()
    history['val_loss'].append(v_loss)

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_counter = 0
        torch.save(memae.state_dict(), os.path.join(cfg.SAVE_DIR, "MemAE_best.pth"))
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{cfg.EPOCHS} | "
              f"Recon: {history['recon_loss'][-1]:.5f} | "
              f"Entropy: {history['entropy_loss'][-1]:.4f} | "
              f"Active slots: {history['attention_sparsity'][-1]:.1f}/{cfg.N_MEMORY_SLOTS} | "
              f"Val: {v_loss:.5f}")

    if patience_counter >= patience:
        print(f"  ⏹️ Early stopping at epoch {epoch+1}")
        break

memae.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR, "MemAE_best.pth")))
print("✅ MemAE Training Complete!")

# 학습 곡선
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0,0].plot(history['recon_loss'], label='Train', linewidth=2, color='blue')
axes[0,0].plot(history['val_loss'], label='Val', linewidth=2, color='orange')
axes[0,0].set_yscale('log'); axes[0,0].set_title("Reconstruction Loss", fontweight='bold')
axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(True, linestyle=':', alpha=0.7); axes[0,0].legend()

axes[0,1].plot(history['entropy_loss'], color='purple', linewidth=2)
axes[0,1].set_title("Memory Attention Entropy (낮을수록 sparse)", fontweight='bold')
axes[0,1].set_xlabel("Epoch"); axes[0,1].grid(True, linestyle=':', alpha=0.7)

axes[1,0].plot(history['attention_sparsity'], color='green', linewidth=2)
axes[1,0].axhline(y=cfg.N_MEMORY_SLOTS, color='gray', linestyle='--',
                  label=f'Total slots ({cfg.N_MEMORY_SLOTS})')
axes[1,0].set_title("Active Memory Slots per Sample", fontweight='bold')
axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("# of active slots")
axes[1,0].grid(True, linestyle=':', alpha=0.7); axes[1,0].legend()

memae.eval()
with torch.no_grad():
    train_tensor = torch.FloatTensor(Train_H).to(cfg.device)
    _, _, _, train_attention = memae(train_tensor)
    avg_attention = train_attention.mean(dim=0).cpu().numpy()

axes[1,1].bar(range(cfg.N_MEMORY_SLOTS), avg_attention, color='steelblue', edgecolor='black')
axes[1,1].set_title("Average Attention per Memory Slot (Healthy)", fontweight='bold')
axes[1,1].set_xlabel("Memory Slot Index"); axes[1,1].set_ylabel("Avg Attention")
axes[1,1].grid(True, linestyle=':', alpha=0.7, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "MemAE_training_curves.png"), dpi=300)
plt.show()


# %% [Cell 5] Anomaly Score Functions
print("\n=== Computing Anomaly Scores ===")
memae.eval()

with torch.no_grad():
    train_tensor = torch.FloatTensor(Train_H).to(cfg.device)
    _, train_z, _, _ = memae(train_tensor)
    train_z_np = train_z.cpu().numpy()
    mu_healthy = np.mean(train_z_np, axis=0)
    cov_healthy = np.cov(train_z_np, rowvar=False)
    cov_inv = np.linalg.inv(cov_healthy + 1e-5 * np.eye(cfg.LATENT_DIM))

def get_recon_error(data_array, log_scale=False):
    """Reconstruction error per sample. log_scale=True면 POD용 로그 변환"""
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        recon, _, _, _ = memae(tensor_data)
        err = torch.mean((recon - tensor_data) ** 2, dim=(1, 2)).cpu().numpy()
        if log_scale:
            return np.log(np.maximum(err, 1e-10))
        return err

def get_mahalanobis_scores(data_array):
    """Latent에서 Mahalanobis 거리 (log-scaled)"""
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        _, z, _, _ = memae(tensor_data)
        z_np = z.cpu().numpy()
        diff = z_np - mu_healthy
        mahal_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        return np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))

def get_memory_score(data_array):
    """💡 MemAE 고유: z와 z_mem의 L2 거리"""
    with torch.no_grad():
        tensor_data = torch.FloatTensor(data_array).to(cfg.device)
        _, z, z_mem, _ = memae(tensor_data)
        return torch.norm(z - z_mem, dim=1).cpu().numpy()

# Threshold (PFA=2%)
val_recon = get_recon_error(Val_H)
val_mahal = get_mahalanobis_scores(Val_H)
val_mem = get_memory_score(Val_H)

th_recon = np.mean(val_recon) + 2 * np.std(val_recon)
th_mahal = np.mean(val_mahal) + 2 * np.std(val_mahal)
th_mem = np.mean(val_mem) + 2 * np.std(val_mem)

print(f"  🟢 Recon Threshold:  {th_recon:.6f}")
print(f"  🟢 Mahal Threshold:  {th_mahal:.4f}")
print(f"  🟢 Memory Threshold: {th_mem:.4f}")


# %% [Cell 6] Detection Rate Evaluation
results_real = {'cases':[], 'recon_rate':[], 'mahal_rate':[], 'mem_rate':[],
                'recon_mean':[], 'recon_std':[]}
results_synth = {'di':[], 'recon_rate':[], 'mahal_rate':[], 'mem_rate':[],
                 'recon_mean':[], 'recon_std':[]}

# POD용 (a, â) 쌍 수집
all_real_a, all_real_ahat = [], []
all_synth_a, all_synth_ahat = [], []

real_damage_data = {}
synth_damage_data = {}

print("\n📊 Real Damage Data:")
for case in cfg.DAMAGE_CASES_B:
    data_d = load_damage_data(os.path.join(cfg.DIR_B_RAW, f"D3_{case}_1.txt"), is_synth=False)
    if data_d is None: continue
    real_damage_data[case] = data_d

    r_s = get_recon_error(data_d)
    m_s = get_mahalanobis_scores(data_d)
    mem_s = get_memory_score(data_d)
    log_r_s = np.log(np.maximum(r_s, 1e-10))  # POD용

    results_real['cases'].append(case)
    results_real['recon_rate'].append(np.mean(r_s > th_recon) * 100.0)
    results_real['mahal_rate'].append(np.mean(m_s > th_mahal) * 100.0)
    results_real['mem_rate'].append(np.mean(mem_s > th_mem) * 100.0)
    results_real['recon_mean'].append(np.mean(r_s))
    results_real['recon_std'].append(np.std(r_s))

    # POD scatter data (log-recon 사용)
    all_real_a.extend([case] * len(log_r_s))
    all_real_ahat.extend(log_r_s.tolist())

    print(f"  Case {case:2d}% | Recon: {results_real['recon_rate'][-1]:5.1f}% | "
          f"Mahal: {results_real['mahal_rate'][-1]:5.1f}% | Mem: {results_real['mem_rate'][-1]:5.1f}%")

print("\n📊 Synthetic Damage Data:")
for di in cfg.SYNTHETIC_DI_STEPS:
    data_s = load_damage_data(os.path.join(cfg.SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
    results_synth['di'].append(di)
    if data_s is None:
        for k in ['recon_rate', 'mahal_rate', 'mem_rate', 'recon_mean', 'recon_std']:
            results_synth[k].append(np.nan)
        continue
    synth_damage_data[round(di, 2)] = data_s

    r_s = get_recon_error(data_s)
    m_s = get_mahalanobis_scores(data_s)
    mem_s = get_memory_score(data_s)
    log_r_s = np.log(np.maximum(r_s, 1e-10))

    results_synth['recon_rate'].append(np.mean(r_s > th_recon) * 100.0)
    results_synth['mahal_rate'].append(np.mean(m_s > th_mahal) * 100.0)
    results_synth['mem_rate'].append(np.mean(mem_s > th_mem) * 100.0)
    results_synth['recon_mean'].append(np.mean(r_s))
    results_synth['recon_std'].append(np.std(r_s))

    all_synth_a.extend([di * 100.0] * len(log_r_s))  # 0~100% 스케일
    all_synth_ahat.extend(log_r_s.tolist())

# 일부만 출력
for i, di in enumerate(results_synth['di']):
    if i % 10 == 0 and not np.isnan(results_synth['recon_rate'][i]):
        print(f"  DI {di:.2f} | Recon: {results_synth['recon_rate'][i]:5.1f}% | "
              f"Mahal: {results_synth['mahal_rate'][i]:5.1f}% | Mem: {results_synth['mem_rate'][i]:5.1f}%")

all_real_a = np.array(all_real_a)
all_real_ahat = np.array(all_real_ahat)
all_synth_a = np.array(all_synth_a)
all_synth_ahat = np.array(all_synth_ahat)


# %% [Cell 7] KS-test (분포 분리도)
print("\n=== KS-test (Score Separation Diagnostic) ===")
data_4 = real_damage_data.get(4)
data_48 = real_damage_data.get(48)
data_synth_10 = synth_damage_data.get(1.00)

print(f"\n📊 KS-test on Reconstruction Error:")
print(f"{'Comparison':<25} {'KS statistic'}")
print("-" * 50)
for label, data in [('Real 4%', data_4), ('Real 48%', data_48), ('Synth 1.0', data_synth_10)]:
    if data is not None:
        ks, _ = ks_2samp(val_recon, get_recon_error(data))
        print(f"Healthy vs {label:<15} {ks:.4f}")


# %% [Cell 8] Plot - Detection Rate
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(results_real['cases'], results_real['recon_rate'], 'r-o', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2, label='Reconstruction Error')
ax.plot(results_real['cases'], results_real['mahal_rate'], 'g-s', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2, label='Latent Mahalanobis')
ax.plot(results_real['cases'], results_real['mem_rate'], 'm-^', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2, label='Memory Distance (z-z_mem)')
ax.axhline(y=2.0, color='gray', linestyle='--', label='False Alarm Rate (2%)')
ax.axhline(y=15.0, color='green', linestyle=':', alpha=0.6, label='Goal (15%)')
ax.set_title("MemAE (Real): % Classified as Damage", fontweight='bold')
ax.set_xlabel("Damage Case (%)"); ax.set_ylabel("Detection Rate (%)")
ax.set_xticks(cfg.DAMAGE_CASES_B)
ymax = max(20, max(results_real['recon_rate'] + results_real['mahal_rate'] + results_real['mem_rate']) * 1.2)
ax.set_ylim([0, ymax])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend()

ax = axes[1]
ax.plot(results_synth['di'], results_synth['recon_rate'], 'r-o', linewidth=2, markersize=5,
        markerfacecolor='white', markeredgewidth=1.5, label='Reconstruction Error')
ax.plot(results_synth['di'], results_synth['mahal_rate'], 'g-s', linewidth=2, markersize=5,
        markerfacecolor='white', markeredgewidth=1.5, label='Latent Mahalanobis')
ax.plot(results_synth['di'], results_synth['mem_rate'], 'm-^', linewidth=2, markersize=5,
        markerfacecolor='white', markeredgewidth=1.5, label='Memory Distance')
ax.axhline(y=2.0, color='gray', linestyle='--', label='False Alarm Rate (2%)')
ax.axhline(y=15.0, color='green', linestyle=':', alpha=0.6, label='Goal (15%)')
ax.set_title("MemAE (Synthetic): % Classified as Damage", fontweight='bold')
ax.set_xlabel("Input Synthetic DI"); ax.set_ylabel("Detection Rate (%)")
ax.set_xticks(np.arange(0.0, 1.1, 0.2))
valid_synth = [v for v in results_synth['recon_rate'] + results_synth['mahal_rate'] + results_synth['mem_rate']
               if not np.isnan(v)]
ymax_s = max(20, max(valid_synth) * 1.2) if valid_synth else 20
ax.set_ylim([0, ymax_s])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "MemAE_detection_rate.png"), dpi=300)
plt.show()


# %% [Cell 9] Plot - Recon Error Trend
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.errorbar(results_real['cases'], results_real['recon_mean'], yerr=results_real['recon_std'],
            fmt='r-o', linewidth=2.5, markersize=8, capsize=5, label='Mean ± Std')
ax.axhline(y=th_recon, color='gray', linestyle='--', linewidth=2,
           label=f'Threshold ({th_recon:.4f})')
ax.set_title("MemAE Real: Recon Error per Damage Level", fontweight='bold')
ax.set_xlabel("Damage Case (%)"); ax.set_ylabel("Reconstruction Error")
ax.set_xticks(cfg.DAMAGE_CASES_B)
ax.grid(True, linestyle=':', alpha=0.7); ax.legend()

ax = axes[1]
ax.errorbar(results_synth['di'], results_synth['recon_mean'], yerr=results_synth['recon_std'],
            fmt='b-s', linewidth=2, markersize=5, capsize=3, alpha=0.7, label='Mean ± Std')
ax.axhline(y=th_recon, color='gray', linestyle='--', linewidth=2,
           label=f'Threshold ({th_recon:.4f})')
ax.set_title("MemAE Synthetic: Recon Error per DI", fontweight='bold')
ax.set_xlabel("Input Synthetic DI"); ax.set_ylabel("Reconstruction Error")
ax.grid(True, linestyle=':', alpha=0.7); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "MemAE_recon_error_trend.png"), dpi=300)
plt.show()


# %% [Cell 10] POD Analysis - MLE Fit
print("\n=== POD Analysis: Heteroscedastic Censored MLE ===")
log_th_recon = np.log(np.maximum(th_recon, 1e-10))  # log-scale threshold

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

def bootstrap_lcb(x, y, a_th, x_range, n_boot=300):
    n = len(x)
    pods = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        p, ok = fit_mle(x[idx], y[idx], a_th)
        if ok:
            pods.append(get_mean_pod(x_range, p, a_th))
    if not pods:
        return np.zeros_like(x_range)
    return np.percentile(np.array(pods), 5, axis=0)

x_range_r = np.linspace(0, 50, 200)
x_range_s = np.linspace(0, 100, 200)

# Real fit
params_r, ok_r = fit_mle(all_real_a, all_real_ahat, log_th_recon)
pod_mean_r = get_mean_pod(x_range_r, params_r, log_th_recon)
pod_lcb_r = bootstrap_lcb(all_real_a, all_real_ahat, log_th_recon, x_range_r)
a90_r = x_range_r[np.argmax(pod_mean_r >= 0.9)] if np.any(pod_mean_r >= 0.9) else np.nan
a90_95_r = x_range_r[np.argmax(pod_lcb_r >= 0.9)] if np.any(pod_lcb_r >= 0.9) else np.nan

# Synth fit
params_s, ok_s = fit_mle(all_synth_a, all_synth_ahat, log_th_recon)
pod_mean_s = get_mean_pod(x_range_s, params_s, log_th_recon)
pod_lcb_s = bootstrap_lcb(all_synth_a, all_synth_ahat, log_th_recon, x_range_s)
a90_s = x_range_s[np.argmax(pod_mean_s >= 0.9)] if np.any(pod_mean_s >= 0.9) else np.nan
a90_95_s = x_range_s[np.argmax(pod_lcb_s >= 0.9)] if np.any(pod_lcb_s >= 0.9) else np.nan

print(f"\n  ───── Real Data ─────")
print(f"   Slope b1: {params_r[1]:.6f}")
print(f"   a90 = {a90_r:.2f}%   a90/95 (LCB) = {a90_95_r:.2f}%")
print(f"  ───── Synth Data ────")
print(f"   Slope b1: {params_s[1]:.6f}")
print(f"   a90 = {a90_s:.2f}%   a90/95 (LCB) = {a90_95_s:.2f}%")


# %% [Cell 11] Plot - MLE Scatter
def plot_mle_scatter(ax, x_data, y_data, params, a_th, title, xlabel, x_limit, color):
    b0, b1, t0, t1 = params
    ax.scatter(x_data, y_data, alpha=0.15, color=color, s=10, label='Data points')
    x_r = np.linspace(0, x_limit, 100)
    mu_y = b0 + b1 * x_r
    tau_y = np.maximum(t0 + t1 * x_r, 1e-8)
    ax.plot(x_r, mu_y, 'k-', linewidth=2.5, label=f'MLE Mean')
    ax.plot(x_r, mu_y + 2*tau_y, 'k--', linewidth=1.5, alpha=0.7, label='$\\pm 2\\sigma$')
    ax.plot(x_r, mu_y - 2*tau_y, 'k--', linewidth=1.5, alpha=0.7)
    ax.fill_between(x_r, mu_y - 2*tau_y, mu_y + 2*tau_y, color='gray', alpha=0.1)
    ax.axhline(y=a_th, color='red', linestyle='-', linewidth=2.5,
               label=f'Threshold (log) = {a_th:.2f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel("Log Reconstruction Error, $\\hat{a}$")
    ax.set_xlim([0, x_limit])
    ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='upper left', fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_mle_scatter(axes[0], all_real_a, all_real_ahat, params_r, log_th_recon,
                 "Real Data: â vs a + MLE Fit", "Damage Case, $a$ (%)", 50, '#E74C3C')
plot_mle_scatter(axes[1], all_synth_a, all_synth_ahat, params_s, log_th_recon,
                 "Synthetic Data: â vs a + MLE Fit", "Mapped Damage, $a$ (%)", 100, '#3498DB')
plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "MemAE_MLE_Scatter.png"), dpi=300)
plt.show()


# %% [Cell 12] Plot - POD Curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(x_range_r, pod_mean_r, color='#E74C3C', linewidth=3,
        label=f'Mean POD ($a_{{90}}$={a90_r:.1f}%)' if not np.isnan(a90_r) else 'Mean POD ($a_{90}$ N/A)')
ax.plot(x_range_r, pod_lcb_r, color='#E74C3C', linewidth=2, linestyle='--',
        label=f'95% LCB ($a_{{90/95}}$={a90_95_r:.1f}%)' if not np.isnan(a90_95_r) else '95% LCB (N/A)')
ax.axhline(0.9, color='black', linestyle=':', linewidth=1.5, label='90% Target')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
if not np.isnan(a90_r):
    ax.axvline(a90_r, color='#E74C3C', linestyle=':', alpha=0.5)
ax.set_title("MemAE Real: POD Curve", fontweight='bold')
ax.set_xlabel("Damage Case, $a$ (%)"); ax.set_ylabel("Probability of Detection")
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 50])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right')

ax = axes[1]
ax.plot(x_range_s, pod_mean_s, color='#3498DB', linewidth=3,
        label=f'Mean POD ($a_{{90}}$={a90_s:.1f}%)' if not np.isnan(a90_s) else 'Mean POD ($a_{90}$ N/A)')
ax.plot(x_range_s, pod_lcb_s, color='#3498DB', linewidth=2, linestyle='--',
        label=f'95% LCB ($a_{{90/95}}$={a90_95_s:.1f}%)' if not np.isnan(a90_95_s) else '95% LCB (N/A)')
ax.axhline(0.9, color='black', linestyle=':', linewidth=1.5, label='90% Target')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
if not np.isnan(a90_s):
    ax.axvline(a90_s, color='#3498DB', linestyle=':', alpha=0.5)
ax.set_title("MemAE Synthetic: POD Curve", fontweight='bold')
ax.set_xlabel("Mapped Damage, $a$ (%)"); ax.set_ylabel("Probability of Detection")
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cfg.SAVE_DIR, "MemAE_POD_Curves.png"), dpi=300)
plt.show()


# %% [Cell 13] Save All Results & Summary
all_results = {
    'config': {
        'n_feat': cfg.N_FEAT, 'kernel_size': cfg.KERNEL_SIZE, 'stride': cfg.STRIDE,
        'alpha': cfg.ALPHA, 'latent_dim': cfg.LATENT_DIM, 'lr': cfg.LR,
        'epochs': cfg.EPOCHS, 'batch_size': cfg.BATCH_SIZE,
        'n_memory_slots': cfg.N_MEMORY_SLOTS,
        'shrink_threshold': cfg.MEMORY_SHRINK_THRESHOLD,
        'lambda_entropy': cfg.LAMBDA_ENTROPY,
        'seed': cfg.SEED
    },
    'training_history': history,
    'thresholds': {
        'recon': th_recon, 'mahal': th_mahal, 'memory': th_mem,
        'log_recon': log_th_recon
    },
    'results_real': results_real,
    'results_synth': results_synth,
    'pod_real': {
        'params': params_r, 'a90': a90_r, 'a90_95': a90_95_r,
        'x_range': x_range_r, 'pod_mean': pod_mean_r, 'pod_lcb': pod_lcb_r
    },
    'pod_synth': {
        'params': params_s, 'a90': a90_s, 'a90_95': a90_95_s,
        'x_range': x_range_s, 'pod_mean': pod_mean_s, 'pod_lcb': pod_lcb_s
    }
}

with open(os.path.join(cfg.SAVE_DIR, "MemAE_full_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)

print("\n" + "=" * 70)
print("✅ MemAE Full Pipeline Complete!")
print("=" * 70)
print(f"\n📂 결과 저장 폴더: {cfg.SAVE_DIR}")
print(f"\n📊 Detection Rate Summary (Real):")
print(f"   Case  4%: Recon={results_real['recon_rate'][0]:.1f}%, "
      f"Mahal={results_real['mahal_rate'][0]:.1f}%, "
      f"Mem={results_real['mem_rate'][0]:.1f}%")
print(f"   Case 48%: Recon={results_real['recon_rate'][-1]:.1f}%, "
      f"Mahal={results_real['mahal_rate'][-1]:.1f}%, "
      f"Mem={results_real['mem_rate'][-1]:.1f}%")
print(f"\n🎯 POD Analysis:")
if not np.isnan(a90_r):
    print(f"   Real:  a_90 = {a90_r:.2f}%  (LCB: {a90_95_r:.2f}%)")
else:
    print(f"   Real:  a_90 NOT reached within 50%")
if not np.isnan(a90_s):
    print(f"   Synth: a_90 = {a90_s:.2f}%  (LCB: {a90_95_s:.2f}%)")
else:
    print(f"   Synth: a_90 NOT reached within 100%")

print(f"\n💾 생성된 파일:")
print(f"   - MemAE1_best.pth")
print(f"   - MemAE1_training_curves.png")
print(f"   - MemAE1_detection_rate.png")
print(f"   - MemAE1_recon_error_trend.png")
print(f"   - MemAE1_MLE_Scatter.png")
print(f"   - MemAE1_POD_Curves.png")
print(f"   - MemAE1_full_results.pkl")
print("=" * 70)