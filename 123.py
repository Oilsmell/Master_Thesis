# ================================================================
# UNIFIED POD PIPELINE — Multi-Seed Averaged Scores
# 각 모델 5회 학습 → score 평균 → 평균값 기반 단일 POD 계산
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel as skl_rbf_kernel
from scipy.spatial.distance import pdist
from scipy.signal import hilbert
from scipy.stats import norm, linregress
from scipy.optimize import minimize
import optuna
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ================================================================
# [0] 공통 설정
# ================================================================
DIR_B_RAW       = r"E:\2ndstructuredata\raw data"
FILE_B          = "healthyclean.txt"
SYNTH_DATA_DIR  = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"

SAVE_DIR_GAN    = r"E:\2ndstructuredata\Code_22_Optimized_MMDGAN"
SAVE_DIR_AE     = r"E:\2ndstructuredata\Code_34_MemAE_Baseline_v2_shrink002"
SAVE_DIR_SVM    = r"E:\2ndstructuredata\Code_MMD_Optuna"
SAVE_DIR_COMP   = r"E:\2ndstructuredata\Unified_POD_MultiSeed_Averaged"

for d in [SAVE_DIR_GAN, SAVE_DIR_AE, SAVE_DIR_SVM, SAVE_DIR_COMP]:
    os.makedirs(d, exist_ok=True)

WINDOW_SIZE        = 128
CHANNELS           = 8
DAMAGE_CASES_B     = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
SYNTHETIC_DI_STEPS = np.round(np.arange(0.0, 1.01, 0.02), 2)
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 5 seed 정의
SEEDS = [42, 123, 777, 2024, 9999]

# POD 평가 X축
x_range_r = np.linspace(0, 50, 200)
x_range_s = np.linspace(0, 100, 200)

print("=" * 70)
print("  UNIFIED POD PIPELINE — Multi-Seed Score Averaging")
print(f"  Device: {device}")
print(f"  Seeds: {SEEDS}")
print("=" * 70)


# ================================================================
# [1] 공통 POD 함수 (Heteroscedastic Censored MLE + Bootstrap)
# ================================================================
def _mll_loss(params, x, y, a_th):
    b0, b1, tau0, tau1 = params
    mu_y = b0 + b1 * x
    tau  = np.maximum(tau0 + tau1 * x, 1e-8)
    cens   = y <= a_th
    uncens = ~cens
    ll  = norm.logpdf(y[uncens], loc=mu_y[uncens], scale=tau[uncens])
    llc = norm.logcdf(a_th,      loc=mu_y[cens],   scale=tau[cens])
    return -(np.sum(ll) + np.sum(llc))

def fit_mle_pod(x, y, a_th):
    slope, intercept, *_ = linregress(x, y)
    init   = [intercept, slope, np.std(y), 0.0]
    bounds = [(None, None), (None, None), (1e-8, None), (0.0, None)]
    res    = minimize(_mll_loss, init, args=(x, y, a_th),
                      bounds=bounds, method='L-BFGS-B')
    return res.x, res.success

def get_mean_pod(x_range, params, a_th):
    b0, b1, t0, t1 = params
    mu  = b0 + b1 * x_range
    tau = np.maximum(t0 + t1 * x_range, 1e-8)
    return norm.cdf((mu - a_th) / tau)

def bootstrap_lcb(x, y, a_th, x_range, n_boot=300, block_size=None):
    n    = len(x)
    pods = []
    if block_size is None:
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            p, ok = fit_mle_pod(x[idx], y[idx], a_th)
            if ok:
                pods.append(get_mean_pod(x_range, p, a_th))
    else:
        n_blocks = max(1, n // block_size)
        for _ in range(n_boot):
            starts = np.random.choice(max(1, n - block_size + 1), n_blocks, replace=True)
            idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])
            idx    = idx[idx < n]
            if len(idx) < 10: continue
            p, ok  = fit_mle_pod(x[idx], y[idx], a_th)
            if ok:
                pods.append(get_mean_pod(x_range, p, a_th))
    if not pods:
        return np.zeros_like(x_range)
    return np.percentile(np.array(pods), 5, axis=0)

def compute_pod(all_a, all_ahat, a_th, x_range, n_boot=300, block_size=None):
    """POD 전체 계산: (pod_mean, pod_lcb, a90, a90_95, params)"""
    params, _  = fit_mle_pod(all_a, all_ahat, a_th)
    pod_mean   = get_mean_pod(x_range, params, a_th)
    pod_lcb    = bootstrap_lcb(all_a, all_ahat, a_th, x_range, n_boot, block_size)
    a90        = x_range[np.argmax(pod_mean >= 0.9)] if np.any(pod_mean >= 0.9) else np.nan
    a90_95     = x_range[np.argmax(pod_lcb  >= 0.9)] if np.any(pod_lcb  >= 0.9) else np.nan
    return pod_mean, pod_lcb, a90, a90_95, params


# ================================================================
# [2] GAN/AE 공통 데이터 로더 (MinMaxScaler, [-1,1] 정규화)
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
    idx      = np.random.permutation(ns)
    Train_H  = norm_data[idx[:TRAIN_SIZE]]
    Val_H    = norm_data[idx[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]]
    print(f"✅ GAN/AE 건강 데이터: Train {Train_H.shape}, Val {Val_H.shape}")

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

# 💡 손상 데이터를 한 번만 로드해서 캐싱 (5 seed 동안 재사용)
print("\n📦 손상 데이터 캐싱 중...")
cached_real_damage = {}
for case in DAMAGE_CASES_B:
    cached_real_damage[case] = load_damage_ganae(os.path.join(DIR_B_RAW, f"D3_{case}_1.txt"))

cached_synth_damage = {}
for di in SYNTHETIC_DI_STEPS:
    cached_synth_damage[round(di, 2)] = load_damage_ganae(
        os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
print(f"   Real: {sum(1 for v in cached_real_damage.values() if v is not None)} / "
      f"Synth: {sum(1 for v in cached_synth_damage.values() if v is not None)}")


# ================================================================
# [3] MMD-GAN 모델 정의
# ================================================================
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
        self.conv1 = nn.ConvTranspose1d(GAN_N1, GAN_N1, GAN_KERNEL_SIZE, stride=2,
                                        padding=GAN_KERNEL_SIZE//2, output_padding=1)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2 = nn.ConvTranspose1d(GAN_N1, CHANNELS, GAN_KERNEL_SIZE, stride=2,
                                        padding=GAN_KERNEL_SIZE//2, output_padding=1)
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
        self.conv1 = nn.Conv1d(CHANNELS, GAN_N2, GAN_KERNEL_SIZE,
                               stride=GAN_STRIDE, padding=GAN_KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2 = nn.Conv1d(GAN_N2, GAN_N2, GAN_KERNEL_SIZE,
                               stride=GAN_STRIDE, padding=GAN_KERNEL_SIZE//2)
        self.lrelu2 = nn.LeakyReLU(GAN_ALPHA)
        self.flatten = nn.Flatten()
        dummy = torch.zeros(1, CHANNELS, 128)
        out = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.fc = nn.Linear(out.view(1, -1).size(1), GAN_FEAT_DIM)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(self.flatten(x))


def train_gan_one_seed(seed):
    """GAN을 한 seed로 학습하고 모델 + Mahalanobis 통계 반환"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    gen    = GAN_Generator().to(device)
    critic = GAN_Critic().to(device)
    opt_g  = optim.Adam(gen.parameters(), lr=GAN_LR)
    opt_c  = optim.Adam(critic.parameters(), lr=GAN_LR)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)),
        batch_size=GAN_BATCH, shuffle=True, drop_last=True)

    for epoch in range(GAN_EPOCHS):
        for (real_data,) in loader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)
            opt_c.zero_grad()
            z = torch.randn(b_size, GAN_LATENT_DIM).to(device)
            fake_data = gen(z)
            feat_real = critic(real_data)
            feat_fake = critic(fake_data.detach())
            (-mmd_loss_calc(feat_real, feat_fake)).backward()
            opt_c.step()
            opt_g.zero_grad()
            feat_fake_g = critic(gen(z))
            mmd_loss_calc(feat_real.detach(), feat_fake_g).backward()
            opt_g.step()

    # Mahalanobis 통계
    critic.eval()
    with torch.no_grad():
        feat_train = critic(torch.FloatTensor(Train_H).to(device)).cpu().numpy()
    mu_h    = np.mean(feat_train, axis=0)
    cov_h   = np.cov(feat_train, rowvar=False)
    cov_inv = np.linalg.inv(cov_h + 1e-5 * np.eye(GAN_FEAT_DIM))

    return critic, mu_h, cov_inv


def gan_score_for_data(critic, mu_h, cov_inv, data):
    with torch.no_grad():
        feat = critic(torch.FloatTensor(data).to(device)).detach().cpu().numpy()
    diff = feat - mu_h
    mahal_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
    return np.log(np.sqrt(np.maximum(mahal_sq, 1e-10)))


# ================================================================
# [4] MemAE 모델 정의
# ================================================================
AE_N_FEAT          = 19
AE_KERNEL_SIZE     = 6
AE_STRIDE          = 3
AE_ALPHA           = 0.3799
AE_LATENT_DIM      = 64
AE_N_MEMORY_SLOTS  = 50
AE_SHRINK_TH       = 0.02
AE_LAMBDA_ENTROPY  = 0.0002
AE_LR              = 0.001
AE_EPOCHS          = 112
AE_BATCH           = 96

class AE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, AE_N_FEAT, AE_KERNEL_SIZE,
                               stride=AE_STRIDE, padding=AE_KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(AE_ALPHA)
        self.conv2 = nn.Conv1d(AE_N_FEAT, AE_N_FEAT, AE_KERNEL_SIZE,
                               stride=AE_STRIDE, padding=AE_KERNEL_SIZE//2)
        self.lrelu2 = nn.LeakyReLU(AE_ALPHA)
        dummy = torch.zeros(1, CHANNELS, 128)
        out = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.conv_out_len = out.size(2)
        self.flat_size = out.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_size, AE_LATENT_DIM)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(x.view(x.size(0), -1))

class AE_Decoder(nn.Module):
    def __init__(self, conv_out_len, flat_size):
        super().__init__()
        self.conv_out_len = conv_out_len
        self.fc = nn.Linear(AE_LATENT_DIM, flat_size)
        self.lrelu0 = nn.LeakyReLU(AE_ALPHA)
        self.deconv1 = nn.ConvTranspose1d(AE_N_FEAT, AE_N_FEAT, AE_KERNEL_SIZE,
                                          stride=AE_STRIDE, padding=AE_KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(AE_ALPHA)
        self.deconv2 = nn.ConvTranspose1d(AE_N_FEAT, CHANNELS, AE_KERNEL_SIZE,
                                          stride=AE_STRIDE, padding=AE_KERNEL_SIZE//2)
        self.tanh = nn.Tanh()
    def forward(self, z):
        x = self.lrelu0(self.fc(z))
        x = x.view(-1, AE_N_FEAT, self.conv_out_len)
        x = self.lrelu1(self.deconv1(x))
        x = self.tanh(self.deconv2(x))
        if x.size(2) < 128: x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

class AE_MemoryModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.Parameter(torch.empty(AE_N_MEMORY_SLOTS, AE_LATENT_DIM))
        nn.init.xavier_normal_(self.memory)
    def forward(self, z):
        z_norm   = F.normalize(z, p=2, dim=1)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        sim      = torch.matmul(z_norm, mem_norm.t())
        attention = F.softmax(sim, dim=1)
        if AE_SHRINK_TH > 0:
            attention = (F.relu(attention - AE_SHRINK_TH) * attention) / \
                        (torch.abs(attention - AE_SHRINK_TH) + 1e-12)
            attention = F.normalize(attention, p=1, dim=1)
        z_mem = torch.matmul(attention, self.memory)
        return z_mem, attention

class MemAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AE_Encoder()
        self.decoder = AE_Decoder(self.encoder.conv_out_len, self.encoder.flat_size)
        self.memory  = AE_MemoryModule()
    def forward(self, x):
        z = self.encoder(x)
        z_mem, attention = self.memory(z)
        recon = self.decoder(z_mem)
        return recon, z, z_mem, attention


def train_memae_one_seed(seed):
    """MemAE를 한 seed로 학습하고 모델 반환"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    memae    = MemAE().to(device)
    opt_ae   = optim.Adam(memae.parameters(), lr=AE_LR)
    mse_loss = nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)),
        batch_size=AE_BATCH, shuffle=True, drop_last=True)

    for epoch in range(AE_EPOCHS):
        memae.train()
        for (real_data,) in loader:
            real_data = real_data.to(device)
            opt_ae.zero_grad()
            recon, z, z_mem, attention = memae(real_data)
            r_loss = mse_loss(recon, real_data)
            e_loss = torch.mean(-torch.sum(attention * torch.log(attention + 1e-12), dim=1))
            (r_loss + AE_LAMBDA_ENTROPY * e_loss).backward()
            opt_ae.step()

    return memae


def ae_score_for_data(memae, data):
    memae.eval()
    with torch.no_grad():
        t = torch.FloatTensor(data).to(device)
        recon, _, _, _ = memae(t)
        err = torch.mean((recon - t) ** 2, dim=(1, 2)).cpu().numpy()
    return np.log(np.maximum(err, 1e-10))


# ================================================================
# [5] Multi-Seed 학습 + Score 평균화 (GAN, MemAE)
# ================================================================
def multi_seed_score_averaging(model_name, train_func, score_func):
    """
    각 seed로 학습 → 각 case의 score 추출 → 5 seed 평균 score 계산
    
    Returns:
        val_avg_scores: (n_val_samples,) — Val 데이터의 평균 score
        real_avg_per_case: {case: (n_samples,)} — case별 평균 score
        synth_avg_per_di: {di: (n_samples,)} — DI별 평균 score
    """
    print(f"\n{'='*70}")
    print(f"  [{model_name}] Multi-Seed Score Averaging ({len(SEEDS)} seeds)")
    print(f"{'='*70}")
    
    # Score 누적 저장
    val_scores_per_seed = []
    real_scores_per_case_per_seed = {case: [] for case in DAMAGE_CASES_B if cached_real_damage[case] is not None}
    synth_scores_per_di_per_seed = {round(di,2): [] for di in SYNTHETIC_DI_STEPS 
                                      if cached_synth_damage.get(round(di,2)) is not None}
    
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  🎲 Seed {seed_idx+1}/{len(SEEDS)} = {seed} | Training...")
        
        # 한 seed 학습
        if model_name == 'MMD-GAN':
            critic, mu_h, cov_inv = train_func(seed)
            # Val
            v_score = score_func(critic, mu_h, cov_inv, Val_H)
            val_scores_per_seed.append(v_score)
            # Real
            for case in real_scores_per_case_per_seed.keys():
                s = score_func(critic, mu_h, cov_inv, cached_real_damage[case])
                real_scores_per_case_per_seed[case].append(s)
            # Synth
            for di_key in synth_scores_per_di_per_seed.keys():
                s = score_func(critic, mu_h, cov_inv, cached_synth_damage[di_key])
                synth_scores_per_di_per_seed[di_key].append(s)
            del critic
        else:  # MemAE
            memae = train_func(seed)
            v_score = score_func(memae, Val_H)
            val_scores_per_seed.append(v_score)
            for case in real_scores_per_case_per_seed.keys():
                s = score_func(memae, cached_real_damage[case])
                real_scores_per_case_per_seed[case].append(s)
            for di_key in synth_scores_per_di_per_seed.keys():
                s = score_func(memae, cached_synth_damage[di_key])
                synth_scores_per_di_per_seed[di_key].append(s)
            del memae
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    Seed {seed} 완료")
    
    # 💡 핵심: 같은 sample index에 대해 5 seed score 평균
    # 각 seed에서 score 추출 시 동일한 데이터 순서로 추출되므로 element-wise 평균 가능
    
    # Val 평균
    val_avg_scores = np.mean(np.stack(val_scores_per_seed, axis=0), axis=0)
    
    # Real 평균 (case별)
    real_avg_per_case = {}
    for case, score_list in real_scores_per_case_per_seed.items():
        real_avg_per_case[case] = np.mean(np.stack(score_list, axis=0), axis=0)
    
    # Synth 평균 (DI별)
    synth_avg_per_di = {}
    for di_key, score_list in synth_scores_per_di_per_seed.items():
        synth_avg_per_di[di_key] = np.mean(np.stack(score_list, axis=0), axis=0)
    
    print(f"\n  ✅ {model_name} 평균화 완료")
    return val_avg_scores, real_avg_per_case, synth_avg_per_di


# ── 5.1 MMD-GAN 실행
gan_val_avg, gan_real_avg, gan_synth_avg = multi_seed_score_averaging(
    'MMD-GAN', train_gan_one_seed, gan_score_for_data
)

# ── 5.2 MemAE 실행
ae_val_avg, ae_real_avg, ae_synth_avg = multi_seed_score_averaging(
    'MemAE', train_memae_one_seed, ae_score_for_data
)


# ================================================================
# [6] MK-MMD (Optuna) — 5 seed × random_state 변경
# ================================================================
print("\n" + "=" * 70)
print("  [MK-MMD] Multi-Seed Score Averaging")
print("=" * 70)

SVM_NUM_SAMPLES   = 1000
SVM_PFA           = 0.05
SVM_OPTUNA_TRIAL  = 30  # 시간 절약

def _windowize_svm(data):
    ns = data.shape[1] // WINDOW_SIZE
    if ns == 0: return None, 0
    data = data[:, :ns * WINDOW_SIZE]
    return data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2), ns

def load_data_svm(path, is_synth=False, seed_for_sampling=42):
    try:
        if is_synth:
            data = np.loadtxt(path, delimiter='\t').T.astype(np.float32)
        else:
            raw  = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
        windows, ns = _windowize_svm(data)
        if windows is None: return None, 0
        if ns > SVM_NUM_SAMPLES and "healthyclean" not in path.lower():
            np.random.seed(seed_for_sampling)
            windows = windows[np.random.choice(ns, SVM_NUM_SAMPLES, replace=False)]
            ns = SVM_NUM_SAMPLES
        return windows, ns
    except Exception as e:
        return None, 0

def extract_features_svm(data):
    if len(data) == 0: return np.array([])
    mean_v = np.mean(data, axis=2); std_v  = np.std(data, axis=2)
    rms_v  = np.sqrt(np.mean(data**2, axis=2))
    peak_v = np.max(np.abs(data), axis=2); p2p_v = np.ptp(data, axis=2)
    crest_v = peak_v / (rms_v + 1e-10)
    centered = data - mean_v[:, :, None]
    skew_v = np.mean(centered**3, axis=2) / (std_v**3 + 1e-10)
    kurt_v = np.mean(centered**4, axis=2) / (std_v**4 + 1e-10)
    zcr_v  = np.mean(np.abs(np.diff(np.sign(data), axis=2)), axis=2) / 2
    fft_mag  = np.abs(np.fft.rfft(data, axis=-1))
    freqs    = np.fft.rfftfreq(WINDOW_SIZE)
    fft_sum  = np.sum(fft_mag, axis=2) + 1e-10
    centroid = np.sum(freqs * fft_mag, axis=2) / fft_sum
    spread   = np.sqrt(np.sum(((freqs - centroid[..., None])**2) * fft_mag, axis=2) / fft_sum)
    p_norm   = fft_mag / fft_sum[..., None]
    sp_ent   = -np.sum(p_norm * np.log2(p_norm + 1e-10), axis=2)
    hf_ratio = np.sum(fft_mag[:, :, fft_mag.shape[-1]//2:], axis=2) / fft_sum
    n_bands = 4
    band_edges = np.linspace(0, fft_mag.shape[2], n_bands + 1, dtype=int)
    band_ratios = []
    for i in range(n_bands):
        band_e = np.sum(fft_mag[:, :, band_edges[i]:band_edges[i+1]], axis=2)
        band_ratios.append(band_e / fft_sum)
    env = np.abs(hilbert(data, axis=-1))
    env_rms = np.sqrt(np.mean(env**2, axis=2))
    env_mean = np.mean(env, axis=2); env_std = np.std(env, axis=2)
    env_kurt = np.mean((env - env_mean[..., None])**4, axis=2) / (env_std**4 + 1e-10)
    N = data.shape[0]; corr_feats = np.zeros((N, 28))
    triu_idx = np.triu_indices(CHANNELS, k=1)
    for i in range(N): corr_feats[i] = np.corrcoef(data[i])[triu_idx]
    return np.concatenate([mean_v, std_v, rms_v, peak_v, p2p_v, crest_v, skew_v, kurt_v, zcr_v,
                           centroid, spread, sp_ent, hf_ratio, np.concatenate(band_ratios, axis=1),
                           env_rms, env_mean, env_std, env_kurt, corr_feats], axis=1)

def compute_mmd_stable(X, Y, base_gamma):
    gammas = [base_gamma * 0.1, base_gamma, base_gamma * 10]
    total = 0.0
    for g in gammas:
        XX = skl_rbf_kernel(X, X, g); YY = skl_rbf_kernel(Y, Y, g); XY = skl_rbf_kernel(X, Y, g)
        total += (XX.mean() + YY.mean() - 2 * XY.mean())
    return total

def get_mmd_scores_sliding(features, ref_dist, chunk_size, base_gamma, stride=2):
    scores = []
    n = len(features)
    if n < chunk_size: return np.array([])
    for i in range(0, n - chunk_size + 1, stride):
        chunk = features[i:i + chunk_size]
        scores.append(compute_mmd_stable(ref_dist, chunk, base_gamma))
    return np.array(scores)


def run_mkmmd_one_seed(seed):
    """MK-MMD을 한 seed로 학습 (Optuna + scoring)"""
    print(f"\n  🎲 MK-MMD Seed = {seed}")
    
    np.random.seed(seed)
    
    # Health 데이터 로드 (seed 고정)
    all_h, _ = load_data_svm(os.path.join(DIR_B_RAW, FILE_B), seed_for_sampling=seed)
    np.random.seed(seed)
    np.random.shuffle(all_h)
    train_h = all_h[:int(len(all_h)*0.7)]
    val_h   = all_h[int(len(all_h)*0.7):]
    
    # Feature extraction
    train_feat = extract_features_svm(train_h)
    val_feat   = extract_features_svm(val_h)
    
    # Target (DI 0.50)
    target_raw, _ = load_data_svm(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.50.txt"),
                                   is_synth=True, seed_for_sampling=seed)
    if target_raw is None:
        target_raw, _ = load_data_svm(os.path.join(DIR_B_RAW, "D3_48_1.txt"), seed_for_sampling=seed)
    target_feat = extract_features_svm(target_raw)
    
    # Scaling
    scaler = RobustScaler()
    train_sc = scaler.fit_transform(train_feat)
    val_sc   = scaler.transform(val_feat)
    target_sc = scaler.transform(target_feat)
    
    # Optuna
    def objective(trial):
        pca_comp = trial.suggest_int('pca_comp', 15, 60)
        chunk_size = trial.suggest_int('chunk_size', 20, 80, step=10)
        gamma_mult = trial.suggest_float('gamma_mult', 0.01, 10.0, log=True)
        pca = PCA(n_components=pca_comp, random_state=seed)
        train_pca = pca.fit_transform(train_sc)
        val_pca   = pca.transform(val_sc); tgt_pca = pca.transform(target_sc)
        distances = pdist(train_pca, metric='sqeuclidean')
        base_gamma = (1.0 / (np.median(distances) + 1e-10)) * gamma_mult
        stride_eval = chunk_size // 2
        v_scores = get_mmd_scores_sliding(val_pca, train_pca, chunk_size, base_gamma, stride=stride_eval)
        t_scores = get_mmd_scores_sliding(tgt_pca, train_pca, chunk_size, base_gamma, stride=stride_eval)
        if len(v_scores) < 5 or len(t_scores) < 5: return -999.0
        return (np.mean(t_scores) - np.mean(v_scores)) / (np.std(v_scores) + 1e-6)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=SVM_OPTUNA_TRIAL, show_progress_bar=False)
    best = study.best_params
    
    # 최적 적용
    pca_final = PCA(n_components=best['pca_comp'], random_state=seed)
    train_final = pca_final.fit_transform(train_sc)
    val_final   = pca_final.transform(val_sc)
    base_gamma  = 1.0 / (np.median(pdist(train_final, metric='sqeuclidean')) + 1e-10)
    final_gamma = base_gamma * best['gamma_mult']
    final_chunk = best['chunk_size']
    
    # Val scores
    val_scores = get_mmd_scores_sliding(val_final, train_final, final_chunk, final_gamma, stride=2)
    
    # Real scores per case
    real_scores = {}
    for case in DAMAGE_CASES_B:
        d, _ = load_data_svm(os.path.join(DIR_B_RAW, f"D3_{case}_1.txt"), seed_for_sampling=seed)
        if d is not None:
            feat = pca_final.transform(scaler.transform(extract_features_svm(d)))
            real_scores[case] = get_mmd_scores_sliding(feat, train_final, final_chunk, final_gamma, stride=1)
    
    # Synth scores per DI
    synth_scores = {}
    for di in SYNTHETIC_DI_STEPS:
        d, _ = load_data_svm(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"),
                              is_synth=True, seed_for_sampling=seed)
        if d is not None:
            feat = pca_final.transform(scaler.transform(extract_features_svm(d)))
            synth_scores[round(di,2)] = get_mmd_scores_sliding(feat, train_final, final_chunk,
                                                                 final_gamma, stride=1)
    
    print(f"    ✅ Seed {seed}: chunk={final_chunk}, n_val_scores={len(val_scores)}")
    return val_scores, real_scores, synth_scores, final_chunk

# ── 5 seed 실행 + score 길이 통일 (MK-MMD는 chunk_size마다 score 개수 다름)
svm_val_all = []
svm_real_all = {case: [] for case in DAMAGE_CASES_B}
svm_synth_all = {round(di,2): [] for di in SYNTHETIC_DI_STEPS}
svm_chunks = []

for seed in SEEDS:
    val_s, real_s, synth_s, chunk = run_mkmmd_one_seed(seed)
    svm_val_all.append(val_s)
    svm_chunks.append(chunk)
    for case, s in real_s.items():
        svm_real_all[case].append(s)
    for di_k, s in synth_s.items():
        svm_synth_all[di_k].append(s)

# 💡 길이 맞추기: 각 seed별로 score 개수가 다를 수 있으므로 최소 길이로 truncate
def truncate_to_min(score_list):
    min_len = min(len(s) for s in score_list)
    return np.stack([s[:min_len] for s in score_list], axis=0)

# Val 평균
svm_val_avg = np.mean(truncate_to_min(svm_val_all), axis=0)

# Real 평균
svm_real_avg = {}
for case, score_list in svm_real_all.items():
    if score_list and all(len(s) > 0 for s in score_list):
        svm_real_avg[case] = np.mean(truncate_to_min(score_list), axis=0)

# Synth 평균
svm_synth_avg = {}
for di_k, score_list in svm_synth_all.items():
    if score_list and all(len(s) > 0 for s in score_list):
        svm_synth_avg[di_k] = np.mean(truncate_to_min(score_list), axis=0)

print(f"\n  ✅ MK-MMD 평균화 완료 (avg chunk size: {np.mean(svm_chunks):.1f})")


# ================================================================
# [7] 평균 score를 사용한 POD 계산 (모든 모델 공통)
# ================================================================
print("\n" + "=" * 70)
print("  [7] 평균 Score 기반 POD 계산")
print("=" * 70)

def build_pod_dataset_from_avg(real_avg_dict, synth_avg_dict):
    """평균 score 딕셔너리에서 (a, â) 배열 생성"""
    real_a, real_ahat = [], []
    for case, scores in real_avg_dict.items():
        real_a.extend([case] * len(scores))
        real_ahat.extend(scores.tolist())
    
    synth_a, synth_ahat = [], []
    for di_k, scores in synth_avg_dict.items():
        synth_a.extend([di_k * 100.0] * len(scores))
        synth_ahat.extend(scores.tolist())
    
    return np.array(real_a), np.array(real_ahat), np.array(synth_a), np.array(synth_ahat)


# ── 7.1 MMD-GAN POD
gan_thr_real = np.mean(gan_val_avg) + 2 * np.std(gan_val_avg)
# Synth threshold: synth DI 0.00 데이터의 평균 score 기반
if 0.00 in gan_synth_avg:
    gan_thr_synth = np.mean(gan_synth_avg[0.00]) + 2 * np.std(gan_synth_avg[0.00])
else:
    gan_thr_synth = gan_thr_real

gan_real_a, gan_real_ahat, gan_synth_a, gan_synth_ahat = \
    build_pod_dataset_from_avg(gan_real_avg, gan_synth_avg)

print("  MMD-GAN POD 계산 중...")
gan_pod_mean_r, gan_pod_lcb_r, gan_a90_r, gan_a90_95_r, gan_params_r = \
    compute_pod(gan_real_a, gan_real_ahat, gan_thr_real, x_range_r)
gan_pod_mean_s, gan_pod_lcb_s, gan_a90_s, gan_a90_95_s, gan_params_s = \
    compute_pod(gan_synth_a, gan_synth_ahat, gan_thr_synth, x_range_s)

print(f"  ✅ MMD-GAN | Real a90={gan_a90_r:.1f}% | Synth a90={gan_a90_s:.1f}%")


# ── 7.2 MemAE POD
ae_log_thr = np.mean(ae_val_avg) + 2 * np.std(ae_val_avg)

ae_real_a, ae_real_ahat, ae_synth_a, ae_synth_ahat = \
    build_pod_dataset_from_avg(ae_real_avg, ae_synth_avg)

print("  MemAE POD 계산 중...")
ae_pod_mean_r, ae_pod_lcb_r, ae_a90_r, ae_a90_95_r, ae_params_r = \
    compute_pod(ae_real_a, ae_real_ahat, ae_log_thr, x_range_r)
ae_pod_mean_s, ae_pod_lcb_s, ae_a90_s, ae_a90_95_s, ae_params_s = \
    compute_pod(ae_synth_a, ae_synth_ahat, ae_log_thr, x_range_s)

print(f"  ✅ MemAE | Real a90={ae_a90_r:.1f}% | Synth a90={ae_a90_s:.1f}%")


# ── 7.3 MK-MMD POD
svm_threshold = np.percentile(svm_val_avg, (1 - SVM_PFA) * 100)
svm_real_a, svm_real_ahat, svm_synth_a, svm_synth_ahat = \
    build_pod_dataset_from_avg(svm_real_avg, svm_synth_avg)

avg_chunk = int(np.mean(svm_chunks))
print("  MK-MMD POD 계산 중 (block bootstrap)...")
svm_pod_mean_r, svm_pod_lcb_r, svm_a90_r, svm_a90_95_r, svm_params_r = \
    compute_pod(svm_real_a, svm_real_ahat, svm_threshold, x_range_r, block_size=avg_chunk)
svm_pod_mean_s, svm_pod_lcb_s, svm_a90_s, svm_a90_95_s, svm_params_s = \
    compute_pod(svm_synth_a, svm_synth_ahat, svm_threshold, x_range_s, block_size=avg_chunk)

print(f"  ✅ MK-MMD | Real a90={svm_a90_r:.1f}% | Synth a90={svm_a90_s:.1f}%")


# ================================================================
# [8] POD 비교 시각화
# ================================================================
print("\n" + "=" * 70)
print("  [8] POD 비교 시각화")
print("=" * 70)

COLORS = {'GAN': '#E74C3C', 'AE': '#3498DB', 'SVM': '#2ECC71'}

# ── 8.1 2-Panel: Real | Synth
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
for name, pm, pl, a90, color in [
    ('MMD-GAN', gan_pod_mean_r, gan_pod_lcb_r, gan_a90_r, COLORS['GAN']),
    ('MemAE',   ae_pod_mean_r,  ae_pod_lcb_r,  ae_a90_r,  COLORS['AE']),
    ('MK-MMD',  svm_pod_mean_r, svm_pod_lcb_r, svm_a90_r, COLORS['SVM']),
]:
    lbl = f'{name} ($a_{{90}}$={a90:.1f}%)' if not np.isnan(a90) else f'{name} (N/A)'
    ax.plot(x_range_r, pm, color=color, linewidth=3, label=lbl)
    ax.plot(x_range_r, pl, color=color, linewidth=1.5, linestyle='--', alpha=0.7)
ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
ax.set_title(f"Real Damage: POD ({len(SEEDS)}-seed avg scores)", fontweight='bold', fontsize=13)
ax.set_xlabel("Damage Case, $a$ (%)", fontsize=11)
ax.set_ylabel("Probability of Detection", fontsize=11)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 50])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=10)

ax = axes[1]
for name, pm, pl, a90, color in [
    ('MMD-GAN', gan_pod_mean_s, gan_pod_lcb_s, gan_a90_s, COLORS['GAN']),
    ('MemAE',   ae_pod_mean_s,  ae_pod_lcb_s,  ae_a90_s,  COLORS['AE']),
    ('MK-MMD',  svm_pod_mean_s, svm_pod_lcb_s, svm_a90_s, COLORS['SVM']),
]:
    lbl = f'{name} ($a_{{90}}$={a90:.1f}%)' if not np.isnan(a90) else f'{name} (N/A)'
    ax.plot(x_range_s, pm, color=color, linewidth=3, label=lbl)
    ax.plot(x_range_s, pl, color=color, linewidth=1.5, linestyle='--', alpha=0.7)
ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
ax.set_title(f"Synthetic Damage: POD ({len(SEEDS)}-seed avg scores)", fontweight='bold', fontsize=13)
ax.set_xlabel("Mapped Synthetic Damage, $a$ (%)", fontsize=11)
ax.set_ylabel("Probability of Detection", fontsize=11)
ax.set_ylim([0, 1.05]); ax.set_xlim([0, 100])
ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_COMP, "POD_Comparison_2Panel.png"), dpi=300)
plt.show()

# ── 8.2 4-Panel
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
panels = [
    (axes[0, 0], x_range_r,
     [(gan_pod_mean_r, 'MMD-GAN', COLORS['GAN']),
      (ae_pod_mean_r,  'MemAE',   COLORS['AE']),
      (svm_pod_mean_r, 'MK-MMD',  COLORS['SVM'])],
     "Real: Mean POD", "Damage Case, $a$ (%)", 50),
    (axes[0, 1], x_range_s,
     [(gan_pod_mean_s, 'MMD-GAN', COLORS['GAN']),
      (ae_pod_mean_s,  'MemAE',   COLORS['AE']),
      (svm_pod_mean_s, 'MK-MMD',  COLORS['SVM'])],
     "Synthetic: Mean POD", "Mapped Damage, $a$ (%)", 100),
    (axes[1, 0], x_range_r,
     [(gan_pod_lcb_r, 'MMD-GAN 95% LCB', COLORS['GAN']),
      (ae_pod_lcb_r,  'MemAE 95% LCB',   COLORS['AE']),
      (svm_pod_lcb_r, 'MK-MMD 95% LCB',  COLORS['SVM'])],
     "Real: 95% LCB", "Damage Case, $a$ (%)", 50),
    (axes[1, 1], x_range_s,
     [(gan_pod_lcb_s, 'MMD-GAN 95% LCB', COLORS['GAN']),
      (ae_pod_lcb_s,  'MemAE 95% LCB',   COLORS['AE']),
      (svm_pod_lcb_s, 'MK-MMD 95% LCB',  COLORS['SVM'])],
     "Synthetic: 95% LCB", "Mapped Damage, $a$ (%)", 100),
]
for ax, xr, curves, title, xlabel, xlim in panels:
    for curve, label, color in curves:
        ax.plot(xr, curve, color=color, linewidth=2.5, label=label)
    ax.axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel("Probability of Detection")
    ax.set_ylim([0, 1.05]); ax.set_xlim([0, xlim])
    ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_COMP, "POD_Comparison_4Panel.png"), dpi=300)
plt.show()

# ── 8.3 요약 테이블
def fmt(v): return f"{v:6.2f}%" if not np.isnan(v) else "    N/A"

print("\n" + "=" * 85)
print(f"        POD 최종 비교 ({len(SEEDS)}-seed score averaged, then single POD fit)")
print("=" * 85)
print(f"  {'Model':<10} {'Threshold Type':<18} {'Real a90':<12} {'Real a90/95':<14} "
      f"{'Synth a90':<12} {'Synth a90/95'}")
print("-" * 85)
print(f"  {'MMD-GAN':<10} {'mean+2σ (≈2% FAR)':<18} {fmt(gan_a90_r):<12} "
      f"{fmt(gan_a90_95_r):<14} {fmt(gan_a90_s):<12} {fmt(gan_a90_95_s)}")
print(f"  {'MemAE':<10} {'mean+2σ (≈2% FAR)':<18} {fmt(ae_a90_r):<12} "
      f"{fmt(ae_a90_95_r):<14} {fmt(ae_a90_s):<12} {fmt(ae_a90_95_s)}")
print(f"  {'MK-MMD':<10} {'95th-pct (5% FAR)':<18} {fmt(svm_a90_r):<12} "
      f"{fmt(svm_a90_95_r):<14} {fmt(svm_a90_s):<12} {fmt(svm_a90_95_s)}")
print("=" * 85)

# ── 8.4 결과 저장
comparison_results = {
    'methodology': 'Multi-seed score averaging then single POD fit',
    'n_seeds': len(SEEDS), 'seeds': SEEDS,
    'x_range_r': x_range_r, 'x_range_s': x_range_s,
    'GAN': {
        'pod_mean_r': gan_pod_mean_r, 'pod_lcb_r': gan_pod_lcb_r,
        'pod_mean_s': gan_pod_mean_s, 'pod_lcb_s': gan_pod_lcb_s,
        'a90_r': gan_a90_r, 'a90_95_r': gan_a90_95_r,
        'a90_s': gan_a90_s, 'a90_95_s': gan_a90_95_s,
        'threshold_real': gan_thr_real, 'threshold_synth': gan_thr_synth,
    },
    'AE': {
        'pod_mean_r': ae_pod_mean_r, 'pod_lcb_r': ae_pod_lcb_r,
        'pod_mean_s': ae_pod_mean_s, 'pod_lcb_s': ae_pod_lcb_s,
        'a90_r': ae_a90_r, 'a90_95_r': ae_a90_95_r,
        'a90_s': ae_a90_s, 'a90_95_s': ae_a90_95_s,
        'threshold': ae_log_thr,
    },
    'SVM': {
        'pod_mean_r': svm_pod_mean_r, 'pod_lcb_r': svm_pod_lcb_r,
        'pod_mean_s': svm_pod_mean_s, 'pod_lcb_s': svm_pod_lcb_s,
        'a90_r': svm_a90_r, 'a90_95_r': svm_a90_95_r,
        'a90_s': svm_a90_s, 'a90_95_s': svm_a90_95_s,
        'threshold': svm_threshold,
        'avg_chunk_size': avg_chunk,
    },
}

with open(os.path.join(SAVE_DIR_COMP, "comparison_results.pkl"), "wb") as f:
    pickle.dump(comparison_results, f)

print(f"\n✅ 결과 저장: {SAVE_DIR_COMP}")
print("   - POD_Comparison_2Panel.png")
print("   - POD_Comparison_4Panel.png")
print("   - comparison_results.pkl")
print("\n🎉 Multi-Seed Score Averaging POD Pipeline 완료!")