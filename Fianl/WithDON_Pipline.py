# ================================================================
#  5-RUN MLE POD PIPELINE (No Bootstrap)
#  MMD-GAN (Origin) | MemAE | MK-MMD (가속 OC-SVM)
#
#  방법론: 각 모델 5회 반복 학습 → 케이스별 탐지율(DR) → 5회 평균
#          → Hit/Miss MLE POD 피팅 (평균 곡선만, LCB 없음)
#  데이터: DeepONet 합성(Synthetic) + 실제(Real) 손상 데이터
# ================================================================

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.signal import hilbert
from scipy.stats import norm
from scipy.optimize import minimize
import optuna
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ================================================================
# [0] 설정
# ================================================================
DIR_B_RAW      = r"E:\2ndstructuredata\raw data"
FILE_B         = "healthyclean.txt"
SYNTH_DATA_DIR = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data"
SAVE_DIR       = r"E:\git\Master_Thesis\Fianl"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE        = 128
CHANNELS           = 8
DAMAGE_CASES_B     = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
SYNTHETIC_DI_STEPS = np.round(np.arange(0.00, 1.01, 0.02), 2)
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_RUNS = 5
SEED   = 42

x_range_r = np.linspace(0, 50, 200)
x_range_s = np.linspace(0, 100, 200)

print("=" * 70)
print(f"  5-RUN MLE POD PIPELINE (No Bootstrap)  |  Device: {device}")
print("=" * 70)


# ================================================================
# [1] Hit/Miss MLE POD (5-run DR 평균, LCB 없음)
# ================================================================
def fit_hit_miss_mle(a_vals, dr_vals):
    """이항 교차엔트로피 최소화로 정규-CDF (mu, sigma) 추정."""
    def neg_log_lik(p):
        mu, sigma = p
        prob = np.clip(norm.cdf(a_vals, loc=mu, scale=sigma), 1e-10, 1 - 1e-10)
        ll   = dr_vals * np.log(prob) + (1 - dr_vals) * np.log(1 - prob)
        return -np.sum(ll)
    init = [np.median(a_vals), np.std(a_vals) + 1e-5]
    res  = minimize(neg_log_lik, init, bounds=[(1e-3, None), (1e-3, None)], method='L-BFGS-B')
    return res.x, res.success

def get_mle_pod_curve(x_range, params):
    mu, sigma = params
    return norm.cdf(x_range, loc=mu, scale=sigma)

def pod_from_runs(dr_runs, a_keys, a_vals, x_range):
    a_vals = np.asarray(a_vals, dtype=float)
    avg_dr = np.array([np.mean(dr_runs[k]) for k in a_keys])
    params, ok = fit_hit_miss_mle(a_vals, avg_dr)
    mean_pod   = get_mle_pod_curve(x_range, params) if ok else np.zeros_like(x_range)
    a90 = x_range[np.argmax(mean_pod >= 0.9)] if np.any(mean_pod >= 0.9) else np.nan
    return mean_pod, a90, params, a_vals, avg_dr


# ================================================================
# [2] GAN / MemAE 공통 데이터 (MinMaxScaler → [-1,1])
# ================================================================
TRAIN_SIZE, VAL_SIZE, NUM_SAMPLES_GANAE = 1000, 500, 500
scaler_B, Train_H, Val_H = None, None, None

def load_health_data_ganae():
    global scaler_B, Train_H, Val_H
    path = os.path.join(DIR_B_RAW, FILE_B)
    raw  = [float(l.split()[1]) for l in open(path) if len(l.split()) >= 2]
    data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
    ns   = data.shape[1] // WINDOW_SIZE
    data = data[:, :ns * WINDOW_SIZE]
    reshaped  = data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
    scaler_B  = MinMaxScaler((0, 1)).fit(reshaped)
    norm_data = (scaler_B.transform(reshaped) * 2.0 - 1.0).reshape(-1, CHANNELS, WINDOW_SIZE)
    np.random.seed(42)
    idx     = np.random.permutation(ns)
    Train_H = norm_data[idx[:TRAIN_SIZE]]
    Val_H   = norm_data[idx[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]]
    print(f"✅ GAN/AE 건강 데이터: Train {Train_H.shape}, Val {Val_H.shape}")

def load_damage_ganae(path, is_synth=False):
    try:
        if is_synth:
            data = np.loadtxt(path, delimiter='\t').T.astype(np.float32)
        else:
            raw  = [float(l.split()[1]) for l in open(path) if len(l.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
        ns = data.shape[1] // WINDOW_SIZE
        if ns == 0:
            return None
        data = data[:, :ns * WINDOW_SIZE]
        reshaped  = data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
        norm_data = (scaler_B.transform(reshaped) * 2.0 - 1.0).reshape(-1, CHANNELS, WINDOW_SIZE)
        if ns > NUM_SAMPLES_GANAE:
            np.random.seed(42)
            norm_data = norm_data[np.random.choice(ns, NUM_SAMPLES_GANAE, replace=False)]
        return norm_data
    except Exception as e:
        print(f"   Error loading {path}: {e}")
        return None

load_health_data_ganae()


# ================================================================
# [3] MMD-GAN (Origin) — 5-run DR
# ================================================================
GAN_N1, GAN_N2  = 35, 19
GAN_KERNEL_SIZE = 6
GAN_ALPHA       = 0.3799
GAN_STRIDE      = 3
GAN_LR          = 0.001926
GAN_EPOCHS      = 112
GAN_BATCH       = 96
GAN_LATENT_DIM  = 100
GAN_FEAT_DIM    = 64

def rbf_kernel_torch(x, y, gamma=1.0):
    x = x.unsqueeze(1); y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss_calc(x, y, gamma=1.0):
    return (rbf_kernel_torch(x, x, gamma).mean()
            + rbf_kernel_torch(y, y, gamma).mean()
            - 2 * rbf_kernel_torch(x, y, gamma).mean())

class GAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc     = nn.Sequential(nn.Linear(GAN_LATENT_DIM, 32 * GAN_N1), nn.LeakyReLU(GAN_ALPHA))
        self.conv1  = nn.ConvTranspose1d(GAN_N1, GAN_N1, GAN_KERNEL_SIZE, 2, GAN_KERNEL_SIZE//2, output_padding=1)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2  = nn.ConvTranspose1d(GAN_N1, CHANNELS, GAN_KERNEL_SIZE, 2, GAN_KERNEL_SIZE//2, output_padding=1)
        self.tanh   = nn.Tanh()
    def forward(self, x):
        x = self.fc(x).view(-1, GAN_N1, 32)
        x = self.lrelu1(self.conv1(x))
        x = self.tanh(self.conv2(x))
        if x.size(2) < 128:
            x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

class GAN_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(CHANNELS, GAN_N2, GAN_KERNEL_SIZE, GAN_STRIDE, GAN_KERNEL_SIZE//2)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2  = nn.Conv1d(GAN_N2, GAN_N2, GAN_KERNEL_SIZE, GAN_STRIDE, GAN_KERNEL_SIZE//2)
        self.lrelu2 = nn.LeakyReLU(GAN_ALPHA)
        self.flatten = nn.Flatten()
        dummy = torch.zeros(1, CHANNELS, 128)
        out   = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.fc = nn.Linear(out.view(1, -1).size(1), GAN_FEAT_DIM)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(self.flatten(x))

def run_gan():
    dr_real  = {c: [] for c in DAMAGE_CASES_B}
    dr_synth = {di: [] for di in SYNTHETIC_DI_STEPS}

    for run in range(N_RUNS):
        print(f"  [GAN Run {run+1}/{N_RUNS}] 학습 중...")
        torch.manual_seed(SEED + run)
        gen, critic = GAN_Generator().to(device), GAN_Critic().to(device)
        opt_g = optim.Adam(gen.parameters(),    lr=GAN_LR)
        opt_c = optim.Adam(critic.parameters(), lr=GAN_LR)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)),
            batch_size=GAN_BATCH, shuffle=True, drop_last=True)

        for _ in tqdm(range(GAN_EPOCHS), desc="    epochs", leave=False):
            for (real,) in loader:
                real = real.to(device); b = real.size(0)
                opt_c.zero_grad()
                z = torch.randn(b, GAN_LATENT_DIM, device=device)
                fr, ff = critic(real), critic(gen(z).detach())
                (-mmd_loss_calc(fr, ff)).backward(); opt_c.step()
                opt_g.zero_grad()
                ffg = critic(gen(z))
                mmd_loss_calc(fr.detach(), ffg).backward(); opt_g.step()

        critic.eval()
        with torch.no_grad():
            feat = critic(torch.FloatTensor(Train_H).to(device)).cpu().numpy()
        mu_h    = feat.mean(0)
        cov_inv = np.linalg.inv(np.cov(feat, rowvar=False) + 1e-5 * np.eye(GAN_FEAT_DIM))

        def score(data):
            with torch.no_grad():
                f = critic(torch.FloatTensor(data).to(device)).cpu().numpy()
            d = f - mu_h
            return np.log(np.sqrt(np.maximum(np.sum(d @ cov_inv * d, axis=1), 1e-10)))

        v = score(Val_H)
        thr_real = v.mean() + 2 * v.std()
        d0 = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), is_synth=True)
        thr_synth = (lambda s: s.mean() + 2 * s.std())(score(d0)) if d0 is not None else thr_real

        for c in DAMAGE_CASES_B:
            d = load_damage_ganae(os.path.join(DIR_B_RAW, f"D3_{c}_1.txt"))
            if d is not None:
                dr_real[c].append(float(np.mean(score(d) > thr_real)))
        for di in SYNTHETIC_DI_STEPS:
            d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
            if d is not None:
                dr_synth[di].append(float(np.mean(score(d) > thr_synth)))
    return dr_real, dr_synth


# ================================================================
# [4] MemAE — 5-run DR
# ================================================================
AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_ALPHA = 19, 6, 3, 0.3799
AE_LATENT  = 64
AE_SLOTS, AE_SHRINK, AE_LAMBDA, AE_TEMP = 50, 0.02, 0.0002, 1.0
AE_LR, AE_EPOCHS, AE_BATCH = 0.001, 112, 96

class AE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_KERNEL//2)
        self.lrelu1 = nn.LeakyReLU(AE_ALPHA)
        self.conv2 = nn.Conv1d(AE_N_FEAT, AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_KERNEL//2)
        self.lrelu2 = nn.LeakyReLU(AE_ALPHA)
        dummy = torch.zeros(1, CHANNELS, 128)
        out   = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.conv_out_len = out.size(2)
        self.flat_size    = out.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_size, AE_LATENT)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(x.view(x.size(0), -1))

class AE_Decoder(nn.Module):
    def __init__(self, conv_out_len, flat_size):
        super().__init__()
        self.conv_out_len = conv_out_len
        self.fc = nn.Linear(AE_LATENT, flat_size)
        self.lrelu0 = nn.LeakyReLU(AE_ALPHA)
        self.deconv1 = nn.ConvTranspose1d(AE_N_FEAT, AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_KERNEL//2)
        self.lrelu1 = nn.LeakyReLU(AE_ALPHA)
        self.deconv2 = nn.ConvTranspose1d(AE_N_FEAT, CHANNELS, AE_KERNEL, AE_STRIDE, AE_KERNEL//2)
        self.tanh = nn.Tanh()
    def forward(self, z):
        x = self.lrelu0(self.fc(z)).view(-1, AE_N_FEAT, self.conv_out_len)
        x = self.lrelu1(self.deconv1(x))
        x = self.tanh(self.deconv2(x))
        if x.size(2) < 128:
            x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

class MemoryModule(nn.Module):
    def __init__(self, n_slots, dim, shrink, temp):
        super().__init__()
        self.shrink, self.temp = shrink, temp
        self.memory = nn.Parameter(torch.empty(n_slots, dim))
        nn.init.xavier_normal_(self.memory)
    def forward(self, z):
        sim = torch.matmul(F.normalize(z, p=2, dim=1),
                           F.normalize(self.memory, p=2, dim=1).t()) / self.temp
        att = F.softmax(sim, dim=1)
        if self.shrink > 0:
            att = (F.relu(att - self.shrink) * att) / (torch.abs(att - self.shrink) + 1e-12)
            att = F.normalize(att, p=1, dim=1)
        return torch.matmul(att, self.memory), att

class MemAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AE_Encoder()
        self.decoder = AE_Decoder(self.encoder.conv_out_len, self.encoder.flat_size)
        self.memory  = MemoryModule(AE_SLOTS, AE_LATENT, AE_SHRINK, AE_TEMP)
    def forward(self, x):
        z = self.encoder(x)
        z_mem, att = self.memory(z)
        return self.decoder(z_mem), att

def run_memae():
    dr_real  = {c: [] for c in DAMAGE_CASES_B}
    dr_synth = {di: [] for di in SYNTHETIC_DI_STEPS}

    for run in range(N_RUNS):
        print(f"  [MemAE Run {run+1}/{N_RUNS}] 학습 중...")
        torch.manual_seed(SEED + run)
        model = MemAE().to(device)
        opt   = optim.Adam(model.parameters(), lr=AE_LR)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)),
            batch_size=AE_BATCH, shuffle=True, drop_last=True)

        model.train()
        for _ in tqdm(range(AE_EPOCHS), desc="    epochs", leave=False):
            for (x,) in loader:
                x = x.to(device)
                opt.zero_grad()
                recon, att = model(x)
                loss = F.mse_loss(recon, x) + AE_LAMBDA * torch.mean(
                    -torch.sum(att * torch.log(att + 1e-12), dim=1))
                loss.backward(); opt.step()

        model.eval()
        def recon_score(data):
            with torch.no_grad():
                x = torch.FloatTensor(data).to(device)
                recon, _ = model(x)
                err = torch.mean((recon - x) ** 2, dim=[1, 2]).cpu().numpy()
            return np.log(np.maximum(err, 1e-10))

        v = recon_score(Val_H)
        thr_real = v.mean() + 2 * v.std()
        d0 = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), is_synth=True)
        thr_synth = (lambda s: s.mean() + 2 * s.std())(recon_score(d0)) if d0 is not None else thr_real

        for c in DAMAGE_CASES_B:
            d = load_damage_ganae(os.path.join(DIR_B_RAW, f"D3_{c}_1.txt"))
            if d is not None:
                dr_real[c].append(float(np.mean(recon_score(d) > thr_real)))
        for di in SYNTHETIC_DI_STEPS:
            d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
            if d is not None:
                dr_synth[di].append(float(np.mean(recon_score(d) > thr_synth)))
    return dr_real, dr_synth


# ================================================================
# [5] MK-MMD (가속 OC-SVM) — 5-run DR
# ================================================================
SVM_NUM_SAMPLES  = 1000
SVM_PFA          = 0.05
SVM_OPTUNA_TRIAL = 40
MAX_REF          = 400

def _windowize_svm(data):
    ns = data.shape[1] // WINDOW_SIZE
    if ns == 0:
        return None, 0
    data = data[:, :ns * WINDOW_SIZE]
    return data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2), ns

def load_data_svm(path, is_synth=False):
    try:
        if is_synth:
            data = np.loadtxt(path, delimiter='\t').T.astype(np.float32)
        else:
            raw  = [float(l.split()[1]) for l in open(path) if len(l.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
        windows, ns = _windowize_svm(data)
        if windows is None:
            return None
        if ns > SVM_NUM_SAMPLES and "healthyclean" not in path.lower():
            np.random.seed(42)
            windows = windows[np.random.choice(ns, SVM_NUM_SAMPLES, replace=False)]
        return windows
    except Exception as e:
        print(f"   Error: {e}")
        return None

def extract_features_svm(data):
    if len(data) == 0:
        return np.array([])
    mean_v = np.mean(data, 2); std_v = np.std(data, 2); rms_v = np.sqrt(np.mean(data**2, 2))
    peak_v = np.max(np.abs(data), 2); p2p_v = np.ptp(data, 2); crest_v = peak_v / (rms_v + 1e-10)
    cen = data - mean_v[:, :, None]
    skew_v = np.mean(cen**3, 2) / (std_v**3 + 1e-10); kurt_v = np.mean(cen**4, 2) / (std_v**4 + 1e-10)
    zcr_v  = np.mean(np.abs(np.diff(np.sign(data), axis=2)), 2) / 2
    fft_mag = np.abs(np.fft.rfft(data, axis=-1)); freqs = np.fft.rfftfreq(WINDOW_SIZE)
    fft_sum = np.sum(fft_mag, 2) + 1e-10; centroid = np.sum(freqs * fft_mag, 2) / fft_sum
    spread  = np.sqrt(np.sum(((freqs - centroid[..., None])**2) * fft_mag, 2) / fft_sum)
    p_norm  = fft_mag / fft_sum[..., None]; sp_ent = -np.sum(p_norm * np.log2(p_norm + 1e-10), 2)
    hf_ratio = np.sum(fft_mag[:, :, fft_mag.shape[-1]//2:], 2) / fft_sum
    edges = np.linspace(0, fft_mag.shape[2], 5, dtype=int)
    bands = [np.sum(fft_mag[:, :, edges[i]:edges[i+1]], 2) / fft_sum for i in range(4)]
    env = np.abs(hilbert(data, axis=-1))
    env_rms = np.sqrt(np.mean(env**2, 2)); env_mean = np.mean(env, 2); env_std = np.std(env, 2)
    env_kurt = np.mean((env - env_mean[..., None])**4, 2) / (env_std**4 + 1e-10)
    T  = data.shape[2]
    zc = (data - data.mean(2, keepdims=True)) / (data.std(2, keepdims=True) + 1e-10)
    corr = np.einsum('nct,ndt->ncd', zc, zc) / T
    tri  = np.triu_indices(CHANNELS, k=1)
    corr_feats = corr[:, tri[0], tri[1]]
    return np.concatenate([mean_v, std_v, rms_v, peak_v, p2p_v, crest_v, skew_v, kurt_v, zcr_v,
                           centroid, spread, sp_ent, hf_ratio, np.concatenate(bands, 1),
                           env_rms, env_mean, env_std, env_kurt, corr_feats], axis=1)

def precompute_ref(ref, base_gamma, seed=SEED):
    if len(ref) > MAX_REF:
        ref = ref[np.random.default_rng(seed).choice(len(ref), MAX_REF, replace=False)]
    gammas  = [base_gamma * 0.1, base_gamma, base_gamma * 10]
    ref_sqd = euclidean_distances(ref, ref, squared=True)
    return ref, gammas, [float(np.exp(-g * ref_sqd).mean()) for g in gammas]

def mmd_scores_fast(features, ref_cache, chunk, stride):
    ref, gammas, xx = ref_cache
    n = len(features)
    if n < chunk:
        return np.array([])
    out = []
    for i in range(0, n - chunk + 1, stride):
        c = features[i:i + chunk]
        d_yy = euclidean_distances(c, c, squared=True)
        d_xy = euclidean_distances(ref, c, squared=True)
        out.append(sum(x + np.exp(-g * d_yy).mean() - 2 * np.exp(-g * d_xy).mean()
                       for g, x in zip(gammas, xx)))
    return np.array(out)

def run_svm():
    dr_real  = {c: [] for c in DAMAGE_CASES_B}
    dr_synth = {di: [] for di in SYNTHETIC_DI_STEPS}

    all_h = load_data_svm(os.path.join(DIR_B_RAW, FILE_B))

    print("  [MK-MMD] 손상 파일 특징 캐싱...")
    feat_cache = {}
    for c in DAMAGE_CASES_B:
        d = load_data_svm(os.path.join(DIR_B_RAW, f"D3_{c}_1.txt"))
        if d is not None:
            feat_cache[('r', c)] = extract_features_svm(d)
    for di in SYNTHETIC_DI_STEPS:
        d = load_data_svm(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
        if d is not None:
            feat_cache[('s', di)] = extract_features_svm(d)

    print("  [MK-MMD] Optuna 최적화 (1회)...")
    np.random.seed(42); sh = all_h.copy(); np.random.shuffle(sh)
    si = int(len(sh) * 0.7); tr0, vl0 = sh[:si], sh[si:]
    tr0_f, vl0_f = extract_features_svm(tr0), extract_features_svm(vl0)
    tgt = load_data_svm(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.50.txt"), is_synth=True)
    if tgt is None:
        tgt = load_data_svm(os.path.join(DIR_B_RAW, "D3_48_1.txt"))
    tgt_f = extract_features_svm(tgt)
    sc0 = RobustScaler().fit(tr0_f)
    trs, vls, tgs = sc0.transform(tr0_f), sc0.transform(vl0_f), sc0.transform(tgt_f)

    def objective(trial):
        pc = trial.suggest_int('pca_comp', 15, 60)
        ck = trial.suggest_int('chunk_size', 20, 80, step=10)
        gm = trial.suggest_float('gamma_mult', 0.01, 10.0, log=True)
        pca = PCA(pc, random_state=42)
        tp, vp, gp = pca.fit_transform(trs), pca.transform(vls), pca.transform(tgs)
        bg = (1.0 / (np.median(pdist(tp, 'sqeuclidean')) + 1e-10)) * gm
        rc = precompute_ref(tp, bg)
        vS = mmd_scores_fast(vp, rc, ck, max(1, ck // 2))
        tS = mmd_scores_fast(gp, rc, ck, max(1, ck // 2))
        if len(vS) < 5 or len(tS) < 5:
            return -999.0
        return (tS.mean() - vS.mean()) / (vS.std() + 1e-6)

    study = optuna.create_study(direction='maximize')
    with tqdm(total=SVM_OPTUNA_TRIAL, desc="    Optuna") as pbar:
        study.optimize(objective, n_trials=SVM_OPTUNA_TRIAL, callbacks=[lambda s, t: pbar.update(1)])
    best = study.best_params
    print(f"    best: PCA={best['pca_comp']}, Chunk={best['chunk_size']}, Gamma×={best['gamma_mult']:.4f}")
    final_chunk = best['chunk_size']
    eval_stride = max(1, final_chunk // 6)

    for run in range(N_RUNS):
        print(f"  [MK-MMD Run {run+1}/{N_RUNS}] 평가 중...")
        rng = np.random.default_rng(SEED + run)
        sh  = all_h.copy(); rng.shuffle(sh)
        si  = int(len(sh) * 0.7); tr, vl = sh[:si], sh[si:]
        sc  = RobustScaler().fit(extract_features_svm(tr))
        pca = PCA(best['pca_comp'], random_state=SEED + run)
        tr_p = pca.fit_transform(sc.transform(extract_features_svm(tr)))
        vl_p = pca.transform(sc.transform(extract_features_svm(vl)))
        bg   = 1.0 / (np.median(pdist(tr_p, 'sqeuclidean')) + 1e-10) * best['gamma_mult']
        rc   = precompute_ref(tr_p, bg, seed=SEED + run)

        v_sc = mmd_scores_fast(vl_p, rc, final_chunk, stride=2)
        thr  = np.percentile(v_sc, (1 - SVM_PFA) * 100)

        def file_dr(key):
            fp = pca.transform(sc.transform(feat_cache[key]))
            s  = mmd_scores_fast(fp, rc, final_chunk, eval_stride)
            return float(np.mean(s > thr)) if len(s) else np.nan

        for c in DAMAGE_CASES_B:
            if ('r', c) in feat_cache:
                dr_real[c].append(file_dr(('r', c)))
        for di in SYNTHETIC_DI_STEPS:
            if ('s', di) in feat_cache:
                dr_synth[di].append(file_dr(('s', di)))
    return dr_real, dr_synth


# ================================================================
# [6] 실행 + MLE POD 계산
# ================================================================
print("\n[GAN]");    gan_dr_r, gan_dr_s = run_gan()
print("\n[MemAE]");  ae_dr_r,  ae_dr_s  = run_memae()
print("\n[MK-MMD]"); svm_dr_r, svm_dr_s = run_svm()

def safe_pod(dr_r, dr_s):
    kr = [k for k in DAMAGE_CASES_B if len(dr_r[k]) > 0]
    ks = [k for k in SYNTHETIC_DI_STEPS if len(dr_s[k]) > 0]
    pr = pod_from_runs(dr_r, kr, np.array(kr, float),          x_range_r)
    ps = pod_from_runs(dr_s, ks, np.array(ks, float) * 100.0,  x_range_s)
    return pr, ps

print("\n[MLE POD 계산]")
gan_pr, gan_ps = safe_pod(gan_dr_r, gan_dr_s)
ae_pr,  ae_ps  = safe_pod(ae_dr_r,  ae_dr_s)
svm_pr, svm_ps = safe_pod(svm_dr_r, svm_dr_s)

print(f"  GAN    | Real a90={gan_pr[1]:.1f}% | Synth a90={gan_ps[1]:.1f}%")
print(f"  MemAE  | Real a90={ae_pr[1]:.1f}%  | Synth a90={ae_ps[1]:.1f}%")
print(f"  MK-MMD | Real a90={svm_pr[1]:.1f}% | Synth a90={svm_ps[1]:.1f}%")


# ================================================================
# [7] 비교 시각화 (Real | Synth) — 평균 POD + DR 산점
# ================================================================
COLORS = {'MMD-GAN': '#E74C3C', 'MemAE': '#3498DB', 'MK-MMD': '#2ECC71'}

def draw_panel(ax, xr, models, title, xlabel, xlim):
    for name, (pm, a90, params, a_vals, avg_dr), color in models:
        lbl = f'{name} ($a_{{90}}$={a90:.1f}%)' if not np.isnan(a90) else f'{name} (N/A)'
        ax.plot(xr, pm, color=color, lw=3, label=lbl)
        ax.scatter(a_vals, avg_dr, color=color, s=22, alpha=0.5, edgecolors='none')
    ax.axhline(0.9, color='k', ls=':', lw=1.5, label='90% Target')
    ax.set_title(title, fontweight='bold'); ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability of Detection')
    ax.set_ylim([0, 1.05]); ax.set_xlim([0, xlim]); ax.grid(True, ls=':', alpha=0.7)
    ax.legend(loc='lower right', fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
draw_panel(axes[0], x_range_r,
           [('MMD-GAN', gan_pr, COLORS['MMD-GAN']),
            ('MemAE',   ae_pr,  COLORS['MemAE']),
            ('MK-MMD',  svm_pr, COLORS['MK-MMD'])],
           "Real Damage Data", "Damage Case, $a$ (%)", 50)
draw_panel(axes[1], x_range_s,
           [('MMD-GAN', gan_ps, COLORS['MMD-GAN']),
            ('MemAE',   ae_ps,  COLORS['MemAE']),
            ('MK-MMD',  svm_ps, COLORS['MK-MMD'])],
           "Synthetic Damage Data", "Mapped Damage, $a$ (%)", 100)
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, "MLE_POD_Comparison.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("\n" + "=" * 70)
print("  MLE POD 요약")
print("=" * 70)
print(f"  {'Model':<10}{'Real a90':>12}{'Synth a90':>12}")
for name, pr, ps in [('MMD-GAN', gan_pr, gan_ps), ('MemAE', ae_pr, ae_ps), ('MK-MMD', svm_pr, svm_ps)]:
    print(f"  {name:<10}{pr[1]:>11.1f}%{ps[1]:>11.1f}%")


# ================================================================
# [8] 결과 저장
# ================================================================
results = {
    'n_runs': N_RUNS,
    'dr_runs': {'gan': (gan_dr_r, gan_dr_s), 'ae': (ae_dr_r, ae_dr_s), 'svm': (svm_dr_r, svm_dr_s)},
    'pod': {'x_range_r': x_range_r, 'x_range_s': x_range_s,
            'gan': (gan_pr, gan_ps), 'ae': (ae_pr, ae_ps), 'svm': (svm_pr, svm_ps)},
}
with open(os.path.join(SAVE_DIR, "MLE_POD_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print("\n✅ 완료!")
print(f"   - 그래프: {fig_path}")
print(f"   - 결과:   {os.path.join(SAVE_DIR, 'MLE_POD_results.pkl')}")