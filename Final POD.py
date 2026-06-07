# -*- coding: utf-8 -*-
"""
MASTER UNIFIED 10-RUN ENSEMBLE MLE POD PIPELINE (무중단 자동화 버전 - 메모리 누수 수정판)
- 목표: Case A1 ~ A8까지 DON 및 NO_DON 모드를 모두 자동 순회
- 업데이트 1: 데이터 누락이나 수학적 에러 발생 시 멈추지 않는 Try-Except 방어막
- 업데이트 2: [핵심 수정] GPU 메모리 누수 방지
    * 메모리 정리(empty_cache/gc.collect)를 finally 블록으로 이동 → 성공/실패 무관하게 항상 정리
    * run_gan / run_memae 에서 매 run 종료 시 모델·옵티마이저를 del 후 empty_cache
    * 모델 실행기 호출 사이에도 캐시 정리
- 업데이트 3: [핵심 수정] 모든 print 가 즉시 출력되도록 flush=True 강제 적용
"""

import functools
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # VS Code 블로킹 방지
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.signal import hilbert
from scipy.stats import norm
from scipy.optimize import minimize
import optuna
import gc
import traceback

# 💡 [수정 3] 모든 print 를 flush=True 로 강제 → 버퍼링으로 인한 "멈춘 것처럼 보임" 방지
#    이렇게 하면 아래 모든 print(...) 호출이 자동으로 즉시 출력됩니다.
print = functools.partial(print, flush=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================================================
# 전역 하이퍼파라미터 및 상수 설정
# ================================================================
WINDOW_SIZE    = 128
CHANNELS       = 8
DAMAGE_CASES_B = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_RUNS = 10  
np.random.seed(999)
SEEDS = np.random.choice(range(100, 10000), size=N_RUNS, replace=False).tolist()

x_range_r = np.linspace(0, 50, 200)
x_range_s = np.linspace(0, 100, 200)
COLORS = {'MMD-GAN': '#E74C3C', 'MemAE': '#3498DB', 'MK-MMD': '#2ECC71'}

# 모델 설정값
GAN_N1, GAN_N2, GAN_KERNEL_SIZE, GAN_ALPHA, GAN_STRIDE = 35, 19, 6, 0.3799, 3
GAN_LR, GAN_EPOCHS, GAN_BATCH, GAN_LATENT_DIM, GAN_FEAT_DIM = 0.001926, 112, 96, 100, 64
AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_ALPHA, AE_LATENT, AE_SLOTS, AE_SHRINK, AE_LAMBDA, AE_TEMP, AE_LR, AE_EPOCHS, AE_BATCH = 19, 6, 3, 0.3799, 64, 50, 0.02, 0.0002, 1.0, 0.001, 112, 96
SVM_NUM_SAMPLES, SVM_PFA, SVM_OPTUNA_TRIAL, MAX_REF = 1000, 0.05, 40, 400


# ================================================================
# 💡 [수정 2] GPU 메모리 정리 헬퍼
# ================================================================
def free_gpu():
    """GPU 캐시와 파이썬 가비지를 정리한다. 어디서 호출해도 안전하다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ================================================================
# 모델 클래스 선언부 (축약 없이 그대로 유지)
# ================================================================
def rbf_kernel_torch(x, y, gamma=1.0):
    x = x.unsqueeze(1); y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss_calc(x, y, gamma=1.0):
    return (rbf_kernel_torch(x, x, gamma).mean() + rbf_kernel_torch(y, y, gamma).mean() - 2 * rbf_kernel_torch(x, y, gamma).mean())

class GAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc     = nn.Sequential(nn.Linear(GAN_LATENT_DIM, 32 * GAN_N1), nn.LeakyReLU(GAN_ALPHA))
        self.conv1  = nn.ConvTranspose1d(GAN_N1, GAN_N1, GAN_KERNEL_SIZE, 2, GAN_KERNEL_SIZE // 2, output_padding=1)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2  = nn.ConvTranspose1d(GAN_N1, CHANNELS, GAN_KERNEL_SIZE, 2, GAN_KERNEL_SIZE // 2, output_padding=1)
        self.tanh   = nn.Tanh()
    def forward(self, x):
        x = self.fc(x).view(-1, GAN_N1, 32)
        x = self.lrelu1(self.conv1(x))
        x = self.tanh(self.conv2(x))
        if x.size(2) < 128: x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

class GAN_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(CHANNELS, GAN_N2, GAN_KERNEL_SIZE, GAN_STRIDE, GAN_KERNEL_SIZE // 2)
        self.lrelu1 = nn.LeakyReLU(GAN_ALPHA)
        self.conv2  = nn.Conv1d(GAN_N2, GAN_N2, GAN_KERNEL_SIZE, GAN_STRIDE, GAN_KERNEL_SIZE // 2)
        self.lrelu2 = nn.LeakyReLU(GAN_ALPHA)
        self.flatten = nn.Flatten()
        dummy = torch.zeros(1, CHANNELS, 128)
        out   = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.fc = nn.Linear(out.view(1, -1).size(1), GAN_FEAT_DIM)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(self.flatten(x))

class AE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_KERNEL // 2)
        self.lrelu1 = nn.LeakyReLU(AE_ALPHA)
        self.conv2 = nn.Conv1d(AE_N_FEAT, AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_KERNEL // 2)
        self.lrelu2 = nn.LeakyReLU(AE_ALPHA)
        dummy = torch.zeros(1, CHANNELS, 128)
        out   = self.conv2(self.lrelu1(self.conv1(dummy)))
        self.conv_out_len, self.flat_size = out.size(2), out.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_size, AE_LATENT)
    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return self.fc(x.view(x.size(0), -1))

class AE_Decoder(nn.Module):
    def __init__(self, conv_out_len, flat_size):
        super().__init__()
        self.conv_out_len, self.fc = conv_out_len, nn.Linear(AE_LATENT, flat_size)
        self.lrelu0 = nn.LeakyReLU(AE_ALPHA)
        self.deconv1 = nn.ConvTranspose1d(AE_N_FEAT, AE_N_FEAT, AE_KERNEL, AE_STRIDE, AE_KERNEL // 2)
        self.lrelu1 = nn.LeakyReLU(AE_ALPHA)
        self.deconv2 = nn.ConvTranspose1d(AE_N_FEAT, CHANNELS, AE_KERNEL, AE_STRIDE, AE_KERNEL // 2)
        self.tanh = nn.Tanh()
    def forward(self, z):
        x = self.lrelu0(self.fc(z)).view(-1, AE_N_FEAT, self.conv_out_len)
        x = self.lrelu1(self.deconv1(x))
        x = self.tanh(self.deconv2(x))
        if x.size(2) < 128: x = F.pad(x, (0, 128 - x.size(2)))
        return x[:, :, :128]

class MemoryModule(nn.Module):
    def __init__(self, n_slots, dim, shrink, temp):
        super().__init__()
        self.shrink, self.temp = shrink, temp
        self.memory = nn.Parameter(torch.empty(n_slots, dim))
        nn.init.xavier_normal_(self.memory)
    def forward(self, z):
        sim = torch.matmul(F.normalize(z, p=2, dim=1), F.normalize(self.memory, p=2, dim=1).t()) / self.temp
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

def extract_features_svm(data):
    if len(data) == 0: return np.array([])
    mean_v = np.mean(data, 2)
    std_v = np.std(data, 2)
    rms_v = np.sqrt(np.mean(data ** 2, 2))
    peak_v = np.max(np.abs(data), 2)
    p2p_v = np.ptp(data, 2)
    crest_v = peak_v / (rms_v + 1e-10)
    cen = data - mean_v[:, :, None]
    skew_v = np.mean(cen ** 3, 2) / (std_v ** 3 + 1e-10)
    kurt_v = np.mean(cen ** 4, 2) / (std_v ** 4 + 1e-10)
    zcr_v  = np.mean(np.abs(np.diff(np.sign(data), axis=2)), 2) / 2
    fft_mag = np.abs(np.fft.rfft(data, axis=-1))
    freqs = np.fft.rfftfreq(WINDOW_SIZE)
    fft_sum = np.sum(fft_mag, axis=2) + 1e-10
    centroid = np.sum(freqs[None, None, :] * fft_mag, axis=2) / fft_sum
    spread  = np.sqrt(np.sum(((freqs - centroid[..., None]) ** 2) * fft_mag, 2) / fft_sum)
    p_norm  = fft_mag / fft_sum[..., None]
    sp_ent = -np.sum(p_norm * np.log2(p_norm + 1e-10), 2)
    hf_ratio = np.sum(fft_mag[:, :, fft_mag.shape[-1] // 2:], 2) / fft_sum
    edges = np.linspace(0, fft_mag.shape[2], 5, dtype=int)
    bands = [np.sum(fft_mag[:, :, edges[i]:edges[i + 1]], 2) / fft_sum for i in range(4)]
    env = np.abs(hilbert(data, axis=-1))
    env_rms = np.sqrt(np.mean(env ** 2, 2))
    env_mean = np.mean(env, 2)
    env_std = np.std(env, 2)
    env_kurt = np.mean((env - env_mean[..., None]) ** 4, 2) / (env_std ** 4 + 1e-10)
    T  = data.shape[2]
    zc = (data - data.mean(2, keepdims=True)) / (data.std(2, keepdims=True) + 1e-10)
    corr_feats = (np.einsum('nct,ndt->ncd', zc, zc) / T)[:, np.triu_indices(CHANNELS, k=1)[0], np.triu_indices(CHANNELS, k=1)[1]]
    return np.concatenate([mean_v, std_v, rms_v, peak_v, p2p_v, crest_v, skew_v, kurt_v, zcr_v,
                           centroid, spread, sp_ent, hf_ratio, np.concatenate(bands, 1),
                           env_rms, env_mean, env_std, env_kurt, corr_feats], axis=1)

def precompute_ref(ref, base_gamma):
    if len(ref) > MAX_REF: ref = ref[np.random.choice(len(ref), MAX_REF, replace=False)]
    gammas  = [base_gamma * 0.1, base_gamma, base_gamma * 10]
    ref_sqd = euclidean_distances(ref, ref, squared=True)
    return ref, gammas, [float(np.exp(-g * ref_sqd).mean()) for g in gammas]

def mmd_scores_fast(features, ref_cache, chunk, stride):
    ref, gammas, xx = ref_cache
    n = len(features)
    if n < chunk: return np.array([])
    out = []
    for i in range(0, n - chunk + 1, stride):
        c = features[i:i + chunk]
        out.append(sum(x + np.exp(-g * euclidean_distances(c, c, squared=True)).mean()
                       - 2 * np.exp(-g * euclidean_distances(ref, c, squared=True)).mean()
                       for g, x in zip(gammas, xx)))
    return np.array(out)

# 💡 [핵심 방어막] 빈 배열이 들어오면 에러를 뱉는 대신 빈 값을 반환하여 튕김 방지
def fit_hit_miss_mle(a_vals, dr_vals):
    if len(a_vals) == 0 or len(dr_vals) == 0:
        return [0, 1], False
        
    def neg_log_lik(p):
        mu, sigma = p
        prob = np.clip(norm.cdf(a_vals, loc=mu, scale=sigma), 1e-10, 1 - 1e-10)
        ll   = dr_vals * np.log(prob) + (1 - dr_vals) * np.log(1 - prob)
        return -np.sum(ll)
    init = [np.median(a_vals), np.std(a_vals) + 1e-5]
    res  = minimize(neg_log_lik, init, bounds=[(1e-3, None), (1e-3, None)], method='L-BFGS-B')
    return res.x, res.success

def pod_from_runs(dr_runs, a_keys, a_vals, x_range):
    a_vals = np.asarray(a_vals, dtype=float)
    if len(a_vals) == 0: # 텅 빈 배열 방어선
        return np.zeros_like(x_range), np.zeros_like(x_range), np.nan, [0, 1], [], []
        
    avg_dr = np.array([np.mean(dr_runs[k]) for k in a_keys])
    if len(avg_dr) == 0:
        return np.zeros_like(x_range), np.zeros_like(x_range), np.nan, [0, 1], [], []

    params, ok = fit_hit_miss_mle(a_vals, avg_dr)
    mean_pod   = norm.cdf(x_range, loc=params[0], scale=params[1]) if ok else np.zeros_like(x_range)
    n = len(a_keys)
    se = params[1] / np.sqrt(n)
    z_lcb = (x_range - params[0]) / (params[1] + norm.ppf(0.95) * se)
    pod_lcb = norm.cdf(z_lcb) if ok else np.zeros_like(x_range)
    a90 = x_range[np.argmax(mean_pod >= 0.9)] if np.any(mean_pod >= 0.9) else np.nan
    return mean_pod, pod_lcb, a90, params, a_vals, avg_dr


# ================================================================
# 핵심 파이프라인 런처 (각 Case / Mode 별 고립 실행)
# ================================================================
def execute_master_pipeline(case_name, data_mode):
    print("\n" + "=" * 70)
    print(f" 🚀 10-RUN MASTER POD PIPELINE | CASE: {case_name} | MODE: {data_mode} ")
    print("=" * 70)

    # 💡 [수정 2] 함수 본문 전체를 try 로 감싸고, finally 에서 무조건 GPU 메모리 정리.
    #    → 중간에 에러가 나든 정상 종료하든 GPU 캐시가 반드시 회수된다.
    try:
        # 1. 경로 세팅 동적 할당
        DIR_B_RAW = r"E:\2ndstructuredata\raw data"
        FILE_B    = "healthyclean.txt"
        SAVE_DIR  = rf"E:\git\benchmarktu1402-master\benchmarktu1402-master\{case_name}\Unified_10Run_Comparison_{data_mode}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        if data_mode == 'NO_DON':
            SYNTH_DATA_DIR = rf"E:\git\benchmarktu1402-master\benchmarktu1402-master\{case_name}\Synthetic_B_Data_Manual"
            SYNTHETIC_DI_STEPS = np.round(np.arange(0.00, 1.01, 0.1), 2)
        else:
            SYNTH_DATA_DIR = rf"E:\git\benchmarktu1402-master\benchmarktu1402-master\{case_name}\Synthetic_B_Data"
            SYNTHETIC_DI_STEPS = np.round(np.arange(0.00, 1.01, 0.02), 2)

        # 💡 폴더 유무 및 내부 파일 존재 여부 방어막
        if not os.path.exists(SYNTH_DATA_DIR) or len(os.listdir(SYNTH_DATA_DIR)) == 0:
            print(f"  -> ⏭️ [스킵] {SYNTH_DATA_DIR} 폴더가 없거나 비어있어 해당 모드를 건너뜁니다.")
            return

        # 2. 데이터 로더
        TRAIN_SIZE, VAL_SIZE, NUM_SAMPLES_GANAE = 1000, 500, 500
        
        path_h = os.path.join(DIR_B_RAW, FILE_B)
        raw_h  = [float(l.split()[1]) for l in open(path_h) if len(l.split()) >= 2]
        data_h = np.array(raw_h, dtype=np.float32).reshape(CHANNELS, -1)
        ns_h   = data_h.shape[1] // WINDOW_SIZE
        data_h = data_h[:, :ns_h * WINDOW_SIZE]
        reshaped_h  = data_h.reshape(CHANNELS, ns_h, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns_h, -1)
        
        scaler_B  = MinMaxScaler((0, 1)).fit(reshaped_h)
        norm_data_h = (scaler_B.transform(reshaped_h) * 2.0 - 1.0).reshape(-1, CHANNELS, WINDOW_SIZE)
        
        rng_init = np.random.RandomState(42)
        idx = rng_init.permutation(ns_h)
        Train_H = norm_data_h[idx[:TRAIN_SIZE]]
        Val_H   = norm_data_h[idx[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]]

        def load_damage_ganae(path, is_synth=False):
            try:
                if is_synth:
                    data = np.loadtxt(path, delimiter='\t').T.astype(np.float32)
                else:
                    raw  = [float(l.split()[1]) for l in open(path) if len(l.split()) >= 2]
                    data = np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
                ns = data.shape[1] // WINDOW_SIZE
                if ns == 0: return None
                data = data[:, :ns * WINDOW_SIZE]
                reshaped  = data.reshape(CHANNELS, ns, WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
                norm_data = (scaler_B.transform(reshaped) * 2.0 - 1.0).reshape(-1, CHANNELS, WINDOW_SIZE)
                if ns > NUM_SAMPLES_GANAE:
                    r = np.random.RandomState(42)
                    norm_data = norm_data[r.choice(ns, NUM_SAMPLES_GANAE, replace=False)]
                return norm_data
            except Exception:
                return None

        # 3. 모델 실행기 모듈
        def run_gan():
            dr_real  = {c: [] for c in DAMAGE_CASES_B}
            dr_synth = {di: [] for di in SYNTHETIC_DI_STEPS}
            for run in range(N_RUNS):
                print(f"  [GAN Run {run+1}/{N_RUNS}] Seed {SEEDS[run]} 학습 중...")
                torch.manual_seed(SEEDS[run])
                gen, critic = GAN_Generator().to(device), GAN_Critic().to(device)
                opt_g = optim.Adam(gen.parameters(), lr=GAN_LR)
                opt_c = optim.Adam(critic.parameters(), lr=GAN_LR)
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)), batch_size=GAN_BATCH, shuffle=True, drop_last=True)
                for _ in range(GAN_EPOCHS):
                    for (real,) in loader:
                        real = real.to(device)
                        b = real.size(0)
                        opt_c.zero_grad()
                        z = torch.randn(b, GAN_LATENT_DIM, device=device)
                        fr, ff = critic(real), critic(gen(z).detach())
                        (-mmd_loss_calc(fr, ff)).backward()
                        opt_c.step()
                        opt_g.zero_grad()
                        ffg = critic(gen(z))
                        mmd_loss_calc(fr.detach(), ffg).backward()
                        opt_g.step()

                critic.eval()
                with torch.no_grad(): feat = critic(torch.FloatTensor(Train_H).to(device)).cpu().numpy()
                mu_h    = feat.mean(0)
                cov_inv = np.linalg.inv(np.cov(feat, rowvar=False) + 1e-5 * np.eye(GAN_FEAT_DIM))
                def score(data):
                    with torch.no_grad(): f = critic(torch.FloatTensor(data).to(device)).cpu().numpy()
                    d = f - mu_h
                    return np.log(np.sqrt(np.maximum(np.sum(d @ cov_inv * d, axis=1), 1e-10)))

                v = score(Val_H)
                thr_real = v.mean() + 2 * v.std()
                d0 = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), is_synth=True)
                thr_synth = (lambda s: s.mean() + 2 * s.std())(score(d0)) if d0 is not None else thr_real

                for c in DAMAGE_CASES_B:
                    d = load_damage_ganae(os.path.join(DIR_B_RAW, f"D3_{c}_1.txt"))
                    if d is not None: dr_real[c].append(float(np.mean(score(d) > thr_real)))
                for di in SYNTHETIC_DI_STEPS:
                    d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
                    if d is not None: dr_synth[di].append(float(np.mean(score(d) > thr_synth)))

                # 💡 [수정 2] 이번 run 의 모델·옵티마이저·로더를 명시적으로 해제 후 GPU 캐시 비우기
                del gen, critic, opt_g, opt_c, loader
                free_gpu()
            return dr_real, dr_synth

        def run_memae():
            dr_real  = {c: [] for c in DAMAGE_CASES_B}
            dr_synth = {di: [] for di in SYNTHETIC_DI_STEPS}
            for run in range(N_RUNS):
                print(f"  [MemAE Run {run+1}/{N_RUNS}] Seed {SEEDS[run]} 학습 중...")
                torch.manual_seed(SEEDS[run])
                model = MemAE().to(device)
                opt   = optim.Adam(model.parameters(), lr=AE_LR)
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.FloatTensor(Train_H)), batch_size=AE_BATCH, shuffle=True, drop_last=True)
                model.train()
                for _ in range(AE_EPOCHS):
                    for (x,) in loader:
                        x = x.to(device)
                        opt.zero_grad()
                        recon, att = model(x)
                        loss = F.mse_loss(recon, x) + AE_LAMBDA * torch.mean(-torch.sum(att * torch.log(att + 1e-12), dim=1))
                        loss.backward()
                        opt.step()
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
                    if d is not None: dr_real[c].append(float(np.mean(recon_score(d) > thr_real)))
                for di in SYNTHETIC_DI_STEPS:
                    d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
                    if d is not None: dr_synth[di].append(float(np.mean(recon_score(d) > thr_synth)))

                # 💡 [수정 2] 이번 run 의 모델·옵티마이저·로더를 명시적으로 해제 후 GPU 캐시 비우기
                del model, opt, loader
                free_gpu()
            return dr_real, dr_synth

        def run_svm():
            print(f"  [MK-MMD SVM] 10-Run 가속 최적화 중...")
            dr_real, dr_synth = {c: [] for c in DAMAGE_CASES_B}, {di: [] for di in SYNTHETIC_DI_STEPS}
            all_h = load_damage_ganae(os.path.join(DIR_B_RAW, FILE_B))
            if all_h is None: all_h = np.zeros((1, CHANNELS, WINDOW_SIZE), dtype=np.float32)

            feat_cache = {}
            for c in DAMAGE_CASES_B:
                d = load_damage_ganae(os.path.join(DIR_B_RAW, f"D3_{c}_1.txt"))
                if d is not None: feat_cache[('r', c)] = extract_features_svm(d)

            for di in SYNTHETIC_DI_STEPS:
                d = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di:.2f}.txt"), is_synth=True)
                if d is not None: feat_cache[('s', di)] = extract_features_svm(d)

            np.random.seed(42)
            sh = all_h.copy()
            np.random.shuffle(sh)
            si  = int(len(sh) * 0.7)
            tr0, vl0 = sh[:si], sh[si:]
            tr0_f, vl0_f = extract_features_svm(tr0), extract_features_svm(vl0)

            # 빈 데이터 에러 우회 방어막
            t_l_raw = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.20.txt"), is_synth=True)
            t_m_raw = load_damage_ganae(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.50.txt"), is_synth=True)
            if t_l_raw is None or len(t_l_raw) == 0: t_l_raw = np.zeros((1, CHANNELS, WINDOW_SIZE), dtype=np.float32)
            if t_m_raw is None or len(t_m_raw) == 0: t_m_raw = np.zeros((1, CHANNELS, WINDOW_SIZE), dtype=np.float32)
                
            t_l = extract_features_svm(t_l_raw)
            t_m = extract_features_svm(t_m_raw)
            
            sc0 = RobustScaler().fit(tr0_f)
            trs, vls, tgs_l, tgs_m = sc0.transform(tr0_f), sc0.transform(vl0_f), sc0.transform(t_l), sc0.transform(t_m)

            def objective(trial):
                pc = trial.suggest_int('pca_comp', 20, 60)
                ck = trial.suggest_int('chunk_size', 20, 60, step=10)
                gm = trial.suggest_float('gamma_mult', 0.01, 10.0, log=True)
                pca = PCA(pc, random_state=42)
                tp, vp = pca.fit_transform(trs), pca.transform(vls)
                gp_l, gp_m = pca.transform(tgs_l), pca.transform(tgs_m)
                bg = (1.0 / (np.median(pdist(tp, 'sqeuclidean')) + 1e-10)) * gm
                rc = precompute_ref(tp, bg)
                vS = mmd_scores_fast(vp, rc, ck, max(1, ck // 2))
                sL = mmd_scores_fast(gp_l, rc, ck, max(1, ck // 2))
                sM = mmd_scores_fast(gp_m, rc, ck, max(1, ck // 2))
                if len(vS) < 5 or len(sL) < 5 or len(sM) < 5: return -999.0
                return ((sL.mean() - vS.mean()) / (vS.std() + 1e-6) + (sM.mean() - vS.mean()) / (vS.std() + 1e-6)) / 2.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=SVM_OPTUNA_TRIAL)
            best = study.best_params
            final_chunk = best['chunk_size']

            for run in range(N_RUNS):
                rng = np.random.default_rng(SEEDS[run])
                sh = all_h.copy()
                rng.shuffle(sh)
                si = int(len(sh) * 0.7)
                tr, vl = sh[:si], sh[si:]
                sc = RobustScaler().fit(extract_features_svm(tr))
                pca = PCA(best['pca_comp'], random_state=SEEDS[run])
                tr_p = pca.fit_transform(sc.transform(extract_features_svm(tr)))
                vl_p = pca.transform(sc.transform(extract_features_svm(vl)))
                bg = 1.0 / (np.median(pdist(tr_p, 'sqeuclidean')) + 1e-10) * best['gamma_mult']
                rc = precompute_ref(tr_p, bg)
                v_sc = mmd_scores_fast(vl_p, rc, final_chunk, stride=2)
                thr = np.percentile(v_sc, (1 - SVM_PFA) * 100)

                def file_dr(key):
                    if key not in feat_cache: return np.nan
                    fp = pca.transform(sc.transform(feat_cache[key]))
                    s = mmd_scores_fast(fp, rc, final_chunk, max(1, final_chunk // 6))
                    return float(np.mean(s > thr)) if len(s) else np.nan

                for c in DAMAGE_CASES_B: dr_real[c].append(file_dr(('r', c)))
                for di in SYNTHETIC_DI_STEPS: dr_synth[di].append(file_dr(('s', di)))
            return dr_real, dr_synth

        # 4. 파이프라인 가동 및 병합
        print(f"\n[{case_name} | {data_mode}] 모델 구동 시작...")
        gan_dr_r, gan_dr_s = run_gan()
        free_gpu()  # 💡 [수정 2] GAN 종료 후 MemAE 시작 전 캐시 비우기
        ae_dr_r,  ae_dr_s  = run_memae()
        free_gpu()  # 💡 [수정 2] MemAE 종료 후 SVM 시작 전 캐시 비우기
        svm_dr_r, svm_dr_s = run_svm()

        # 💡 데이터 누락 방어막 적용된 함수 호출
        def safe_pod(dr_r, dr_s):
            kr = [k for k in DAMAGE_CASES_B if len(dr_r[k]) > 0 and not np.isnan(np.mean(dr_r[k]))]
            ks = [k for k in SYNTHETIC_DI_STEPS if len(dr_s[k]) > 0 and not np.isnan(np.mean(dr_s[k]))]
            
            if len(kr) == 0:
                print(f"  -> [경고] {case_name} {data_mode}: 유효한 Real Damage 데이터가 없습니다.")
                pr = (np.zeros_like(x_range_r), np.zeros_like(x_range_r), np.nan, [0, 1], [], [])
            else:
                pr = pod_from_runs(dr_r, kr, np.array(kr, float), x_range_r)
                
            if len(ks) == 0:
                print(f"  -> [경고] {case_name} {data_mode}: 유효한 Synthetic 데이터가 없습니다.")
                ps = (np.zeros_like(x_range_s), np.zeros_like(x_range_s), np.nan, [0, 1], [], [])
            else:
                ps = pod_from_runs(dr_s, ks, np.array(ks, float) * 100.0, x_range_s)
            return pr, ps

        gan_pr, gan_ps = safe_pod(gan_dr_r, gan_dr_s)
        ae_pr,  ae_ps  = safe_pod(ae_dr_r,  ae_dr_s)
        svm_pr, svm_ps = safe_pod(svm_dr_r, svm_dr_s)

        # 5. 그림 생성 및 저장
        def draw_panel(ax, xr, models, title, xlabel, xlim):
            for name, (pm, pl, a90, params, a_vals, avg_dr), color in models:
                lbl = f'{name} ($a_{{90}}$={a90:.1f}%)' if not np.isnan(a90) else f'{name} (N/A)'
                ax.plot(xr, pm, color=color, lw=3, label=lbl)
                ax.plot(xr, pl, color=color, lw=1.5, ls='--', alpha=0.6)
                if len(a_vals) > 0 and len(avg_dr) > 0:
                    ax.scatter(a_vals, avg_dr, color=color, s=22, alpha=0.5)
            ax.axhline(0.9, color='k', ls=':', lw=1.5, label='90% Target')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('POD')
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([0, xlim])
            ax.grid(True, ls=':')
            ax.legend(loc='lower right', fontsize=9)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        draw_panel(axes[0], x_range_r,
                   [('MMD-GAN', gan_pr, COLORS['MMD-GAN']),
                    ('MemAE', ae_pr, COLORS['MemAE']),
                    ('MK-MMD', svm_pr, COLORS['MK-MMD'])],
                   f"Real Damage Data ({data_mode}) - {case_name}", "Damage Case, $a$ (%)", 50)
        draw_panel(axes[1], x_range_s,
                   [('MMD-GAN', gan_ps, COLORS['MMD-GAN']),
                    ('MemAE', ae_ps, COLORS['MemAE']),
                    ('MK-MMD', svm_ps, COLORS['MK-MMD'])],
                   f"Synthetic Damage Data ({data_mode}) - {case_name}", "Mapped Damage, $a$ (%)", 100)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"Master_Ensemble_MLE_POD_Comparison_{data_mode}.png"), dpi=300)
        plt.clf()
        plt.close('all')

        print(f"\n✅ 완료! [{case_name} | {data_mode}] 파이프라인 결과 저장이 종료되었습니다.")

    finally:
        # 💡 [수정 2] 성공/실패/스킵 무관하게 항상 GPU 메모리 회수.
        #    → 다음 Case 로 넘어가기 전에 누수가 누적되지 않도록 보장한다.
        plt.close('all')
        free_gpu()


# ================================================================
# 루프 가동: 무적(Bomb-proof) Try-Except 적용
# ================================================================
if __name__ == "__main__":
    cases_to_run = [f"Case A{i}" for i in range(1, 9)]
    modes_to_run = ['DON', 'NO_DON']
    
    print(f"▶️ 전체 자동화 파이프라인 대기열: {cases_to_run}")
    
    for case in cases_to_run:
        for mode in modes_to_run:
            try:
                execute_master_pipeline(case, mode)
            except Exception as e:
                # 에러가 발생해도 전체 루프가 멈추지 않고 다음 케이스로 강제 진행
                print(f"\n❌ [치명적 오류 발생] {case} - {mode} 실행 중 에러가 발생했습니다: {e}")
                traceback.print_exc()
                print("➡️ 스크립트를 멈추지 않고 다음 케이스로 강제 진행합니다...\n")
            finally:
                # 💡 [수정 2] 한 케이스가 끝날 때마다(에러 포함) GPU 메모리를 한 번 더 회수
                free_gpu()
            
    print("\n🏁 모든 Case(A1 ~ A8)와 Mode(DON, NO_DON)의 통합 평가 시도가 완전히 종료되었습니다!")