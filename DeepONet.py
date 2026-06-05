# -*- coding: utf-8 -*-
"""
Case A1 ~ Case A8 자동화 파이프라인
- 목표: 리스트에 정의된 모든 Case 폴더를 순회하며 DeepONet 학습 및 합성 데이터 51개 자동 생성
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg') # 플롯을 화면에 띄우지 않고 백그라운드에서 처리
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gc

# =========================================================
# 1. Models & MMD Functions (전역 선언)
# =========================================================
def rbf_kernel(x, y, gamma=1.0):
    x = x.unsqueeze(1); y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss(x, y, gamma=1.0):
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

class Autoencoder(nn.Module):
    def __init__(self, input_dim=128*8, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 256), nn.SiLU(),
            nn.Linear(256, 512), nn.SiLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class FourierFeature(nn.Module):
    def __init__(self, input_dim, mapping_size=128, scale=10):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    def forward(self, x):
        projected = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

class DeepONet(nn.Module):
    def __init__(self, branch_dim=9, trunk_dim=2, hidden_dim=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fourier = FourierFeature(trunk_dim)
        self.trunk = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, branch_in, trunk_in):
        B = self.branch(branch_in)
        T = self.trunk(self.fourier(trunk_in))
        return torch.matmul(B, T.T) + self.bias

# =========================================================
# 2. Main Automation Pipeline
# =========================================================
def run_pipeline(case_name):
    print(f"\n{'='*60}")
    print(f" 🚀 Starting Automated Pipeline for: {case_name}")
    print(f"{'='*60}")
    
    # 해당 Case에 맞춘 동적 환경 설정
    class Config:
        DIR_A = rf"E:\git\benchmarktu1402-master\benchmarktu1402-master\{case_name}"
        SAVE_DIR = DIR_A
        DIR_B = r"E:\2ndstructuredata\raw data" 
        FILE_B_H = "healthyclean.txt"
        
        WINDOW_SIZE = 128
        LATENT_DIM = 8
        SELECTED_NODES = [3, 21, 39, 57, 63, 81, 99, 117]
        DAMAGE_CASES_A = list(range(1, 11))
        
        AE_EPOCHS = 500
        DON_EPOCHS = 500
        BATCH_SIZE = 128
        LR = 0.001
        
        LAMBDA_MMD = 0.3
        RESIDUAL_SCALE = 0.9 
        SYNTHETIC_DI_STEPS = np.arange(0.0, 1.01, 0.02) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    if not os.path.exists(cfg.SAVE_DIR): os.makedirs(cfg.SAVE_DIR)

    # 유틸리티 함수: 파일명을 case_name에 맞춰 동적으로 불러옴
    def load_data(is_A=False, case_num=None):
        try:
            if is_A:
                if case_num is None:
                    path = os.path.join(cfg.DIR_A, f"{case_name}_H_accelerations.dat")
                else:
                    path = os.path.join(cfg.DIR_A, f"{case_name}_D{case_num}_accelerations.dat")
                data = np.loadtxt(path)
                data = data[:, cfg.SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[cfg.SELECTED_NODES, :]
            else:
                if case_num is None:
                    path = os.path.join(cfg.DIR_B, cfg.FILE_B_H)
                else:
                    path = os.path.join(cfg.DIR_B, f"D3_{case_num}_1.txt")
                raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
                data = np.array(raw, dtype=np.float32).reshape(8, -1)
                
            n_samples = data.shape[1] // cfg.WINDOW_SIZE
            return data[:, :n_samples * cfg.WINDOW_SIZE].astype(np.float32)
        except Exception as e:
            print(f"Error loading (is_A={is_A}, case={case_num} in {case_name}): {e}")
            return None

    def calc_di(healthy, damaged):
        window = 2000
        n_wins = min(healthy.shape[1], damaged.shape[1]) // window
        if n_wins < 1: n_wins = 1; window = min(healthy.shape[1], damaged.shape[1])
        di_list = [abs(np.percentile(kurtosis(healthy[i, :n_wins*window].reshape(n_wins, window), axis=1, fisher=False), 95) - 
                       np.percentile(kurtosis(damaged[i, :n_wins*window].reshape(n_wins, window), axis=1, fisher=False), 95)) 
                   for i in range(8)]
        return np.mean(di_list)

    def reshape_for_scaler(data):
        ns = data.shape[1] // cfg.WINDOW_SIZE
        return data.reshape(8, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1), ns

    # 검증 로직 시작
    log_file_path = os.path.join(cfg.SAVE_DIR, f"Validation_Log_{case_name}.txt")
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + "\n")

    log_print(f"=== Phase 1: Data Loading & Normalization ({case_name}) ===")
    raw_h_A = load_data(is_A=True)
    raw_h_B = load_data(is_A=False)
    
    # 데이터가 없으면 루프 스킵 (에러 방지)
    if raw_h_A is None or raw_h_B is None:
        log_print(f"❌ {case_name}의 데이터를 불러올 수 없어 건너뜁니다.")
        log_file.close()
        return

    reshaped_h_A, ns_A = reshape_for_scaler(raw_h_A)
    reshaped_h_B, ns_B = reshape_for_scaler(raw_h_B)
    
    scaler_A = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_A)
    scaler_B = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_B)
    
    norm_h_A = scaler_A.transform(reshaped_h_A)
    norm_h_B = scaler_B.transform(reshaped_h_B)
    
    log_print(f"=== Phase 2: Autoencoder Training (MMD={cfg.LAMBDA_MMD}) ===")
    ae = Autoencoder().to(cfg.device)
    optimizer_ae = optim.Adam(ae.parameters(), lr=cfg.LR)
    criterion_recon = nn.MSELoss()
    
    loader_A = DataLoader(TensorDataset(torch.FloatTensor(norm_h_A)), batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    loader_B = DataLoader(TensorDataset(torch.FloatTensor(norm_h_B)), batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    
    for epoch in range(cfg.AE_EPOCHS):
        ae.train()
        total_recon = 0
        for (batch_A,), (batch_B,) in zip(loader_A, loader_B):
            b_A, b_B = batch_A.to(cfg.device), batch_B.to(cfg.device)
            optimizer_ae.zero_grad()
            
            recon_A, z_A_batch = ae(b_A)
            recon_B, z_B_batch = ae(b_B)
            
            loss_r = criterion_recon(recon_A, b_A) + criterion_recon(recon_B, b_B)
            loss_m = mmd_loss(z_A_batch, z_B_batch)
            loss = loss_r + (cfg.LAMBDA_MMD * loss_m)
            
            loss.backward()
            optimizer_ae.step()
            total_recon += loss_r.item()
            
        if (epoch+1) % 100 == 0: 
            log_print(f"   AE Epoch {epoch+1}/{cfg.AE_EPOCHS} | Recon Loss: {total_recon/len(loader_A):.6f}")
    
    ae.eval()
    with torch.no_grad():
        z_A = ae.encoder(torch.FloatTensor(norm_h_A).to(cfg.device)).cpu().numpy()
        z_B = ae.encoder(torch.FloatTensor(norm_h_B).to(cfg.device)).cpu().numpy()

    log_print(f"\n=== Phase 3: DeepONet Residual Training ({case_name} D1~10) ===")
    dis_A_raw = []
    for c in cfg.DAMAGE_CASES_A:
        raw_d = load_data(is_A=True, case_num=c)
        if raw_d is not None:
            dis_A_raw.append(calc_di(raw_h_A, raw_d))
        else:
            dis_A_raw.append(0) # 예외 처리
            
    min_di_A, max_di_A = min(dis_A_raw), max(dis_A_raw)
    
    X_branch, Y_target = [], []
    for i, c in enumerate(cfg.DAMAGE_CASES_A):
        raw_d = load_data(is_A=True, case_num=c)
        if raw_d is None: continue
        
        reshaped_d, _ = reshape_for_scaler(raw_d)
        norm_d = scaler_A.transform(reshaped_d)
        
        target_res = (norm_d - norm_h_A) * cfg.RESIDUAL_SCALE 
        di_norm = (dis_A_raw[i] - min_di_A) / ((max_di_A - min_di_A) + 1e-8)
        
        b_in = np.hstack([z_A, np.full((ns_A, 1), di_norm)])
        X_branch.append(b_in)
        Y_target.append(target_res)
        
    X_branch = np.vstack(X_branch)
    Y_target = np.vstack(Y_target)
    
    don = DeepONet(branch_dim=9, trunk_dim=2).to(cfg.device)
    optimizer_don = optim.Adam(don.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer_don, step_size=300, gamma=0.5) 
    
    T_grid, X_grid = np.meshgrid(np.linspace(0, 1, cfg.WINDOW_SIZE), np.linspace(0, 1, 8))
    trunk_in = torch.FloatTensor(np.stack([T_grid.flatten(), X_grid.flatten()], axis=1)).to(cfg.device)
    loader_don = DataLoader(TensorDataset(torch.FloatTensor(X_branch), torch.FloatTensor(Y_target)), batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    for epoch in range(cfg.DON_EPOCHS):
        don.train()
        total_loss = 0
        for b_in, target in loader_don:
            optimizer_don.zero_grad()
            preds = don(b_in.to(cfg.device), trunk_in)
            loss = nn.MSELoss()(preds, target.to(cfg.device))
            loss.backward()
            optimizer_don.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 100 == 0: 
            log_print(f"   DeepONet Epoch {epoch+1}/{cfg.DON_EPOCHS} | Loss: {total_loss/len(loader_don):.6f}")

    log_print(f"\n=== Phase 4: Generate DENSE Synthetic Data (51 Steps for {case_name}) ===")
    
    synth_dir = os.path.join(cfg.SAVE_DIR, "Synthetic_B_Data")
    if not os.path.exists(synth_dir): os.makedirs(synth_dir)
    
    don.eval()
    gen_dis_B_dense = []
    
    for di_norm in cfg.SYNTHETIC_DI_STEPS:
        b_in = torch.FloatTensor(np.hstack([z_B, np.full((len(norm_h_B), 1), di_norm)])).to(cfg.device)
        
        with torch.no_grad():
            pred_res = don(b_in, trunk_in).cpu().numpy() / cfg.RESIDUAL_SCALE
            
        gen_phys = scaler_B.inverse_transform(norm_h_B + pred_res)
        gen_data = gen_phys.reshape(len(norm_h_B), 8, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(8, -1)
        
        gen_di = calc_di(raw_h_B, gen_data)
        gen_dis_B_dense.append(gen_di)
        log_print(f"Input DI: {di_norm:<4.2f} | Generated Real DI: {gen_di:.6f}")
        
        file_name = f"Synthetic_B_DI_{di_norm:.2f}.txt"
        save_path = os.path.join(synth_dir, file_name)
        np.savetxt(save_path, gen_data.T, fmt='%.6e', delimiter='\t')

    # =========================================================
    # 5. Save Growth Map
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(cfg.SYNTHETIC_DI_STEPS, gen_dis_B_dense, 'b-o', linewidth=2, markersize=6, label=f'Synthetic DI Growth ({case_name} Trained)')
    
    base_di = gen_dis_B_dense[0]
    plt.axhline(y=base_di, color='green', linestyle='--', alpha=0.7, label=f'Healthy Baseline (DI={base_di:.6f})')
    
    plt.title(f"Structure B: Synthetic Damage Index Growth Map (Trained on {case_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Input Damage Factor (0.02 steps)", fontsize=12)
    plt.ylabel("Generated Kurtosis Damage Index", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(cfg.SAVE_DIR, f"Synthetic_B_Growth_Map_{case_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close('all') # 메모리 누수 방지

    log_print(f"\n✅ {case_name} 완료. 총 {len(cfg.SYNTHETIC_DI_STEPS)}개의 합성 데이터 파일이 저장되었습니다.")
    log_file.close()

    # GPU 메모리 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# =========================================================
# 실행 블록: 리스트에 정의된 Case들을 순회하며 실행
# =========================================================
if __name__ == "__main__":
    # Case A1부터 Case A8까지 정의
    target_cases = [f"Case A{i}" for i in range(1, 9)]
    
    for case in target_cases:
        run_pipeline(case)
        
    print("\n🎉 모든 Case(A1 ~ A8)의 자동화 처리가 완료되었습니다!")