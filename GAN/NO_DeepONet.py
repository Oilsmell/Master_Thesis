# =========================================================
# Manual Residual Injection (1:1 Window Matching) & Baseline POD
# =========================================================
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1. Configuration
# =========================================================
class Config:
    DIR_A   = r"E:\Benchmark Code\benchmarktu1402-master\f_accerlerations\ds1"
    DIR_B   = r"E:\2ndstructuredata\raw data"
    FILE_B  = "healthyclean.txt"
    
    SAVE_DIR = r"E:\2ndstructuredata\Code_37_Manual_Residual"
    
    WINDOW_SIZE     = 128
    SELECTED_NODES  = [3, 21, 39, 57, 63, 81, 99, 117]   # A의 8 채널
    DAMAGE_CASES_A  = list(range(1, 11))                  # f1 ~ f10
    
    # 💡 구조물 B의 실제 손상 케이스 (Real POD 평가용)
    DAMAGE_CASES_B  = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    
    RESIDUAL_SCALE = 1.0

cfg = Config()
if not os.path.exists(cfg.SAVE_DIR): 
    os.makedirs(cfg.SAVE_DIR)

synth_dir = os.path.join(cfg.SAVE_DIR, "Synthetic_B_Data")
if not os.path.exists(synth_dir):
    os.makedirs(synth_dir)


# =========================================================
# 2. Utility Functions
# =========================================================
def load_data(path, is_A=False):
    try:
        if is_A:
            data = np.loadtxt(path)
            data = data[:, cfg.SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[cfg.SELECTED_NODES, :]
        else:
            raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(8, -1)
        n_samples = data.shape[1] // cfg.WINDOW_SIZE
        return data[:, :n_samples * cfg.WINDOW_SIZE].astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def reshape_for_scaler(data):
    ns = data.shape[1] // cfg.WINDOW_SIZE
    reshaped = data.reshape(8, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1)
    return reshaped, ns

def calc_di(healthy, damaged):
    window = 2000
    n_wins = min(healthy.shape[1], damaged.shape[1]) // window
    if n_wins < 1: 
        n_wins = 1
        window = min(healthy.shape[1], damaged.shape[1])
    di_list = []
    for i in range(8):
        h_k = kurtosis(healthy[i, :n_wins*window].reshape(n_wins, window), axis=1, fisher=False)
        d_k = kurtosis(damaged[i, :n_wins*window].reshape(n_wins, window), axis=1, fisher=False)
        di_list.append(abs(np.percentile(h_k, 95) - np.percentile(d_k, 95)))
    return np.mean(di_list)

# 💡 베이스라인 POD용 간이 점수 추출 함수 (물리적 RMS 기반)
def get_window_scores(data):
    ns = data.shape[1] // cfg.WINDOW_SIZE
    windows = data[:, :ns * cfg.WINDOW_SIZE].reshape(8, ns, cfg.WINDOW_SIZE)
    rms = np.sqrt(np.mean(windows**2, axis=2)) 
    return np.mean(rms, axis=0) 

def probit_func(x, mu, sigma):
    return norm.cdf(x, loc=mu, scale=sigma)

# =========================================================
# 3. Main Pipeline
# =========================================================
def main():
    print("=" * 70)
    print("  [Code_37] Manual Residual Injection & Baseline POD")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────
    # Phase 1: Load + Healthy-based Independent MinMaxScaler
    # ─────────────────────────────────────────────────────────
    print("\n=== Phase 1: Independent MinMaxScaler [0,1] ===")
    
    raw_h_A = load_data(os.path.join(cfg.DIR_A, "fh_accelerations.dat"), is_A=True)
    raw_h_B = load_data(os.path.join(cfg.DIR_B, cfg.FILE_B), is_A=False)
    
    if raw_h_A is None or raw_h_B is None:
        print("❌ Failed to load healthy data!")
        return
    
    reshaped_h_A, ns_A = reshape_for_scaler(raw_h_A)
    reshaped_h_B, ns_B = reshape_for_scaler(raw_h_B)
    
    min_ns = min(ns_A, ns_B)
    print(f"   A windows: {ns_A},  B windows: {ns_B} -> Aligned to {min_ns} windows")
    
    scaler_A = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_A)
    scaler_B = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_B)
    
    with open(os.path.join(cfg.SAVE_DIR, "scaler_A.pkl"), 'wb') as f:
        pickle.dump(scaler_A, f)
    with open(os.path.join(cfg.SAVE_DIR, "scaler_B.pkl"), 'wb') as f:
        pickle.dump(scaler_B, f)
    
    norm_h_A = scaler_A.transform(reshaped_h_A)[:min_ns, :] 
    norm_h_B = scaler_B.transform(reshaped_h_B)[:min_ns, :] 
    
    # ─────────────────────────────────────────────────────────
    # Phase 2: Extract 1:1 Window Matching Residuals
    # ─────────────────────────────────────────────────────────
    print("\n=== Phase 2: Extract 1:1 Window Residuals from A ===")
    
    residuals = {}    
    dis_A_raw = {}    
    
    for c in cfg.DAMAGE_CASES_A:
        path = os.path.join(cfg.DIR_A, f"f{c}_accelerations.dat")
        raw_d = load_data(path, is_A=True)
        if raw_d is None: continue
        
        reshaped_d, _ = reshape_for_scaler(raw_d)
        norm_d = scaler_A.transform(reshaped_d)[:min_ns, :]   
        
        residual_per_sample = (norm_d - norm_h_A) * cfg.RESIDUAL_SCALE
        residuals[c] = residual_per_sample
        
        di = calc_di(raw_h_A, raw_d)
        dis_A_raw[c] = di
        
    sorted_cases = sorted(residuals.keys(), key=lambda c: dis_A_raw[c])
    
    # ─────────────────────────────────────────────────────────
    # Phase 3: Inject Residuals into B's Healthy Data
    # ─────────────────────────────────────────────────────────
    print("\n=== Phase 3: Inject 1:1 Residuals into B ===")
    
    gen_dis_B = {} 
    gen_data_dict = {} # POD 평가용으로 메모리에 보관
    di_steps_output = np.linspace(0.1, 1.0, len(sorted_cases))
    
    for i, c in enumerate(sorted_cases):
        di_step = di_steps_output[i]
        norm_synth_B = norm_h_B + residuals[c]
        gen_phys = scaler_B.inverse_transform(norm_synth_B)
        gen_data = gen_phys.reshape(min_ns, 8, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(8, -1)
        
        gen_data_dict[di_step] = gen_data
        gen_di = calc_di(raw_h_B, gen_data)
        gen_dis_B[c] = gen_di
        
        save_path = os.path.join(synth_dir, f"Synthetic_B_DI_{di_step:.2f}.txt")
        np.savetxt(save_path, gen_data.T, fmt='%.6e', delimiter='\t')
        print(f"   f{c:2d} → DI_step={di_step:.2f} | B_gen_DI={gen_di:.6f}")
    
    healthy_phys = scaler_B.inverse_transform(norm_h_B)
    healthy_data = healthy_phys.reshape(min_ns, 8, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(8, -1)
    np.savetxt(os.path.join(synth_dir, "Synthetic_B_DI_0.00.txt"), healthy_data.T, fmt='%.6e', delimiter='\t')
    
    # ─────────────────────────────────────────────────────────
    # Phase 4 & 5 (생략: 기존의 Diagnostic Plots, Waveforms 저장은 유지됨)
    # ─────────────────────────────────────────────────────────
    # ... (기존 코드가 동작하여 차트 2개를 저장합니다)

    # ─────────────────────────────────────────────────────────
    # Phase 6: Baseline Statistical POD (Real vs Synth)
    # ─────────────────────────────────────────────────────────
    print("\n=== Phase 6: Baseline Statistical POD (Real vs Synthetic B) ===")
    
    # 1. 임계값 설정 (Healthy B 기준)
    h_scores = get_window_scores(raw_h_B)
    threshold = np.mean(h_scores) + 2 * np.std(h_scores)
    print(f"   Baseline Threshold (Mean + 2*Std): {threshold:.6f}")

    # 2. Real B Data 탐지율(DR) 계산
    dr_real = []
    for c in cfg.DAMAGE_CASES_B:
        path = os.path.join(cfg.DIR_B, f"D3_{c}_1.txt")
        d = load_data(path, is_A=False)
        if d is not None:
            scores = get_window_scores(d)
            dr_real.append(np.mean(scores > threshold))
        else:
            dr_real.append(0.0)

    # 3. Synthetic B Data 탐지율(DR) 계산
    dr_synth = []
    for di_step in di_steps_output:
        scores = get_window_scores(gen_data_dict[di_step])
        dr_synth.append(np.mean(scores > threshold))

    # 4. Probit 곡선 피팅
    x_range_r = np.linspace(0, 50, 200)
    x_range_s = np.linspace(0, 100, 200)
    
    try:
        popt_r, _ = curve_fit(probit_func, cfg.DAMAGE_CASES_B, dr_real, bounds=([0, 0], [100, 50]))
        pod_mean_r = probit_func(x_range_r, *popt_r)
        a90_r = x_range_r[np.argmax(pod_mean_r >= 0.9)] if np.any(pod_mean_r >= 0.9) else np.nan
    except:
        pod_mean_r = np.zeros_like(x_range_r); a90_r = np.nan

    try:
        popt_s, _ = curve_fit(probit_func, di_steps_output * 100, dr_synth, bounds=([0, 0], [150, 50]))
        pod_mean_s = probit_func(x_range_s, *popt_s)
        a90_s = x_range_s[np.argmax(pod_mean_s >= 0.9)] if np.any(pod_mean_s >= 0.9) else np.nan
    except:
        pod_mean_s = np.zeros_like(x_range_s); a90_s = np.nan

    print(f"   Baseline POD | Real a90: {a90_r:.1f}%, Synth a90: {a90_s:.1f}%")

    # 5. POD 비교 차트 출력
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(x_range_r, pod_mean_r, color='#2ECC71', linewidth=3, label=f'Baseline POD (a90={a90_r:.1f}%)')
    axes[0].plot(cfg.DAMAGE_CASES_B, dr_real, 'ko', alpha=0.5, label='Actual DR')
    axes[0].axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
    axes[0].set_title("Real Damage: Baseline Statistical POD", fontweight='bold')
    axes[0].set_xlabel("Damage Case (%)"); axes[0].set_ylabel("Probability of Detection")
    axes[0].set_ylim([0, 1.05]); axes[0].grid(True, linestyle=':', alpha=0.7); axes[0].legend(loc='lower right')
    
    axes[1].plot(x_range_s, pod_mean_s, color='#3498DB', linewidth=3, label=f'Baseline POD (a90={a90_s:.1f}%)')
    axes[1].plot(di_steps_output * 100, dr_synth, 'ko', alpha=0.5, label='Synthetic DR')
    axes[1].axhline(0.9, color='black', linestyle=':', linewidth=2, label='90% Target')
    axes[1].set_title("Synthetic Damage: Baseline Statistical POD", fontweight='bold')
    axes[1].set_xlabel("Mapped Synthetic Damage (%)"); axes[1].set_ylabel("Probability of Detection")
    axes[1].set_ylim([0, 1.05]); axes[1].grid(True, linestyle=':', alpha=0.7); axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.SAVE_DIR, "Manual_Residual_POD_Comparison.png"), dpi=300)
    plt.show()
    
    print("\n" + "=" * 70)
    print(f"✅ All Completed! Saved Baseline POD curves.")
    print("=" * 70)

if __name__ == "__main__":
    main()