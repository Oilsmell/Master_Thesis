# ================================================================
# Data Synthesis: Normalized Residual Transfer (Structure A -> B)
# ================================================================
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

print("=== Phase 1: Setup & Configuration ===")

DIR_A_RAW      = r"E:\Benchmark Code\benchmarktu1402-master\f_accerlerations\ds1" 
DIR_B_RAW      = r"E:\2ndstructuredata\raw data"   

# 💡 구조물 A와 B의 정상 데이터 파일명이 다를 수 있으므로 분리합니다!
FILE_HEALTHY_A = "여기에_구조물A의_정상데이터_파일명을_적어주세요.txt"  # <--- 이 부분을 실제 파일명으로 수정해 주세요! (예: healthy.txt)
FILE_HEALTHY_B = "healthyclean.txt"                

SYNTH_DATA_DIR = r"E:\2ndstructuredata\Code_9_Final_Export\Synthetic_B_Data_Residual"
if not os.path.exists(SYNTH_DATA_DIR):
    os.makedirs(SYNTH_DATA_DIR)

CHANNELS = 8
DAMAGE_CASES = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]

def load_raw_txt(path):
    try:
        raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
        return np.array(raw, dtype=np.float32).reshape(CHANNELS, -1)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# ================================================================
print("\n=== Phase 2: Load Healthy Data & Fit Scalers ===")

print("Loading Healthy Data for A and B...")
# 💡 분리된 파일명으로 각각 로드합니다.
H_A_raw = load_raw_txt(os.path.join(DIR_A_RAW, FILE_HEALTHY_A))
H_B_raw = load_raw_txt(os.path.join(DIR_B_RAW, FILE_HEALTHY_B))

min_len = min(H_A_raw.shape[1], H_B_raw.shape[1])
H_A = H_A_raw[:, :min_len].T  
H_B = H_B_raw[:, :min_len].T

# 2. 독립적인 정규화 (MinMaxScaler: 0 ~ 1)
# 구조물별 고유한 강성/진폭 차이를 통일된 스케일 공간으로 매핑
print("Fitting MinMaxScaler for A and B...")
scaler_A = MinMaxScaler()
scaler_B = MinMaxScaler()

H_A_scaled = scaler_A.fit_transform(H_A)
H_B_scaled = scaler_B.fit_transform(H_B)

# 💡 기준점 생성: DI 0.00 (잔차가 0인 구조물 B의 정상 상태) 저장
np.savetxt(os.path.join(SYNTH_DATA_DIR, "Synthetic_B_DI_0.00.txt"), H_B, delimiter='\t', fmt='%.6f')


# ================================================================
print("\n=== Phase 3: Calculate Residuals & Synthesize ===")

for case in tqdm(DAMAGE_CASES, desc="Transferring Damage"):
    # 1. 구조물 A의 손상 데이터 로드
    D_A_raw = load_raw_txt(os.path.join(DIR_A_RAW, f"D3_{case}_1.txt"))
    if D_A_raw is None:
        continue
    
    # 길이 맞춤 및 변환
    D_A = D_A_raw[:, :min_len].T
    
    # 2. 정규화 공간으로 이동
    D_A_scaled = scaler_A.transform(D_A)
    
    # 3. 잔차(Residual) 추출: 손상 신호 - 정상 신호 (정규화 상태)
    # 물리적 스케일이 제거된 '순수한 손상 패턴'만 획득
    residual_scaled = D_A_scaled - H_A_scaled
    
    # 4. 구조물 B 정상 데이터에 잔차 이식 (정규화 상태)
    Synth_B_scaled = H_B_scaled + residual_scaled
    
    # 5. 구조물 B의 물리적 스케일(원래 단위)로 복원
    Synth_B = scaler_B.inverse_transform(Synth_B_scaled)
    
    # 6. 파일 저장 (MMD-GAN에서 읽을 수 있도록 Synthetic_B_DI_0.XX 포맷으로 저장)
    # case 4 -> DI 0.04 / case 48 -> DI 0.48
    di_value = case / 100.0
    save_path = os.path.join(SYNTH_DATA_DIR, f"Synthetic_B_DI_{di_value:.2f}.txt")
    
    # MMD-GAN의 load_damage_ganae는 is_synth=True일 때 delimiter='\t'로 읽음
    np.savetxt(save_path, Synth_B, delimiter='\t', fmt='%.6f')

print(f"\n🎉 합성 완료! 정규화된 잔차 기반 합성 데이터가 다음 경로에 저장되었습니다:")
print(f"👉 {SYNTH_DATA_DIR}")