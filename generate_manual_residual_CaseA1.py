# -*- coding: utf-8 -*-
"""
Case A1 수동 잔차 합성 데이터 생성 (NO DeepONet)
- 목표: DeepONet 없이 구조물 A의 잔차를 구조물 B에 직접 이식
- 출력: 0.00 ~ 1.00 (0.1 간격) 총 11개의 합성 데이터 파일 저장
"""

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class Config:
    DIR_A = r"E:\git\benchmarktu1402-master\benchmarktu1402-master\Case A1"
    DIR_B = r"E:\2ndstructuredata\raw data" 
    FILE_B_H = "healthyclean.txt"
    
    # DON(딥오넷) 결과와 섞이지 않도록 Manual 폴더로 분리
    SAVE_DIR = r"E:\git\benchmarktu1402-master\benchmarktu1402-master\Case A1\Synthetic_B_Data_Manual"
    
    WINDOW_SIZE = 128
    CHANNELS = 8
    SELECTED_NODES = [3, 21, 39, 57, 63, 81, 99, 117]
    DAMAGE_CASES_A = list(range(1, 11))

if not os.path.exists(Config.SAVE_DIR): 
    os.makedirs(Config.SAVE_DIR)

def load_data(is_A=False, case_num=None):
    try:
        if is_A:
            path = os.path.join(Config.DIR_A, "Case A1_H_accelerations.dat" if case_num is None else f"Case A1_D{case_num}_accelerations.dat")
            data = np.loadtxt(path)
            data = data[:, Config.SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[Config.SELECTED_NODES, :]
        else:
            path = os.path.join(Config.DIR_B, Config.FILE_B_H if case_num is None else f"D3_{case_num}_1.txt")
            raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(Config.CHANNELS, -1)
            
        ns = data.shape[1] // Config.WINDOW_SIZE
        return data[:, :ns * Config.WINDOW_SIZE].astype(np.float32)
    except Exception as e:
        print(f"Error loading (is_A={is_A}, case={case_num}): {e}")
        return None

def reshape_for_scaler(data):
    ns = data.shape[1] // Config.WINDOW_SIZE
    return data.reshape(Config.CHANNELS, ns, Config.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1), ns

def main():
    print("=== [NO DON] 수동 잔차 전이 파이프라인 시작 ===")
    
    # 1. Healthy 데이터 로드 및 정규화
    raw_h_A = load_data(is_A=True)
    raw_h_B = load_data(is_A=False)
    
    reshaped_h_A, ns_A = reshape_for_scaler(raw_h_A)
    reshaped_h_B, ns_B = reshape_for_scaler(raw_h_B)
    
    scaler_A = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_A)
    scaler_B = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_B)
    
    norm_h_A = scaler_A.transform(reshaped_h_A)
    norm_h_B = scaler_B.transform(reshaped_h_B)
    
    # 두 구조물의 샘플 수 맞추기
    min_ns = min(ns_A, ns_B)
    norm_h_A_cut = norm_h_A[:min_ns]
    norm_h_B_cut = norm_h_B[:min_ns]

    # 2. DI 0.00 (Healthy) 저장
    gen_phys_0 = scaler_B.inverse_transform(norm_h_B_cut)
    gen_data_0 = gen_phys_0.reshape(min_ns, Config.CHANNELS, Config.WINDOW_SIZE).transpose(1, 0, 2).reshape(Config.CHANNELS, -1)
    np.savetxt(os.path.join(Config.SAVE_DIR, "Synthetic_B_DI_0.00.txt"), gen_data_0.T, fmt='%.6e', delimiter='\t')
    print("Saved: Synthetic_B_DI_0.00.txt (Healthy)")

    # 3. Damage Cases (1~10) 잔차 이식 및 저장
    di_steps = np.round(np.arange(0.10, 1.01, 0.10), 2)
    
    for idx, c in enumerate(Config.DAMAGE_CASES_A):
        raw_d_A = load_data(is_A=True, case_num=c)
        reshaped_d_A, ns_d_A = reshape_for_scaler(raw_d_A)
        norm_d_A = scaler_A.transform(reshaped_d_A)
        
        actual_min = min(min_ns, ns_d_A)
        
        # [핵심] 순수 정규화 잔차 계산: (A_Damage - A_Healthy)
        residual = norm_d_A[:actual_min] - norm_h_A_cut[:actual_min]
        
        # 구조물 B에 이식: B_Healthy + Residual
        norm_gen_B = norm_h_B_cut[:actual_min] + residual
        
        # 물리적 스케일(역정규화) 복원
        gen_phys_B = scaler_B.inverse_transform(norm_gen_B)
        gen_data_B = gen_phys_B.reshape(actual_min, Config.CHANNELS, Config.WINDOW_SIZE).transpose(1, 0, 2).reshape(Config.CHANNELS, -1)
        
        di_label = di_steps[idx]
        file_name = f"Synthetic_B_DI_{di_label:.2f}.txt"
        np.savetxt(os.path.join(Config.SAVE_DIR, file_name), gen_data_B.T, fmt='%.6e', delimiter='\t')
        print(f"Saved: {file_name} (from Case A1_D{c})")

    print(f"\n✅ 수동 잔차 합성 데이터(11개) 저장 완료: {Config.SAVE_DIR}")

if __name__ == "__main__":
    main()