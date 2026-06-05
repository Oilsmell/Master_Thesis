# -*- coding: utf-8 -*-
"""
Case A1 ~ A8 수동 잔차 합성 데이터 생성 자동화 (NO DeepONet)
- 업데이트: 파일명 자동 탐색 기능(glob) 적용. 파일 이름 앞부분이 달라도 무조건 찾아서 실행함.
"""

import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import gc

def run_pipeline(case_name):
    print(f"\n{'='*60}")
    print(f" 🚀 [NO DON] 수동 잔차 전이 시작: {case_name}")
    print(f"{'='*60}")
    
    class Config:
        DIR_A = rf"E:\git\benchmarktu1402-master\benchmarktu1402-master\{case_name}"
        DIR_B = r"E:\2ndstructuredata\raw data" 
        FILE_B_H = "healthyclean.txt"
        
        SAVE_DIR = rf"E:\git\benchmarktu1402-master\benchmarktu1402-master\{case_name}\Synthetic_B_Data_Manual"
        
        WINDOW_SIZE = 128
        CHANNELS = 8
        SELECTED_NODES = [3, 21, 39, 57, 63, 81, 99, 117]
        DAMAGE_CASES_A = list(range(1, 11))

    if not os.path.exists(Config.SAVE_DIR): 
        os.makedirs(Config.SAVE_DIR)

    # 💡 [핵심 수정] 파일 이름이 달라도 무조건 찾아내는 자동 탐색 로직
    def load_data(is_A=False, case_num=None):
        try:
            if is_A:
                if case_num is None:
                    # Healthy 파일 찾기 (*_H_accelerations.dat)
                    search_pattern = os.path.join(Config.DIR_A, "*_H_accelerations.dat")
                    files = glob.glob(search_pattern)
                    if not files:
                        print(f"  -> [오류] {Config.DIR_A} 경로에 Healthy 파일이 없습니다!")
                        return None
                    path = files[0]
                else:
                    # Damaged 파일 찾기 (*_D1_accelerations.dat 등)
                    search_pattern = os.path.join(Config.DIR_A, f"*_D{case_num}_accelerations.dat")
                    files = glob.glob(search_pattern)
                    if not files:
                        print(f"  -> [오류] {Config.DIR_A} 경로에 D{case_num} 손상 파일이 없습니다!")
                        return None
                    path = files[0]
                
                data = np.loadtxt(path)
                data = data[:, Config.SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[Config.SELECTED_NODES, :]
            else:
                path = os.path.join(Config.DIR_B, Config.FILE_B_H if case_num is None else f"D3_{case_num}_1.txt")
                raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
                data = np.array(raw, dtype=np.float32).reshape(Config.CHANNELS, -1)
                
            ns = data.shape[1] // Config.WINDOW_SIZE
            return data[:, :ns * Config.WINDOW_SIZE].astype(np.float32)
        except Exception as e:
            print(f"  -> [오류] 데이터 로딩 실패 (is_A={is_A}, case={case_num}): {e}")
            return None

    def reshape_for_scaler(data):
        ns = data.shape[1] // Config.WINDOW_SIZE
        return data.reshape(Config.CHANNELS, ns, Config.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1), ns

    # 1. Healthy 데이터 로드
    raw_h_A = load_data(is_A=True)
    raw_h_B = load_data(is_A=False)
    
    if raw_h_A is None or raw_h_B is None:
        print(f"❌ {case_name}의 핵심 데이터를 찾을 수 없어 건너뜁니다.")
        return
    else:
        print(f"✅ {case_name}의 Healthy 데이터 로드 성공!")

    reshaped_h_A, ns_A = reshape_for_scaler(raw_h_A)
    reshaped_h_B, ns_B = reshape_for_scaler(raw_h_B)
    
    scaler_A = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_A)
    scaler_B = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_B)
    
    norm_h_A = scaler_A.transform(reshaped_h_A)
    norm_h_B = scaler_B.transform(reshaped_h_B)
    
    min_ns = min(ns_A, ns_B)
    norm_h_A_cut = norm_h_A[:min_ns]
    norm_h_B_cut = norm_h_B[:min_ns]

    # 2. DI 0.00 (Healthy) 저장
    gen_phys_0 = scaler_B.inverse_transform(norm_h_B_cut)
    gen_data_0 = gen_phys_0.reshape(min_ns, Config.CHANNELS, Config.WINDOW_SIZE).transpose(1, 0, 2).reshape(Config.CHANNELS, -1)
    np.savetxt(os.path.join(Config.SAVE_DIR, "Synthetic_B_DI_0.00.txt"), gen_data_0.T, fmt='%.6e', delimiter='\t')
    print("  -> Saved: Synthetic_B_DI_0.00.txt (Healthy)")

    # 3. Damage Cases (1~10) 잔차 이식
    di_steps = np.round(np.arange(0.10, 1.01, 0.10), 2)
    
    for idx, c in enumerate(Config.DAMAGE_CASES_A):
        raw_d_A = load_data(is_A=True, case_num=c)
        if raw_d_A is None:
            continue
            
        reshaped_d_A, ns_d_A = reshape_for_scaler(raw_d_A)
        norm_d_A = scaler_A.transform(reshaped_d_A)
        
        actual_min = min(min_ns, ns_d_A)
        residual = norm_d_A[:actual_min] - norm_h_A_cut[:actual_min]
        norm_gen_B = norm_h_B_cut[:actual_min] + residual
        
        gen_phys_B = scaler_B.inverse_transform(norm_gen_B)
        gen_data_B = gen_phys_B.reshape(actual_min, Config.CHANNELS, Config.WINDOW_SIZE).transpose(1, 0, 2).reshape(Config.CHANNELS, -1)
        
        di_label = di_steps[idx]
        file_name = f"Synthetic_B_DI_{di_label:.2f}.txt"
        np.savetxt(os.path.join(Config.SAVE_DIR, file_name), gen_data_B.T, fmt='%.6e', delimiter='\t')
        print(f"  -> Saved: {file_name} (from D{c})")

    print(f"🎉 {case_name} 수동 잔차 데이터 생성 완료!")
    gc.collect()

if __name__ == "__main__":
    target_cases = [f"Case A{i}" for i in range(1, 9)]
    
    for case in target_cases:
        run_pipeline(case)
        
    print("\n🏁 모든 Case(A1 ~ A8)의 수동 잔차 전이 자동화가 완벽하게 종료되었습니다!")