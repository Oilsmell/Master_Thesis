# -*- coding: utf-8 -*-
"""
=============================================================================
Structure B - Physical Characteristic Comparison  (AUTOMATED, multi state/case)
  Healthy  vs  Manual-Residual Synthetic  vs  DeepONet Synthetic
=============================================================================
목적: POD가 비슷해도 물리적 특성(모달 구조/스펙트럼/에너지 분포)까지 보존했는지는
별개다. FEM-1 물리량(FFT/PSD, 고유진동수, RMS, 첨도, 웨이블릿 패킷 에너지)을
B의 세 데이터셋에 동일 적용해 B 건강데이터 기준으로 두 합성셋을 비교한다.

★ A 지문(fingerprint) 검출 ★
수동잔차 = B_healthy + (A_damage - A_healthy)를 '샘플 인덱스' 기준으로 더하므로
A의 모달 진동이 '정규화 주파수(cycles/sample)'의 고정 위치(0.020/0.022/0.055/0.060)에
남는다. B 수동데이터에만 이 밴드가 커지면 = A 물리 지문 오염. B의 절대 FS 몰라도 성립.

★ 자동화 ★
CFG.STATES 와 CFG.CASE_START~CASE_END 만 지정하면
  E:\...\compare1\state_{S}\Case A{i}\  (Synthetic_B_Data_Manual, Synthetic_B_Data)
전부를 순회하며 각 케이스의 Physical_Compare 폴더에 그림/CSV를 저장하고,
루트에 마스터 요약 CSV(Physical_Compare_MASTER_summary.csv)를 만든다.
건강/실측 B 데이터는 공통이므로 1회만 로드해 캐시한다.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, find_peaks
from scipy.stats import kurtosis

try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

# NumPy 2.0 호환: np.trapz 제거됨 → np.trapezoid 사용 (구버전 폴백)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ============================================================================
# 1. CONFIG  --  실행 전 이 부분만 확인/수정
# ============================================================================
class CFG:
    # ▼▼▼ 자동화 범위 지정 ▼▼▼
    ROOT       = r"E:\git\benchmarktu1402-master\benchmarktu1402-master\compare1"
    STATES     = [1, 4]        # 돌릴 state 번호들 (예: [1, 4] 또는 [1, 3, 4, 6])
    CASE_START = 0             # Case A{START} 부터
    CASE_END   = 8             # Case A{END} 까지 (양끝 포함). 나중에 늘면 이 값만 수정.
    # ▲▲▲ ------------------ ▲▲▲

    DIR_B_RAW  = r"E:\2ndstructuredata\raw data"   # B 원본(건강+실측손상), 모든 케이스 공통
    FILE_B_H   = "healthyclean.txt"

    CHANNELS    = 8
    WINDOW_SIZE = 128
    FS_B = 1000                # 알면 숫자(Hz), 모르면 None → 정규화 주파수(cycles/sample)

    # 구조물 A 정보 (지문 검출용, FEM-1 기준)
    FS_A     = 1000.0
    A_MODES  = [20.0, 22.0, 55.0, 60.0]
    A_MODES_NORM = [0.020, 0.022, 0.055, 0.060]   # = A_MODES / FS_A
    BAND_HALF = 0.0015

    OVERLAY_DI = [0.20, 0.50, 1.00]

    REAL_B_CASES = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    USE_REAL_B   = True

    NPERSEG = 1024

    # ↓↓ per-run 에서 set_run()이 자동으로 채움 (직접 수정 불필요) ↓↓
    STATE = None
    CASE_NAME = None
    MANUAL_DIR = None
    DON_DIR = None
    OUT_DIR = None


def set_run(state, case_name):
    """현재 처리할 (state, case)에 맞춰 경로를 설정."""
    base = os.path.join(CFG.ROOT, f"state_{state}", case_name)
    CFG.STATE = state
    CFG.CASE_NAME = case_name
    CFG.MANUAL_DIR = os.path.join(base, "Synthetic_B_Data_Manual")
    CFG.DON_DIR    = os.path.join(base, "Synthetic_B_Data")
    CFG.OUT_DIR    = os.path.join(base, "Physical_Compare")


def tag():
    return f"state_{CFG.STATE} | {CFG.CASE_NAME}"


# ============================================================================
# 2. DATA LOADERS  --  생성 스크립트의 저장/로드 방식과 정확히 일치
# ============================================================================
def load_stream_file(path, channels=CFG.CHANNELS):
    """B 원본(건강/실측손상): 각 줄 2번째 토큰 → (8, T) folding."""
    try:
        raw = [float(l.split()[1]) for l in open(path, "r") if len(l.split()) >= 2]
        data = np.array(raw, dtype=np.float32)
        T = len(data) // channels
        if T == 0:
            return None
        return data[:channels * T].reshape(channels, T)
    except Exception as e:
        print(f"  [load_stream 실패] {os.path.basename(path)}: {e}")
        return None


def load_synth_file(path):
    """합성: np.savetxt(gen.T, delimiter='\\t') → (T,8), .T 하여 (8,T)."""
    try:
        data = np.loadtxt(path, delimiter="\t").astype(np.float32)
        if data.ndim == 1:
            return None
        return data.T
    except Exception as e:
        print(f"  [load_synth 실패] {os.path.basename(path)}: {e}")
        return None


def synth_path(folder, di):
    return os.path.join(folder, f"Synthetic_B_DI_{di:.2f}.txt")


def available_di(folder):
    if not os.path.isdir(folder):
        return []
    dis = []
    for f in os.listdir(folder):
        if f.startswith("Synthetic_B_DI_") and f.endswith(".txt"):
            try:
                dis.append(round(float(f.replace("Synthetic_B_DI_", "").replace(".txt", "")), 2))
            except ValueError:
                pass
    return sorted(set(dis))


# ============================================================================
# 3. PHYSICAL FEATURE FUNCTIONS  (FEM-1 코드의 정석 방식)
# ============================================================================
def freq_axis(n, fs):
    if fs:
        return np.fft.rfftfreq(n, d=1.0 / fs)
    return np.fft.rfftfreq(n, d=1.0)      # cycles/sample


def compute_fft(data, fs=None):
    """8채널 평균 진폭 스펙트럼 (detrend + Hanning + 진폭보존 정규화)."""
    n = data.shape[1]
    sig = detrend(data, axis=1, type="linear")
    win = np.hanning(n)
    freqs = freq_axis(n, fs)
    mag = np.abs(np.fft.rfft(sig * win, axis=1)) * 2.0 / np.sum(win)
    return freqs, np.mean(mag, axis=0)


def compute_psd(data, fs=None, nperseg=CFG.NPERSEG):
    """Welch PSD, 8채널 평균. fs 없으면 fs=1.0 → 정규화 주파수."""
    use_fs = fs if fs else 1.0
    nper = min(nperseg, data.shape[1])
    freqs, pxx = welch(data, fs=use_fs, window="hann",
                       nperseg=nper, axis=1, detrend="linear")
    return freqs, np.mean(pxx, axis=0)


def top_peaks(freqs, mag, n_peaks=4, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = freqs[-1]
    mask = (freqs >= fmin) & (freqs <= fmax)
    f, m = freqs[mask], mag[mask]
    if len(m) < 3:
        return np.array([np.nan] * n_peaks)
    peaks, _ = find_peaks(m, prominence=np.max(m) * 0.02)
    if len(peaks) == 0:
        return np.array([np.nan] * n_peaks)
    order = np.argsort(m[peaks])[::-1][:n_peaks]
    sel = np.sort(peaks[order])
    out = list(f[sel])
    while len(out) < n_peaks:
        out.append(np.nan)
    return np.array(out[:n_peaks])


def track_nearest_peak(freqs, mag, ref_freq, search_half):
    mask = (freqs >= ref_freq - search_half) & (freqs <= ref_freq + search_half)
    if not mask.any():
        return np.nan
    f, m = freqs[mask], mag[mask]
    return f[np.argmax(m)]


def rms_p95(data, win=CFG.WINDOW_SIZE):
    vals = []
    for s in range(data.shape[0]):
        sig = data[s]
        nw = len(sig) // win
        if nw == 0:
            continue
        w = sig[:nw * win].reshape(nw, win)
        vals.append(np.percentile(np.sqrt(np.mean(w ** 2, axis=1)), 95))
    return float(np.mean(vals)) if vals else np.nan


def kurt_p95(data, win=CFG.WINDOW_SIZE):
    vals = []
    for s in range(data.shape[0]):
        sig = data[s]
        nw = len(sig) // win
        if nw == 0:
            continue
        w = sig[:nw * win].reshape(nw, win)
        vals.append(np.percentile(kurtosis(w, axis=1, fisher=False), 95))
    return float(np.mean(vals)) if vals else np.nan


def wpe_distribution(data, wavelet="db4", level=4):
    """웨이블릿 패킷 에너지 분포(밴드별 정규화 에너지) → 8채널 평균 벡터."""
    if not HAS_PYWT:
        return None
    dists = []
    for s in range(data.shape[0]):
        try:
            wp = pywt.WaveletPacket(data=data[s], wavelet=wavelet,
                                    mode="symmetric", maxlevel=level)
            nodes = [n.path for n in wp.get_level(level, "freq")]
            energies = np.array([np.sum(wp[p].data ** 2) for p in nodes])
            dists.append(energies / (energies.sum() + 1e-20))
        except Exception:
            continue
    if not dists:
        return None
    return np.mean(np.stack(dists, 0), axis=0)


def a_fingerprint_ratio(freqs, mag):
    """A의 정규화 모달 밴드에 집중된 스펙트럼 에너지 비율.
       freqs 는 반드시 정규화 주파수(cycles/sample)."""
    tot = _trapz(mag, freqs) + 1e-20
    band = 0.0
    for fm in CFG.A_MODES_NORM:
        mask = (freqs >= fm - CFG.BAND_HALF) & (freqs <= fm + CFG.BAND_HALF)
        if mask.any():
            band += _trapz(mag[mask], freqs[mask])
    return band / tot


# ============================================================================
# 4. DATA LOADING  (공통 캐시 + per-run 합성)
# ============================================================================
_SHARED = {"healthy": None, "real": None, "loaded": False}


def load_shared():
    """건강 + 실측 B 손상 데이터를 1회만 로드해 캐시 (모든 state/case 공통)."""
    if _SHARED["loaded"]:
        return _SHARED["healthy"], _SHARED["real"]
    print("[공통 데이터 로드] B 건강 + 실측 손상 ...")
    healthy = load_stream_file(os.path.join(CFG.DIR_B_RAW, CFG.FILE_B_H))
    print(f"  B Healthy: {'OK ' + str(healthy.shape) if healthy is not None else 'MISSING'}")
    real = {}
    if CFG.USE_REAL_B:
        for c in CFG.REAL_B_CASES:
            d = load_stream_file(os.path.join(CFG.DIR_B_RAW, f"D3_{c}_1.txt"))
            if d is not None:
                real[c] = d
        print(f"  Real B damage cases: {list(real.keys())}")
    _SHARED.update(healthy=healthy, real=real, loaded=True)
    return healthy, real


def load_run_synth(healthy, real):
    """현재 CFG 경로의 수동/DON 합성 데이터를 로드하여 ds 구성."""
    man_di = available_di(CFG.MANUAL_DIR)
    don_di = available_di(CFG.DON_DIR)
    print(f"  Manual DI : {man_di}")
    print(f"  DeepONet  : {len(don_di)} files")
    ds = {"healthy": healthy, "real": real,
          "manual": {di: load_synth_file(synth_path(CFG.MANUAL_DIR, di)) for di in man_di},
          "don":    {di: load_synth_file(synth_path(CFG.DON_DIR, di))    for di in don_di}}
    ds["manual"] = {k: v for k, v in ds["manual"].items() if v is not None}
    ds["don"]    = {k: v for k, v in ds["don"].items()    if v is not None}
    return ds


# ============================================================================
# 5. PLOTS
# ============================================================================
def FLABEL():
    return "Frequency (Hz)" if CFG.FS_B else "Normalized Frequency (cycles/sample)"


def FMAX():
    return CFG.FS_B / 2 if CFG.FS_B else 0.5


def _a_mode_lines(ax):
    for fm_norm, fm_hz in zip(CFG.A_MODES_NORM, CFG.A_MODES):
        xpos = fm_hz if (CFG.FS_B == CFG.FS_A) else (fm_norm * CFG.FS_B if CFG.FS_B else fm_norm)
        ax.axvline(xpos, color="green", ls=":", lw=1.0, alpha=0.6)


def _style_spec(ax, title, method):
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel(FLABEL())
    ax.set_ylabel("Amplitude" if method == "FFT" else "PSD")
    ax.set_xlim(0, FMAX())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_spectra_overlay(ds):
    """[Fig 1] 대표 DI에서 FFT/PSD.
       ★ 겹침 방지: 3행 = Healthy / Manual / DeepONet 를 각각 따로 (한 칸 한 데이터셋)."""
    rows = [("B Healthy", "healthy", "k"),
            ("Manual",    "manual",  "#1f77b4"),
            ("DeepONet",  "don",     "#d62728")]
    for method, comp in [("FFT", compute_fft), ("PSD", compute_psd)]:
        ncol = len(CFG.OVERLAY_DI)
        fig, axes = plt.subplots(3, ncol, figsize=(6 * ncol, 13), squeeze=False)

        for r, (rlabel, rkey, rcolor) in enumerate(rows):
            for j, di in enumerate(CFG.OVERLAY_DI):
                ax = axes[r][j]
                if rkey == "healthy":
                    data = ds["healthy"]
                else:
                    data = ds[rkey].get(round(di, 2))
                if data is not None:
                    f, m = comp(data, CFG.FS_B)
                    ax.semilogy(f, m + 1e-20, color=rcolor, lw=1.3, label=rlabel)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                _a_mode_lines(ax)
                # Healthy 행은 DI와 무관(항상 동일)이므로 제목에서 DI 생략
                title = (f"{method} | {rlabel}" if rkey == "healthy"
                         else f"{method} | {rlabel} @ DI={di:.2f}")
                _style_spec(ax, title, method)

        note = ("green dotted = Structure-A modal positions "
                + ("(Hz)" if CFG.FS_B == CFG.FS_A else "(A norm-mode mapped to axis)"))
        fig.suptitle(f"[{tag()}] {method} Spectrum  (rows: Healthy / Manual / DeepONet)   |   {note}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(CFG.OUT_DIR, f"Fig1_{method}_Spectrum_Overlay.png")
        plt.savefig(out, dpi=250); plt.close("all")
        print(f"  saved: {os.path.basename(out)}")


def plot_a_fingerprint(ds):
    """[Fig 2] A 지문 비율 vs DI (정규화 주파수 기준)."""
    def ratio(data):
        f, m = compute_fft(data, fs=None)
        return a_fingerprint_ratio(f, m)

    base = ratio(ds["healthy"]) if ds["healthy"] is not None else np.nan
    man_di = sorted(ds["manual"].keys()); don_di = sorted(ds["don"].keys())
    man_r = [ratio(ds["manual"][d]) for d in man_di]
    don_r = [ratio(ds["don"][d])    for d in don_di]
    real_r = [ratio(ds["real"][c]) for c in sorted(ds["real"].keys())] if ds["real"] else []

    plt.figure(figsize=(11, 6))
    if not np.isnan(base):
        plt.axhline(base, color="k", ls="--", lw=1.5, label=f"B Healthy baseline ({base:.4f})")
    plt.plot([d * 100 for d in man_di], man_r, "o-", color="#1f77b4", lw=2, ms=7, label="Manual")
    plt.plot([d * 100 for d in don_di], don_r, "s-", color="#d62728", lw=1.6, ms=4, label="DeepONet")
    if real_r:
        plt.axhline(np.mean(real_r), color="green", ls="-.", lw=1.5,
                    label=f"Real B damage mean ({np.mean(real_r):.4f})")
    plt.title(f"[{tag()}] Structure-A Fingerprint Energy Ratio vs Damage\n"
              f"(A normalized modes {[round(x,3) for x in CFG.A_MODES_NORM]} cycles/sample)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Damage level (DI x 100)")
    plt.ylabel("Energy fraction in A modal bands")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout()
    out = os.path.join(CFG.OUT_DIR, "Fig2_A_Fingerprint_Ratio.png")
    plt.savefig(out, dpi=250); plt.close("all")
    print(f"  saved: {os.path.basename(out)}")
    return base, man_di, man_r, don_di, don_r, real_r


def plot_natural_freq_vs_di(ds):
    """[Fig 3] B 고유진동수 이동 vs DI."""
    if ds["healthy"] is None:
        return
    f_h, m_h = compute_psd(ds["healthy"], CFG.FS_B)
    b_modes = top_peaks(f_h, m_h, n_peaks=4, fmin=f_h[1])
    b_modes = b_modes[~np.isnan(b_modes)]
    if len(b_modes) == 0:
        print("  [Fig3] B 모드 자동검출 실패 — 건너뜀"); return
    search_half = FMAX() * 0.03
    man_di = sorted(ds["manual"].keys()); don_di = sorted(ds["don"].keys())

    fig, axes = plt.subplots(1, len(b_modes), figsize=(5 * len(b_modes), 4.5), squeeze=False)
    axes = axes[0]
    for ax, bm in zip(axes, b_modes):
        man_t = [track_nearest_peak(*compute_psd(ds["manual"][d], CFG.FS_B), bm, search_half) for d in man_di]
        don_t = [track_nearest_peak(*compute_psd(ds["don"][d], CFG.FS_B), bm, search_half) for d in don_di]
        ax.axhline(bm, color="k", ls="--", lw=1.2, label="Healthy mode")
        ax.plot([d * 100 for d in man_di], man_t, "o-", color="#1f77b4", label="Manual")
        ax.plot([d * 100 for d in don_di], don_t, "s-", color="#d62728", ms=3, label="DeepONet")
        ax.set_title(f"B mode @ {bm:.4f}", fontweight="bold")
        ax.set_xlabel("Damage (DI x 100)"); ax.set_ylabel(FLABEL())
        ax.grid(True, alpha=0.4); ax.legend(fontsize=8)
    fig.suptitle(f"[{tag()}] Structure-B Natural Frequency Shift vs Damage",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(CFG.OUT_DIR, "Fig3_NaturalFreq_vs_DI.png")
    plt.savefig(out, dpi=250); plt.close("all")
    print(f"  saved: {os.path.basename(out)}")


def plot_timedomain_trends(ds):
    """[Fig 4] RMS-p95, Kurtosis-p95, WPE 거리 vs DI."""
    man_di = sorted(ds["manual"].keys()); don_di = sorted(ds["don"].keys())
    wpe_h = wpe_distribution(ds["healthy"]) if ds["healthy"] is not None else None

    def wpe_dist(data):
        if wpe_h is None:
            return np.nan
        w = wpe_distribution(data)
        return np.nan if w is None else float(np.linalg.norm(w - wpe_h, ord=1))

    base = {"RMS (95th pct)": rms_p95(ds["healthy"]) if ds["healthy"] is not None else np.nan,
            "Kurtosis (95th pct)": kurt_p95(ds["healthy"]) if ds["healthy"] is not None else np.nan,
            "WPE dist. to Healthy (L1)": 0.0}
    fns = {"RMS (95th pct)": rms_p95, "Kurtosis (95th pct)": kurt_p95,
           "WPE dist. to Healthy (L1)": wpe_dist}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, fns):
        fn = fns[name]
        man_v = [fn(ds["manual"][d]) for d in man_di]
        don_v = [fn(ds["don"][d])    for d in don_di]
        if not np.isnan(base[name]):
            ax.axhline(base[name], color="k", ls="--", lw=1.2, label="Healthy")
        ax.plot([d * 100 for d in man_di], man_v, "o-", color="#1f77b4", lw=2, ms=6, label="Manual")
        ax.plot([d * 100 for d in don_di], don_v, "s-", color="#d62728", lw=1.6, ms=3, label="DeepONet")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Damage (DI x 100)"); ax.grid(True, alpha=0.4); ax.legend(fontsize=9)
    fig.suptitle(f"[{tag()}] Time-domain / Energy Feature Trends vs Damage",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(CFG.OUT_DIR, "Fig4_TimeDomain_Trends.png")
    plt.savefig(out, dpi=250); plt.close("all")
    print(f"  saved: {os.path.basename(out)}")


def plot_wpe_bars(ds):
    """[Fig 5] 최고 DI에서 웨이블릿 패킷 에너지 분포."""
    if not HAS_PYWT or ds["healthy"] is None:
        return
    wpe_h = wpe_distribution(ds["healthy"])
    if wpe_h is None:
        return
    x = np.arange(len(wpe_h))
    plt.figure(figsize=(13, 5))
    plt.bar(x - 0.25, wpe_h, width=0.25, color="k", alpha=0.6, label="B Healthy")
    if ds["manual"]:
        dm = max(ds["manual"].keys()); wm = wpe_distribution(ds["manual"][dm])
        if wm is not None:
            plt.bar(x, wm, width=0.25, color="#1f77b4", alpha=0.8, label=f"Manual DI={dm:.2f}")
    if ds["don"]:
        dd = max(ds["don"].keys()); wd = wpe_distribution(ds["don"][dd])
        if wd is not None:
            plt.bar(x + 0.25, wd, width=0.25, color="#d62728", alpha=0.8, label=f"DeepONet DI={dd:.2f}")
    plt.title(f"[{tag()}] Wavelet Packet Energy Distribution (db4, level 4)", fontweight="bold")
    plt.xlabel("WPT band index (low -> high frequency)")
    plt.ylabel("Normalized energy")
    plt.grid(True, axis="y", alpha=0.3); plt.legend()
    plt.tight_layout()
    out = os.path.join(CFG.OUT_DIR, "Fig5_WPE_Distribution.png")
    plt.savefig(out, dpi=250); plt.close("all")
    print(f"  saved: {os.path.basename(out)}")


# ============================================================================
# 6. PER-RUN SUMMARY  +  판정
# ============================================================================
def write_summary(ds, fp_result):
    base, man_di, man_r, don_di, don_r, real_r = fp_result
    lines = ["metric,dataset,DI,value"]
    for d, v in zip(man_di, man_r):
        lines.append(f"A_fingerprint_ratio,manual,{d:.2f},{v:.6f}")
    for d, v in zip(don_di, don_r):
        lines.append(f"A_fingerprint_ratio,deeponet,{d:.2f},{v:.6f}")
    if not np.isnan(base):
        lines.append(f"A_fingerprint_ratio,healthy,,{base:.6f}")
    if real_r:
        lines.append(f"A_fingerprint_ratio,real_mean,,{np.mean(real_r):.6f}")
    out = os.path.join(CFG.OUT_DIR, "summary_metrics.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {os.path.basename(out)}")

    verdict = "unclear"
    mp = np.nanmax(man_r) if man_r else np.nan
    dp = np.nanmax(don_r) if don_r else np.nan
    if not np.isnan(base) and man_r and don_r:
        print(f"  [해석] A-지문 | Healthy={base:.4f} | Manual(max)={mp:.4f} | DeepONet(max)={dp:.4f}"
              + (f" | Real={np.mean(real_r):.4f}" if real_r else ""))
        if mp > base * 1.5 and mp > dp * 1.2:
            verdict = "contaminated"
            print("         → Manual A-지문 오염 ↑ / DeepONet 억제 → '물리 보존' 뒷받침")
        else:
            verdict = "similar"
            print("         → 두 방식 A-지문 차이 뚜렷하지 않음 (Fig1 육안 확인 권장)")
    return {"healthy": base, "manual_max": mp, "deeponet_max": dp,
            "real_mean": (np.mean(real_r) if real_r else np.nan), "verdict": verdict}


# ============================================================================
# 7. 한 케이스 처리
# ============================================================================
def run_one(healthy, real):
    """현재 CFG(state/case)에 대해 5개 그림 + CSV 생성. 상태 문자열 반환."""
    print("\n" + "=" * 70)
    print(f"  Physical Comparison | {tag()} | "
          f"FS_B={'auto(normalized)' if not CFG.FS_B else CFG.FS_B}")
    print("=" * 70)

    base_dir = os.path.join(CFG.ROOT, f"state_{CFG.STATE}", CFG.CASE_NAME)
    if not os.path.isdir(base_dir):
        print(f"  ⏭️  폴더 없음 → 스킵: {base_dir}")
        return "skip(no folder)", None
    ds = load_run_synth(healthy, real)
    if not ds["manual"] and not ds["don"]:
        print("  ⏭️  합성 데이터 없음 → 스킵 (MANUAL_DIR/DON_DIR 확인)")
        return "skip(no synth)", None

    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    print("[1/5] 스펙트럼 (Manual/DON 분리) ...");  plot_spectra_overlay(ds)
    print("[2/5] A 지문 비율 ...");                fp = plot_a_fingerprint(ds)
    print("[3/5] 고유진동수 이동 ...");             plot_natural_freq_vs_di(ds)
    print("[4/5] 시간영역/에너지 추세 ...");         plot_timedomain_trends(ds)
    print("[5/5] 웨이블릿 에너지 분포 ...");         plot_wpe_bars(ds)
    metrics = write_summary(ds, fp)
    print(f"  [OK] 저장: {CFG.OUT_DIR}")
    return "ok", metrics


# ============================================================================
# 8. MAIN  --  state × case 자동 순회
# ============================================================================
def main():
    healthy, real = load_shared()
    if healthy is None:
        print("\n[X] B 건강데이터 로드 실패 — DIR_B_RAW/FILE_B_H 확인.")
        return

    cases = [f"Case A{i}" for i in range(CFG.CASE_START, CFG.CASE_END + 1)]
    print(f"\n▶️ 자동 순회 대기열: states={CFG.STATES} × cases={cases}")

    master = ["state,case,status,healthy,manual_max,deeponet_max,real_mean,verdict"]
    results = []
    for state in CFG.STATES:
        for case in cases:
            set_run(state, case)
            try:
                status, metrics = run_one(healthy, real)
            except Exception as e:
                import traceback
                print(f"  ❌ 오류 → 계속 진행: {e}")
                traceback.print_exc()
                status, metrics = f"error({e})", None
            results.append((state, case, status))
            if metrics:
                master.append(f"{state},{case},{status},{metrics['healthy']:.6f},"
                              f"{metrics['manual_max']:.6f},{metrics['deeponet_max']:.6f},"
                              f"{metrics['real_mean']:.6f},{metrics['verdict']}")
            else:
                master.append(f"{state},{case},{status},,,,,")

    # 마스터 요약 CSV
    master_path = os.path.join(CFG.ROOT, "Physical_Compare_MASTER_summary.csv")
    try:
        with open(master_path, "w", encoding="utf-8") as f:
            f.write("\n".join(master))
        print(f"\n📄 마스터 요약: {master_path}")
    except Exception as e:
        print(f"\n[!] 마스터 요약 저장 실패: {e}")

    # 콘솔 최종 표
    print("\n" + "=" * 70)
    print("  전체 결과 요약")
    print("=" * 70)
    for state, case, status in results:
        print(f"  state_{state} | {case:<10} | {status}")
    if not HAS_PYWT:
        print("\n   (참고) pywt 미설치 → 웨이블릿 항목 건너뜀. pip install PyWavelets")
    print("\n🏁 모든 state × case 순회 완료.")


if __name__ == "__main__":
    main()