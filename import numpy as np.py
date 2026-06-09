import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 1. 시드 고정 및 데이터 생성 (정상 베이스라인)
np.random.seed(42)
n_samples = 1000

# X축 방향으로는 변동이 크고(분산 25 -> 표준편차 5), Y축 방향으로는 변동이 작은(분산 1 -> 표준편차 1) 데이터
mean_normal = [0, 0]
cov_normal = [[25, 0], 
              [0, 1]]

X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_samples)

# 2. 손상 데이터 생성 (10% ~ 100% 손상으로 가면서 중심이 이동)
damage_steps = np.linspace(10, 100, 10)  # 10%, 20%, ..., 100%
X_damage = []
for step in damage_steps:
    # 손상이 진행됨에 따라 X축으로 1.5~15, Y축으로 0.4~4.0 만큼 이동
    shift_x = step * 0.15
    shift_y = step * 0.04
    X_damage.append([shift_x, shift_y])
X_damage = np.array(X_damage)

# 3. 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# -----------------------------------------------------------------
# [왼쪽 그래프] 원본 공간 (축 비율 1:1 고정 -> 유클리디언이 진짜 '원'으로 보임)
# -----------------------------------------------------------------
ax1.scatter(X_normal[:, 0], X_normal[:, 1], color='royalblue', alpha=0.4, label='Normal Baseline')
scatter_dmg = ax1.scatter(X_damage[:, 0], X_damage[:, 1], c=damage_steps, cmap='autumn', edgecolors='black', s=100, zorder=5, label='Damage (10% -> 100%)')

# 1) 유클리디언 거리 기준 정상 경계 (축 비율이 1:1이므로 완벽한 원으로 그려짐)
# 정상 데이터의 가장 먼 퍼짐을 커버하기 위해 반지름을 12로 설정
euclidean_radius = 12
circle = plt.Circle((0, 0), euclidean_radius, color='gray', linestyle='--', fill=False, linewidth=2, label='Euclidean Boundary (Circle)')
ax1.add_patch(circle)

# 2) 공분산(마할라노비스) 기준 정상 경계 (데이터의 실제 형태를 반영한 타원)
# X축 반경 = 5 * 2 = 10, Y축 반경 = 1 * 2 = 2 (약 95% 포함)
ellipse = Ellipse((0, 0), width=10*2, height=2*2, edgecolor='forestgreen', linestyle='-', fill=False, linewidth=2.5, label='Covariance Boundary (Ellipse)')
ax1.add_patch(ellipse)

# 축 비율을 1:1로 강제 설정 (X축의 1단위와 Y축의 1단위 길이가 같아짐)
ax1.set_aspect('equal', adjustable='box')

# 모든 바운더리와 손상 데이터가 한눈에 들어오도록 줌아웃(축 범위 조정)
ax1.set_xlim(-18, 18)
ax1.set_ylim(-18, 18)  # Y축 범위를 X축과 맞춰 원형이 왜곡되지 않게 함
ax1.set_xlabel('Feature X (High Variance)')
ax1.set_ylabel('Feature Y (Low Variance)')
ax1.set_title('Original Latent Space:\nTrue Geometric Shapes (1:1 Aspect Ratio)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# -----------------------------------------------------------------
# [오른쪽 그래프] 마할라노비스 변환 후 공간 (정상 데이터가 원형으로 압축됨)
# -----------------------------------------------------------------
cov_inv_sqrt = np.diag([1/5.0, 1/1.0])  # 각 축의 표준편차로 나누어 정규화

X_normal_trans = X_normal @ cov_inv_sqrt
X_damage_trans = X_damage @ cov_inv_sqrt

ax2.scatter(X_normal_trans[:, 0], X_normal_trans[:, 1], color='royalblue', alpha=0.4, label='Normal Baseline (Normalized)')
ax2.scatter(X_damage_trans[:, 0], X_damage_trans[:, 1], c=damage_steps, cmap='autumn', edgecolors='black', s=100, zorder=5, label='Damage (10% -> 100%)')

# 변환된 공간에서는 마할라노비스 경계가 반지름 2.4인 원형이 됨
normalized_circle = plt.Circle((0, 0), 2.4, color='forestgreen', linestyle='-', fill=False, linewidth=2.5, label='Mahalanobis Boundary')
ax2.add_patch(normalized_circle)

ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-5, 6)
ax2.set_ylim(-5, 6)
ax2.set_xlabel('Normalized Feature X')
ax2.set_ylabel('Normalized Feature Y')
ax2.set_title('Mahalanobis Transformed Space:\nAll Directions Standardized')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# 컬러바 추가
cbar = fig.colorbar(scatter_dmg, ax=[ax1, ax2], orientation='horizontal', pad=0.1, shrink=0.5)
cbar.set_label('Damage Severity (%)')

plt.tight_layout()
plt.show()