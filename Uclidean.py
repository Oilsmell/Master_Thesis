import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

# 시드 고정 및 샘플 수
np.random.seed(42)
n_samples = 800

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# -----------------------------------------------------------------
# 1. 원형 분포 (Circular Distribution)
# -----------------------------------------------------------------
# X와 Y의 분산이 같고 상관관계가 없는 상태 (독립)
mean1 = [0, 0]
cov1 = [[4, 0], 
        [0, 4]]
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# 아웃라이어 후보 샘플 (특정 방향으로 튄 값)
outlier1 = np.array([[5, 5]])

axs[0].scatter(X1[:, 0], X1[:, 1], color='royalblue', alpha=0.4, label='Normal Data')
axs[0].scatter(outlier1[:, 0], outlier1[:, 1], color='crimson', edgecolors='black', s=120, zorder=5, label='Outlier')

# 원형일 때 임계점: 유클리디언과 마할라노비스가 완전히 일치 (똑같은 원)
th_circle = plt.Circle((0, 0), 5.5, color='gray', linestyle='--', fill=False, linewidth=2, label='Euclidean Boundary')
th_mahal  = plt.Circle((0, 0), 5.5, color='forestgreen', linestyle='-', fill=False, linewidth=2, label='Mahalanobis Boundary')
axs[0].add_patch(th_circle)
axs[0].add_patch(th_mahal)

axs[0].set_xlim(-8, 8)
axs[0].set_ylim(-8, 8)
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_title('1. Circular Shape\n(Euclidean == Mahalanobis)')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper left')

# -----------------------------------------------------------------
# 2. 바른 정사각형 분포 (Axis-Aligned Square/Rectangular)
# -----------------------------------------------------------------
# 균일 분포(Uniform)로 사각형 모양 생성 (각 축은 독립적이나 경계가 사각형)
X2_x = np.random.uniform(-4, 4, n_samples)
X2_y = np.random.uniform(-4, 4, n_samples)
X2 = np.column_stack((X2_x, X2_y))

# 아웃라이어 후보 샘플 (모서리 근처에 애매하게 튄 값)
outlier2 = np.array([[4.2, 4.2]])

axs[1].scatter(X2[:, 0], X2[:, 1], color='royalblue', alpha=0.4, label='Normal Data')
axs[1].scatter(outlier2[:, 0], outlier2[:, 1], color='crimson', edgecolors='black', s=120, zorder=5, label='Outlier')

# 유클리디언 임계점: 모서리까지 다 덮으려다 보니 반지름이 아주 큰 원이 됨
th_circle2 = plt.Circle((0, 0), 5.65, color='gray', linestyle='--', fill=False, linewidth=2, label='Euclidean Boundary')
# 마할라노비스 임계점: 데이터의 전체적인 퍼짐(분산)이 X, Y 동일하므로 원형으로 인지
th_mahal2  = plt.Circle((0, 0), 4.5, color='forestgreen', linestyle='-', fill=False, linewidth=2, label='Mahalanobis Boundary')
axs[1].add_patch(th_circle2)
axs[1].add_patch(th_mahal2)

axs[1].set_xlim(-8, 8)
axs[1].set_ylim(-8, 8)
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_title('2. Square Shape\n(Euclidean overestimates corner)')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper left')

# -----------------------------------------------------------------
# 3. 비스듬한 마름모/타원형 분포 (Rotated Diamond / Elongated)
# -----------------------------------------------------------------
# 강한 상관관계를 가진 공분산 행렬 설정
mean3 = [0, 0]
cov3 = [[5, 4.2], 
        [4.2, 5]]  # X가 커질 때 Y도 커지는 비스듬한 마름모꼴 타원 생성
X3 = np.random.multivariate_normal(mean3, cov3, n_samples)

# 아웃라이어 후보 샘플 (평소 진동 축과 어긋난 방향으로 튄 값)
outlier3 = np.array([[-3.5, 3.5]])

axs[2].scatter(X3[:, 0], X3[:, 1], color='royalblue', alpha=0.4, label='Normal Data')
axs[2].scatter(outlier3[:, 0], outlier3[:, 1], color='crimson', edgecolors='black', s=120, zorder=5, label='Outlier')

# 유클리디언 임계점: 가장 멀리 뻗은 장축을 덮기 위해 거대한 원을 형성
th_circle3 = plt.Circle((0, 0), 6.5, color='gray', linestyle='--', fill=False, linewidth=2, label='Euclidean Boundary')
# 마할라노비스 임계점: 데이터의 상관관계(누운 방향)를 반영한 영리한 타원 형성
th_mahal3 = Ellipse((0, 0), width=6.2*2, height=1.3*2, angle=45, edgecolor='forestgreen', linestyle='-', fill=False, linewidth=2.5, label='Mahalanobis Boundary')
axs[2].add_patch(th_circle3)
axs[2].add_patch(th_mahal3)

axs[2].set_xlim(-8, 8)
axs[2].set_ylim(-8, 8)
axs[2].set_aspect('equal', adjustable='box')
axs[2].set_title('3. Rotated Diamond Shape\n(Mahalanobis captures correlation)')
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper left')

plt.tight_layout()
plt.show()