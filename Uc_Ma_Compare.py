import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

# 시드 고정 및 샘플 수
np.random.seed(45)
n_samples = 800

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# -----------------------------------------------------------------
# [그래프 1] 완벽한 45도 회전 마름모꼴 정사각형 데이터셋 생성
# -----------------------------------------------------------------
# 1) 먼저 축과 평행한 가로세로가 똑바른 정사각형 데이터 생성 (-3부터 +3 범위)
X_square = np.random.uniform(-3, 3, (n_samples, 2))

# 2) 선형대수학의 45도 회전 변환 행렬 (Rotation Matrix) 정의
theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 3) 정사각형 데이터를 45도 회전시켜 완벽한 마름모꼴 정사각형으로 변환 (선형 변환 적용)
X_diamond = X_square @ R.T

# 대각선 방향(단축 방향)으로 살짝 탈출한 아웃라이어 설정 (마름모 변 바로 바깥)
# 이 점은 중심에서의 직선거리가 마름모의 꼭짓점 거리보다 짧음
outlier_diamond = np.array([[-0.2, 3.2]])

ax1.scatter(X_diamond[:, 0], X_diamond[:, 1], color='royalblue', alpha=0.4, label='Normal Data')
ax1.scatter(outlier_diamond[:, 0], outlier_diamond[:, 1], color='crimson', edgecolors='black', s=150, zorder=5, label='Outlier')

# 유클리디언 경계: 마름모의 가장 먼 꼭짓점(대각선 끝)을 다 덮어야 하므로 거대한 원이 됨
# 회전된 마름모의 꼭짓점 거리는 약 4.24이므로 반지름을 4.3으로 설정
circle1 = plt.Circle((0, 0), 4.3, color='gray', linestyle='--', fill=False, linewidth=2, label='Euclidean Boundary (Circle)')
ax1.add_patch(circle1)

# 마할라노비스 경계: 데이터의 45도 회전된 상관관계와 분산 구조를 반영한 타원
# 균일 분포 기반의 마름모이지만 공분산은 45도 방향의 퍼짐을 인지하므로 이에 맞는 타원을 형성함
ellipse1 = Ellipse((0, 0), width=4.24*2, height=1.73*2, angle=45, edgecolor='forestgreen', linestyle='-', fill=False, linewidth=2.5, label='Mahalanobis Boundary (Ellipse)')
ax1.add_patch(ellipse1)

ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlabel('Feature X')
ax1.set_ylabel('Feature Y')
ax1.set_title('1. 45-Degree Rotated Square (Diamond)\n(Correlation exists, Outlier hides inside Euclidean Circle)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# -----------------------------------------------------------------
# [그래프 2] 동일한 유클리디언 거리를 가졌으나 마할라노비스 거리에선 정상 vs 이상치인 경우
# -----------------------------------------------------------------
# 가로로 길쭉한 타원형 정상 데이터 생성 (X축 분산 16, Y축 분산 1)
mean2 = [0, 0]
cov2 = [[16, 0], 
        [0, 1]]
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# 중심 (0,0)으로부터 '유클리디언 직선 거리'가 약 5.1로 거의 동일한 두 개의 점 설정
point_A = np.array([[5.1, 0.0]]) # X축 선상 (정상 범위 내)
point_B = np.array([[0.0, 5.1]]) # Y축 선상 (완전한 이상치)

ax2.scatter(X2[:, 0], X2[:, 1], color='royalblue', alpha=0.4, label='Normal Baseline')
ax2.scatter(point_A[:, 0], point_A[:, 1], color='orange', edgecolors='black', s=150, zorder=5, label='Point A (Inside Ellipse -> Normal)')
ax2.scatter(point_B[:, 0], point_B[:, 1], color='crimson', edgecolors='black', s=150, zorder=5, label='Point B (Outside Ellipse -> Outlier)')

# 유클리디언 경계: 반지름 5.3인 원을 그려 두 점을 모두 포함시킴
circle2 = plt.Circle((0, 0), 5.3, color='gray', linestyle='--', fill=False, linewidth=2, label='Euclidean Boundary (r=5.3)')
ax2.add_patch(circle2)

# 마할라노비스 경계: 데이터의 실제 분산을 반영한 타원
ellipse2 = Ellipse((0, 0), width=8*2, height=2*2, edgecolor='forestgreen', linestyle='-', fill=False, linewidth=2.5, label='Mahalanobis Boundary (Ellipse)')
ax2.add_patch(ellipse2)

ax2.set_xlim(-12, 12)
ax2.set_ylim(-12, 12)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel('Feature X (High Variance)')
ax2.set_ylabel('Feature Y (Low Variance)')
ax2.set_title('2. Same Euclidean Distance\n(Point A is Normal vs Point B is Outlier)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()