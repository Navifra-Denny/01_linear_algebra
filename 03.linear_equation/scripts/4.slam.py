#!/usr/bin/env python3

import numpy as np
import pandas as pd

# 상태 변수: x = [x1, x2, l1]
A = np.array([
    [-1, 1, 0],  # x2 - x1 = u
    [-1, 0, 1],  # l1 - x1 = z1
    [0, -1, 1]   # l1 - x2 = z2
])

b = np.array([
    1.0,   # 이동 거리 u
    2.0,   # 관측 z1
    1.0    # 관측 z2
])

# 최소자승 해 계산
x_est = np.linalg.lstsq(A, b, rcond=None)[0]

# 결과 정리 및 출력
labels = ['x1 (초기 위치)', 'x2 (이동 후 위치)', 'l1 (랜드마크 위치)']
df = pd.DataFrame({'변수': labels, '추정값': np.round(x_est, 4)})

# 표 출력
print(df.to_string(index=False))
