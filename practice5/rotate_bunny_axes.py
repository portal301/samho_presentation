#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
버니 포인트 클라우드 좌표축 회전 스크립트 (기본 45도)
====================================================
한 번 실행으로 `data/bunny.xyz`를 읽어 지정한 축으로 지정한 각도만큼 회전한
새 파일을 저장합니다. ICP 데모에서 정렬 효과를 직관적으로 보이기 위한 준비용.

기본값
- 축: Z축
- 각도: +45도 (도 단위, 시계 반대 방향)
- 출력: data/bunny_rotated_45deg.xyz

옵션 (명령행 인자)
- 첫 번째 인자: 축 (x|y|z)
- 두 번째 인자: 각도(deg, 실수 가능)
- 예: `python rotate_bunny_axes.py z 30`

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
"""

import os
import sys
import math
import numpy as np

# ------------------------------------------------------------
# 1) 입력/출력 경로
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(script_dir, "data", "bunny.xyz")

# 기본 파라미터
axis = 'z'
angle_deg = 45.0

# 명령행 인자 처리
if len(sys.argv) >= 2:
    axis = sys.argv[1].lower()
if len(sys.argv) >= 3:
    try:
        angle_deg = float(sys.argv[2])
    except Exception:
        print("경고: 각도 파싱 실패. 기본 45도로 진행합니다.")

angle_tag = f"{int(angle_deg)}deg" if abs(angle_deg - int(angle_deg)) < 1e-9 else f"{angle_deg}deg"
out_name = f"bunny_rotated_{angle_tag}.xyz" if axis == 'z' and abs(angle_deg-45.0) < 1e-9 else f"bunny_rotated_{axis}_{angle_tag}.xyz"
out_path = os.path.join(script_dir, "data", out_name)

print("=" * 60)
print("버니 좌표축 회전 스크립트")
print("=" * 60)
print(f"입력 파일: {in_path}")
print(f"축/각도: axis={axis}, angle={angle_deg} deg")

if not os.path.exists(in_path):
    print("오류: 입력 파일을 찾을 수 없습니다.")
    sys.exit(1)

# ------------------------------------------------------------
# 2) 데이터 로드
# ------------------------------------------------------------
points = np.loadtxt(in_path, delimiter=' ', dtype=np.float64)
if points.ndim != 2 or points.shape[1] < 3:
    print("오류: XYZ 형식이 아닙니다.")
    sys.exit(1)

num_points = len(points)
print(f"포인트 수: {num_points}")

# ------------------------------------------------------------
# 3) 회전 행렬 구성
# ------------------------------------------------------------
theta = math.radians(angle_deg)
ct = math.cos(theta)
st = math.sin(theta)

if axis == 'x':
    R = np.array([[1, 0, 0],
                  [0, ct, -st],
                  [0, st,  ct]], dtype=np.float64)
elif axis == 'y':
    R = np.array([[ ct, 0, st],
                  [  0, 1,  0],
                  [-st, 0, ct]], dtype=np.float64)
else:  # 'z' or any other
    R = np.array([[ct, -st, 0],
                  [st,  ct, 0],
                  [ 0,   0, 1]], dtype=np.float64)

# ------------------------------------------------------------
# 4) 회전 적용 및 저장
# ------------------------------------------------------------
pts_xyz = points[:, :3]
rotated = (R @ pts_xyz.T).T

os.makedirs(os.path.join(script_dir, "data"), exist_ok=True)
np.savetxt(out_path, rotated, fmt='%.6f', delimiter=' ')

print(f"출력 파일: {out_path}")
print("완료!")

