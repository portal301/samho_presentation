#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
버니 포인트 클라우드 Z축 이동/중앙정렬 스크립트
===========================================
한 번 실행으로 `data/bunny.xyz`를 읽어 Z축을 이동한 새 파일을 생성합니다.

기본 동작:
- Z축 중앙정렬: z의 평균값을 0으로 맞추도록 전체를 이동 → `data/bunny_centered_z.xyz` 저장

옵션:
- 명령행 인자에 수치(예: -0.05)를 주면, z축을 해당 값만큼 이동 → `data/bunny_shifted_z.xyz` 저장
  - 음수: 아래로 내림 (z 감소)
  - 양수: 위로 올림 (z 증가)

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
"""

import os
import sys
import numpy as np

# ----------------------------------------------
# 1) 파일 경로 설정 (이 스크립트 기준)
# ----------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(script_dir, "data", "bunny_origin.xyz")

print("=" * 60)
print("버니 Z축 이동/중앙정렬 스크립트")
print("=" * 60)
print(f"입력 파일: {in_path}")

if not os.path.exists(in_path):
    print("오류: 입력 파일을 찾을 수 없습니다.")
    sys.exit(1)

# ----------------------------------------------
# 2) XYZ 로드
# ----------------------------------------------
points = np.loadtxt(in_path, delimiter=' ', dtype=np.float64)
if points.ndim != 2 or points.shape[1] < 3:
    print("오류: XYZ 형식이 아닙니다.")
    sys.exit(1)

z_vals = points[:, 2]
z_min, z_max, z_mean = float(np.min(z_vals)), float(np.max(z_vals)), float(np.mean(z_vals))
print(f"z 통계: min={z_min:.6f}, max={z_max:.6f}, mean={z_mean:.6f}")

# ----------------------------------------------
# 3) 이동량 결정: 인자 있으면 delta_z, 없으면 중앙정렬(= -mean)
# ----------------------------------------------
use_offset = False
delta_z = 0.0

if len(sys.argv) > 1:
    try:
        delta_z = float(sys.argv[1])
        use_offset = True
        print(f"요청: z축을 {delta_z:+.6f} 만큼 이동")
    except Exception:
        print("경고: 인자를 실수로 해석할 수 없습니다. 중앙정렬로 진행합니다.")

if use_offset:
    shift = delta_z
    out_path = os.path.join(script_dir, "data", "bunny_shifted_z.xyz")
else:
    shift = -z_mean  # 평균을 0으로 이동
    out_path = os.path.join(script_dir, "data", "bunny_centered_z.xyz")

print(f"적용 이동량 shift_z = {shift:+.6f}")

# ----------------------------------------------
# 4) 좌표 적용 및 저장
# ----------------------------------------------
points_out = points.copy()
points_out[:, 1] = points[:, 1] - shift

os.makedirs(os.path.join(script_dir, "data"), exist_ok=True)
np.savetxt(out_path, points_out, fmt='%.6f', delimiter=' ')

new_z = points_out[:, 1]
print(f"출력 파일: {out_path}")
print("새 z 통계: min={:.6f}, max={:.6f}, mean={:.6f}".format(float(np.min(new_z)), float(np.max(new_z)), float(np.mean(new_z))))
print("\n완료!")

