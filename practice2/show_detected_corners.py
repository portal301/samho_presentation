#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
체커보드 코너 검출 단일 데모
===========================
한 번 실행으로 다음을 시연합니다.

1) 체커보드 이미지 로드 및 리사이즈 (1280x720)
2) 그레이 변환 → 체커보드 코너 검출 → 서브픽셀 정밀화
3) 검출된 코너를 원본 위에 오버레이해 한 창으로 표시

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
"""

import os
import sys
import cv2
import numpy as np

# ----------------------------------------------
# 1) 데이터 경로 구성 (이 스크립트 기준 상대 경로)
# ----------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "data", "checkerboard_sample.jpg")

print("=" * 60)
print("체커보드 코너 검출 데모 시작")
print("=" * 60)
print(f"이미지 경로: {img_path}")

# ----------------------------------------------
# 2) 이미지 로드 및 전처리
# ----------------------------------------------
img = cv2.imread(img_path)
if img is None:
    print("오류: 이미지를 로드할 수 없습니다. 경로를 확인하세요.")
    sys.exit(1)

img = cv2.resize(img, (1280, 720))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------------------------------
# 3) 체커보드 코너 검출 및 정밀화
# ----------------------------------------------
CHECKERBOARD = (9, 6)  # (가로 내부 코너 수, 세로 내부 코너 수)
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if not ret:
    print("체커보드 코너를 찾지 못했습니다. 이미지 상태를 확인하세요.")
    sys.exit(0)

# 서브픽셀 정밀화
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# 코너 시각화
img_vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners_refined, ret)

# ----------------------------------------------
# 4) 결과 표시
# ----------------------------------------------
cv2.imshow('Detected Corners', img_vis)
print("창에서 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n데모 완료!")

