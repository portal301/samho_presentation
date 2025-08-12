#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
리프로젝션 포인트 시각화 단일 데모
===============================
한 번 실행으로 다음을 시연합니다.

1) 체커보드 이미지 로드 및 리사이즈 (1280x720)
2) 체커보드 코너 검출/정밀화 → 카메라 캘리브레이션
3) 체커보드 3D 그리드 포인트를 다시 이미지로 투영하여(리프로젝션) 원본 위에 표시

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
"""

import os
import sys
import cv2
import numpy as np

# ----------------------------------------------
# 1) 데이터 경로 구성
# ----------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "data", "checkerboard_sample.jpg")

print("=" * 60)
print("리프로젝션 포인트 시각화 데모 시작")
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
# 3) 코너 검출/정밀화 및 카메라 캘리브레이션
# ----------------------------------------------
CHECKERBOARD = (9, 6)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
if not ret:
    print("체커보드 코너를 찾지 못했습니다. 이미지 상태를 확인하세요.")
    sys.exit(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

objpoints = [objp]
imgpoints = [corners_refined]

ret_cam, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if not ret_cam:
    print("캘리브레이션에 실패했습니다.")
    sys.exit(1)

# ----------------------------------------------
# 4) 리프로젝션 포인트 계산 및 시각화
# ----------------------------------------------
img_reproj = img.copy()
imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
for p in imgpoints2:
    cv2.circle(img_reproj, tuple(p.ravel().astype(int)), 5, (0, 255, 0), -1)

cv2.imshow('Reprojected Points', img_reproj)
print("창에서 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n데모 완료!")

