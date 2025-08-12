#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
평균 필터(Mean Filter) 단일 데모
===============================
한 번 실행으로 다음을 시연합니다.

1) 두 개의 노이즈 이미지 로드 (`practice4/data/noise_moon_1.png`, `practice4/data/noise_moon_2.png`)
2) 이미지들을 높이에 맞게 정렬해 좌우 결합
3) 평균 필터(5x5)를 적용한 결과와 원본을 z/x 키로 토글 표시
   - z: 원본 보기
   - x: 필터 적용 결과 보기
   - ESC: 종료

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
"""

import os
import sys
import cv2
import numpy as np

# ----------------------------------------------
# 1) 데이터 경로 구성 (이 스크립트 기준)
# ----------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(script_dir, "data", "noise_moon_1.png")
img2_path = os.path.join(script_dir, "data", "noise_moon_2.png")

print("=" * 60)
print("평균 필터(Mean) 데모 시작 - z: 원본, x: 필터, ESC: 종료")
print("=" * 60)
print(f"이미지1: {img1_path}")
print(f"이미지2: {img2_path}")

# ----------------------------------------------
# 2) 이미지 로드
# ----------------------------------------------
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("오류: 이미지 파일을 찾을 수 없습니다.")
    sys.exit(1)

# ----------------------------------------------
# 3) 사이즈 맞추기 (더 큰 높이에 맞춰 리사이즈)
# ----------------------------------------------
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
max_h = max(h1, h2)

if h1 != max_h:
    img1 = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
if h2 != max_h:
    img2 = cv2.resize(img2, (int(w2 * max_h / h2), max_h))

# ----------------------------------------------
# 4) 평균 필터 적용 (5x5)
# ----------------------------------------------
img1_f = cv2.blur(img1, (5, 5))
img2_f = cv2.blur(img2, (5, 5))

# ----------------------------------------------
# 5) 표시 텍스트 및 윈도우 설정
# ----------------------------------------------
window_name = "Mean Filter Demo (z: original, x: filtered, ESC: quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 600)

show_filtered = False  # False=원본, True=필터

while True:
    if show_filtered:
        left = img1_f.copy()
        right = img2_f.copy()
        filter_label = "mean filter (5x5)"
    else:
        left = img1.copy()
        right = img2.copy()
        filter_label = "original image"

    combined = np.hstack([left, right])

    cv2.putText(combined, "Left: noise_moon_1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "Right: noise_moon_2", (left.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"View: {filter_label} | z: original, x: filtered, ESC: quit", (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(window_name, combined)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('z'):
        show_filtered = False
        print("원본 보기")
    elif key == ord('x'):
        show_filtered = True
        print("평균 필터 적용 보기")

cv2.destroyAllWindows()
print("\n데모 종료")

