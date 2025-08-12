#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
양방향 필터(Bilateral Filter) 단일 데모
===================================
z/x로 원본/필터 결과를 토글하여 좌우 결합된 두 이미지를 비교합니다.
"""

import os
import sys
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(script_dir, "data", "noise_moon_1.png")
img2_path = os.path.join(script_dir, "data", "noise_moon_2.png")

print("=" * 60)
print("양방향 필터 데모 시작 - z: 원본, x: 필터, ESC: 종료")
print("=" * 60)

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
if img1 is None or img2 is None:
    print("오류: 이미지 파일을 찾을 수 없습니다.")
    sys.exit(1)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
max_h = max(h1, h2)
if h1 != max_h:
    img1 = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
if h2 != max_h:
    img2 = cv2.resize(img2, (int(w2 * max_h / h2), max_h))

# d=9, sigmaColor=75, sigmaSpace=75 (filter_comparison.py와 동일)
img1_f = cv2.bilateralFilter(img1, 9, 75, 75)
img2_f = cv2.bilateralFilter(img2, 9, 75, 75)

window_name = "Bilateral Filter Demo (z: original, x: filtered, ESC: quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 600)

show_filtered = False

while True:
    if show_filtered:
        left = img1_f.copy()
        right = img2_f.copy()
        label = "bilateral filter (d=9, sc=75, ss=75)"
    else:
        left = img1.copy()
        right = img2.copy()
        label = "original image"

    combined = np.hstack([left, right])
    cv2.putText(combined, "Left: noise_moon_1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "Right: noise_moon_2", (left.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"View: {label} | z: original, x: filtered, ESC: quit", (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(window_name, combined)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break
    elif key == ord('z'):
        show_filtered = False
        print("원본 보기")
    elif key == ord('x'):
        show_filtered = True
        print("양방향 필터 적용 보기")

cv2.destroyAllWindows()
print("\n데모 종료")

