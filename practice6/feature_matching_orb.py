#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ORB 특징점 매칭 데모 스크립트
=============================
한 번 실행으로 다음을 시연합니다.

1) 입력 이미지 로드 (`data/sample.jpg` 우선, 없으면 `practice3/data/candies.png` 대체)
2) 이미지에 기하 변환(스케일, 회전, 시프트)을 가해 쌍을 생성
3) ORB 특징 검출 및 기술자 매칭(BFMatcher, Hamming)
4) Lowe's ratio test로 좋은 매치 선별
5) 원본/변환 쌍과 매칭 결과를 한 창에 표시

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
- 기존 코드는 수정하지 않으며, 이 파일만 추가 실행하면 됩니다.
"""

import os
import sys
import cv2
import numpy as np

# ------------------------------------------------------------
# 1) 입력 이미지 경로 설정 및 로드
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "data", "candies.png")

print("=" * 60)
print("ORB 특징점 매칭 데모 시작")
print("=" * 60)
print(f"이미지 경로: {img_path}")

img = cv2.imread(img_path)
if img is None:
    print("오류: 이미지를 로드할 수 없습니다. 경로를 확인하세요.")
    sys.exit(1)

# 컬러 → 그레이 변환 (ORB는 그레이 입력 권장)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# 2) 기하 변환으로 두 번째 이미지 생성 (스케일+회전+시프트)
# ------------------------------------------------------------
scale = 0.9    # 스케일 축소
angle = 25.0   # 각도(도)
tx, ty = 30, 20  # x/y 시프트(픽셀)

h, w = gray.shape[:2]
center = (w // 2, h // 2)
M_rot = cv2.getRotationMatrix2D(center, angle, scale)  # 회전+스케일
M_rot[:, 2] += [tx, ty]  # 시프트 추가

warped = cv2.warpAffine(gray, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ------------------------------------------------------------
# 3) ORB 특징 검출 및 기술자 계산
# ------------------------------------------------------------
num_features = 1000  # 최대 키포인트 개수
orb = cv2.ORB_create(nfeatures=num_features)

keypoints1, descriptors1 = orb.detectAndCompute(gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(warped, None)

print(f"키포인트 수 - 원본: {len(keypoints1)}, 변환: {len(keypoints2)}")

if descriptors1 is None or descriptors2 is None:
    print("오류: 디스크립터 계산에 실패했습니다.")
    sys.exit(1)

# ------------------------------------------------------------
# 4) 매칭 (BFMatcher + Hamming, kNN=2) 및 Lowe ratio test
# ------------------------------------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

ratio_thresh = 0.75
good_matches = []
for m, n in raw_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

print(f"좋은 매치 수 (ratio<{ratio_thresh}): {len(good_matches)}")

# ------------------------------------------------------------
# 5) 매칭 시각화 (일부 상위 매치만 그리기)
# ------------------------------------------------------------
good_matches = sorted(good_matches, key=lambda x: x.distance)
draw_count = min(80, len(good_matches))
matched_vis = cv2.drawMatches(
    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), keypoints1,
    cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), keypoints2,
    good_matches[:draw_count], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 상단에 텍스트 오버레이: 파라미터/매치 수
info_text = f"ORB nfeatures={num_features} | good_matches={len(good_matches)} | scale={scale}, angle={angle}, shift=({tx},{ty})"
cv2.putText(matched_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

# ------------------------------------------------------------
# 6) 결과 표시 (한 번에 한 창, 아무 키나 누르면 종료)
# ------------------------------------------------------------
matched_vis = cv2.resize(matched_vis, (1280, 720), fx=0.5, fy=0.5)
cv2.imshow("ORB Matching Result", matched_vis)
print("창에서 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n데모 완료!")

