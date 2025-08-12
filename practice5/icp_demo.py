#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICP(Point-to-Point) 정렬 데모 스크립트
=====================================
한 번 실행으로 다음을 시연합니다.

1) 노이즈가 있는 버니(`bunny_noisy_extreme.xyz`)와 깨끗한 버니(`bunny.xyz`) 로드
2) 다운샘플링으로 가속화 및 시각화 색상 지정
3) Point-to-Point ICP 정렬 수행 (초기 변환 = 단위행렬)
4) 정렬 전/후를 각각 한 번씩 3D 창에 표시 (창을 닫으면 다음 단계로 진행)
5) Fitness, RMSE 및 사용한 파라미터를 콘솔에 출력

주의
- 함수/클래스 없이 위에서 아래로 순차 실행되도록 작성되었습니다.
- 기존 코드는 수정하지 않으며, 이 파일만 추가 실행하면 됩니다.
"""

import os
import sys
import copy
import numpy as np
import open3d as o3d

# ------------------------------------------------------------
# 1) 데이터 경로 설정 (이 스크립트 파일 기준 상대 경로)
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# 회전된 버니를 소스로 사용
source_path =  os.path.join(data_dir, "bunny_rotated.xyz")

# 원본 버니에 icp를 맞추기.
clean_path = os.path.join(data_dir, "bunny.xyz")
noisy_path = source_path

print("=" * 60)
print("ICP(Point-to-Point) 정렬 데모 시작")
print("=" * 60)
print(f"데이터 경로 (타겟: 깨끗한 버니): {clean_path}")
print(f"데이터 경로 (소스: 회전/노이즈 버니): {noisy_path}")

# ------------------------------------------------------------
# 2) XYZ 파일 로드 (numpy → Open3D PointCloud 변환)
# ------------------------------------------------------------
if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
    print("오류: 포인트 클라우드 데이터(.xyz)를 찾을 수 없습니다. 경로를 확인하세요.")
    sys.exit(1)

# 공백 구분 텍스트(.xyz) 로드: 각 줄이 "x y z"
clean_points = np.loadtxt(clean_path, delimiter=' ', dtype=np.float32)
noisy_points = np.loadtxt(noisy_path, delimiter=' ', dtype=np.float32)

print(f"깨끗한 버니 포인트 수: {len(clean_points)}")
print(f"노이즈 버니 포인트 수: {len(noisy_points)}")

# Open3D PointCloud 객체 생성
clean_pcd = o3d.geometry.PointCloud()
clean_pcd.points = o3d.utility.Vector3dVector(clean_points)

noisy_pcd = o3d.geometry.PointCloud()
noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)

# ------------------------------------------------------------
# 3) 다운샘플링 (시연 속도/안정성 향상) 및 스케일 적응적 파라미터
# ------------------------------------------------------------
# 데이터 스케일에 따라 파라미터가 지나치게 크거나 작지 않도록
clean_bbox = clean_pcd.get_axis_aligned_bounding_box()
diag_len = np.linalg.norm(np.array(clean_bbox.get_max_bound()) - np.array(clean_bbox.get_min_bound()))
voxel_size = max(0.005, diag_len * 0.01)  # 대략 대각선의 1%를 복셀 크기로 사용 (하한 0.005)

print(f"\n복셀 다운샘플링 수행 (voxel_size={voxel_size:.4f})")
clean_pcd_down = clean_pcd.voxel_down_sample(voxel_size)
noisy_pcd_down = noisy_pcd.voxel_down_sample(voxel_size)

print(f"다운샘플 후 - 깨끗한: {len(clean_pcd_down.points)} / 소스: {len(noisy_pcd_down.points)}")

# ------------------------------------------------------------
# 4) 정렬 전 시각화 (색상 지정 후 한 창에 같이 표시)
# ------------------------------------------------------------
print("\n정렬 전 상태를 시각화합니다. 창을 닫으면 전역 초기정렬(RANSAC) → ICP 순서로 진행합니다.")

# 시각화용 복사본에 색 입히기 (원본은 보존)
clean_vis_before = copy.deepcopy(clean_pcd_down)
noisy_vis_before = copy.deepcopy(noisy_pcd_down)
clean_vis_before.paint_uniform_color([0.7, 0.7, 0.7])  # 회색: 타겟
noisy_vis_before.paint_uniform_color([1.0, 0.0, 0.0])  # 빨강: 소스

o3d.visualization.draw_geometries(
    [clean_vis_before, noisy_vis_before],
    window_name="Before Registration (Close to continue)",
    width=960,
    height=720,
)

# ------------------------------------------------------------
# 5) 전역 초기 정렬 (FPFH + RANSAC) → ICP(Point-to-Point) 정제
# ------------------------------------------------------------
distance_threshold = voxel_size * 1.5
normal_radius = voxel_size * 2.0
fpfh_radius = voxel_size * 5.0

print("\n전역 초기 정렬 (FPFH + RANSAC) 수행 중...")

# 법선 추정
clean_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
noisy_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

# FPFH 특징
clean_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    clean_pcd_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100),
)
noisy_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    noisy_pcd_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100),
)

# RANSAC 기반 전역 정합
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    noisy_pcd_down, clean_pcd_down, noisy_fpfh, clean_fpfh,
    mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000),
)

print("전역 초기 정렬 결과")
print("- fitness: {:.6f}".format(ransac_result.fitness))
print("- inlier RMSE: {:.6f}".format(ransac_result.inlier_rmse))

# 전역 결과 시각화(옵션):
aligned_global = copy.deepcopy(noisy_pcd_down)
aligned_global.transform(ransac_result.transformation)
clean_vis_global = copy.deepcopy(clean_pcd_down)
clean_vis_global.paint_uniform_color([0.7, 0.7, 0.7])
aligned_global.paint_uniform_color([0.0, 0.6, 1.0])  # 하늘색
o3d.visualization.draw_geometries(
    [clean_vis_global, aligned_global],
    window_name="After Global Init (Close to run ICP refine)",
    width=960,
    height=720,
)

# ICP 정제
max_corr_dist = voxel_size * 1.0
print("\nICP 정제 수행 중...")
print(f"- max_correspondence_distance = {max_corr_dist:.4f}")
print("- 추정 방식 = Point-to-Point")

icp_result = o3d.pipelines.registration.registration_icp(
    noisy_pcd_down,
    clean_pcd_down,
    max_corr_dist,
    ransac_result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)

print("\nICP 결과")
print("- fitness (일치 비율): {:.6f}".format(icp_result.fitness))
print("- inlier RMSE: {:.6f}".format(icp_result.inlier_rmse))
print("- 변환 행렬:\n{}".format(icp_result.transformation))

# ------------------------------------------------------------
# 6) 정렬 후 시각화 (정렬된 소스를 초록색으로 표시)
# ------------------------------------------------------------
print("\n정렬 후 상태를 시각화합니다. 창을 닫으면 스크립트가 종료됩니다.")

clean_vis_after = copy.deepcopy(clean_pcd_down)
aligned_source = copy.deepcopy(noisy_pcd_down)
clean_vis_after.paint_uniform_color([0.7, 0.7, 0.7])   # 회색: 타겟
aligned_source.transform(icp_result.transformation)     # 소스에 변환 적용
aligned_source.paint_uniform_color([0.0, 1.0, 0.0])     # 초록: 정렬된 소스

o3d.visualization.draw_geometries(
    [clean_vis_after, aligned_source],
    window_name="After ICP (Close this window to finish)",
    width=960,
    height=720,
)

print("\n데모 완료!")

