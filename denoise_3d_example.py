#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 3D 포인트 클라우드 디노이징 튜토리얼
==========================================

Open3D를 활용한 기본적인 포인트 클라우드 디노이징
bunny_noisy.xyz를 디노이징하고 bunny.xyz와 비교

작성자: AI Assistant
"""

import open3d as o3d  # Open3D 라이브러리 임포트 (3D 포인트 클라우드 처리용)
import numpy as np    # 수치 계산용 라이브러리
import time          # 시간 측정용

print("=" * 50)
print("간단한 3D 포인트 클라우드 디노이징 튜토리얼")
print("=" * 50)

# 시작 시간 기록
start_time = time.time()

print("\n1단계: 포인트 클라우드 파일 로드")
print("-" * 30)

# bunny_noisy.xyz 파일에서 포인트 좌표 읽기
print("노이즈가 있는 버니 파일을 읽는 중...")
noisy_points = []  # 노이즈 포인트를 저장할 리스트
with open("data/bunny_noisy.xyz", 'r') as f:  # 파일을 읽기 모드로 열기
    for line in f:  # 파일의 각 줄을 순회
        coords = line.strip().split()  # 줄을 공백으로 분리하여 좌표 추출
        if len(coords) >= 3:  # x, y, z 좌표가 모두 있는지 확인
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])  # 문자열을 실수로 변환
            noisy_points.append([x, y, z])  # 포인트 리스트에 추가

# Open3D 포인트 클라우드 객체 생성
noisy_pcd = o3d.geometry.PointCloud()  # 빈 포인트 클라우드 생성
noisy_pcd.points = o3d.utility.Vector3dVector(np.array(noisy_points))  # 포인트 좌표 설정
print(f"노이즈 버니 포인트 수: {len(noisy_points)}")

# bunny.xyz 파일에서 깨끗한 포인트 좌표 읽기 (비교용)
print("깨끗한 버니 파일을 읽는 중...")
clean_points = []  # 깨끗한 포인트를 저장할 리스트
with open("data/bunny.xyz", 'r') as f:  # 파일을 읽기 모드로 열기
    for line in f:  # 파일의 각 줄을 순회
        coords = line.strip().split()  # 줄을 공백으로 분리하여 좌표 추출
        if len(coords) >= 3:  # x, y, z 좌표가 모두 있는지 확인
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])  # 문자열을 실수로 변환
            clean_points.append([x, y, z])  # 포인트 리스트에 추가

# Open3D 포인트 클라우드 객체 생성
clean_pcd = o3d.geometry.PointCloud()  # 빈 포인트 클라우드 생성
clean_pcd.points = o3d.utility.Vector3dVector(np.array(clean_points))  # 포인트 좌표 설정
print(f"깨끗한 버니 포인트 수: {len(clean_points)}")

print("\n2단계: 디노이징 과정")
print("-" * 30)

print("2-1단계: 복셀 다운샘플링 (노이즈 감소)")
# 복셀 다운샘플링: 3D 공간을 작은 정육면체(복셀)로 나누고, 각 복셀 내의 포인트들을 하나로 통합
voxel_size = 0.05  # 복셀 크기 설정 (작을수록 더 많은 포인트 유지)
pcd_down = noisy_pcd.voxel_down_sample(voxel_size=voxel_size)  # 복셀 다운샘플링 수행
print(f"다운샘플링 후 포인트 수: {len(pcd_down.points)}")

print("2-2단계: 통계적 아웃라이어 제거")
# 통계적 아웃라이어 제거: 주변 포인트들과 거리가 먼 이상치 포인트들을 제거
nb_neighbors = 20  # 이웃 포인트 수 (주변 20개 포인트를 기준으로 판단)
std_ratio = 2.0    # 표준편차 비율 (평균에서 2배 표준편차 이상 벗어나면 아웃라이어로 판단)
pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)  # 아웃라이어 제거
print(f"아웃라이어 제거 후 포인트 수: {len(pcd_clean.points)}")

print("2-3단계: 법선 벡터 계산")
# 법선 벡터: 각 포인트에서 표면의 수직 방향을 나타내는 벡터
pcd_clean.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)  # 법선 계산 파라미터
)
print("법선 벡터 계산 완료")

print("2-4단계: DBSCAN 클러스터링으로 노이즈 제거")
# DBSCAN: 밀도 기반 클러스터링으로 연결된 포인트들을 그룹화
eps = 0.05         # 이웃 포인트 간 최대 거리 (더 작게 설정)
min_points = 5     # 클러스터로 인정할 최소 포인트 수 (더 작게 설정)
labels = np.array(pcd_clean.cluster_dbscan(eps=eps, min_points=min_points))  # DBSCAN 클러스터링 수행
max_label = labels.max()  # 클러스터 개수 확인
print(f"클러스터 수: {max_label + 1}")

# 클러스터가 생성되었는지 확인
if max_label >= 0:  # 클러스터가 존재하는 경우
    # 가장 큰 클러스터만 유지 (노이즈 제거)
    cluster_sizes = np.bincount(labels[labels >= 0])  # 각 클러스터의 크기 계산
    largest_cluster_label = np.argmax(cluster_sizes)  # 가장 큰 클러스터의 라벨 찾기
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]  # 가장 큰 클러스터의 인덱스들
    denoised_pcd = pcd_clean.select_by_index(largest_cluster_indices)  # 가장 큰 클러스터만 선택
    print(f"최대 클러스터 포인트 수: {len(denoised_pcd.points)}")
else:  # 클러스터가 생성되지 않은 경우 (모든 포인트가 노이즈로 분류)
    print("클러스터가 생성되지 않았습니다. 아웃라이어 제거 결과를 그대로 사용합니다.")
    denoised_pcd = pcd_clean  # 아웃라이어 제거 결과를 그대로 사용

print("\n3단계: 성능 평가")
print("-" * 30)

print("3-1단계: 노이즈 버니와 깨끗한 버니 간 거리 계산")
# ICP (Iterative Closest Point) 알고리즘으로 두 포인트 클라우드를 정렬
result = o3d.pipelines.registration.registration_icp(
    noisy_pcd, clean_pcd, 0.1, np.eye(4),  # ICP 정렬 수행
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
noisy_aligned = noisy_pcd.transform(result.transformation)  # 정렬된 노이즈 포인트 클라우드

# 포인트 간 거리 계산
noisy_points_array = np.asarray(noisy_aligned.points)  # numpy 배열로 변환
clean_points_array = np.asarray(clean_pcd.points)      # numpy 배열로 변환
noisy_distances = []  # 거리를 저장할 리스트
for point in noisy_points_array:  # 각 노이즈 포인트에 대해
    distances = np.linalg.norm(clean_points_array - point, axis=1)  # 모든 깨끗한 포인트까지의 거리 계산
    min_dist = np.min(distances)  # 가장 가까운 포인트까지의 거리
    noisy_distances.append(min_dist)  # 최소 거리 저장
noisy_to_clean_dist = np.mean(noisy_distances)  # 평균 거리 계산
print(f"노이즈 버니 → 깨끗한 버니 평균 거리: {noisy_to_clean_dist:.6f}")

print("3-2단계: 디노이징된 버니와 깨끗한 버니 간 거리 계산")
# 디노이징된 포인트 클라우드도 정렬
result = o3d.pipelines.registration.registration_icp(
    denoised_pcd, clean_pcd, 0.1, np.eye(4),  # ICP 정렬 수행
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
denoised_aligned = denoised_pcd.transform(result.transformation)  # 정렬된 디노이징 포인트 클라우드

# 포인트 간 거리 계산
denoised_points_array = np.asarray(denoised_aligned.points)  # numpy 배열로 변환
denoised_distances = []  # 거리를 저장할 리스트
for point in denoised_points_array:  # 각 디노이징 포인트에 대해
    distances = np.linalg.norm(clean_points_array - point, axis=1)  # 모든 깨끗한 포인트까지의 거리 계산
    min_dist = np.min(distances)  # 가장 가까운 포인트까지의 거리
    denoised_distances.append(min_dist)  # 최소 거리 저장
denoised_to_clean_dist = np.mean(denoised_distances)  # 평균 거리 계산
print(f"디노이징된 버니 → 깨끗한 버니 평균 거리: {denoised_to_clean_dist:.6f}")

print("3-3단계: 개선률 계산")
# 개선률 = (원본 거리 - 디노이징 거리) / 원본 거리 * 100
improvement = ((noisy_to_clean_dist - denoised_to_clean_dist) / noisy_to_clean_dist) * 100
print(f"개선률: {improvement:.2f}%")

print("\n4단계: 결과 저장")
print("-" * 30)

# 디노이징된 포인트 클라우드를 .xyz 파일로 저장
output_filename = "data/simple_denoised_bunny.xyz"
denoised_points_array = np.asarray(denoised_pcd.points)  # 디노이징된 포인트들을 numpy 배열로 변환
with open(output_filename, 'w') as f:  # 파일을 쓰기 모드로 열기
    for point in denoised_points_array:  # 각 포인트에 대해
        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")  # x, y, z 좌표를 파일에 저장
print(f"디노이징 결과가 '{output_filename}' 파일로 저장되었습니다.")

print("\n5단계: 통계 정보")
print("-" * 30)

# 각 단계별 포인트 수 출력
print(f"원본 노이즈 포인트 수: {len(noisy_points)}")
print(f"디노이징 후 포인트 수: {len(denoised_pcd.points)}")
print(f"깨끗한 원본 포인트 수: {len(clean_points)}")

# 총 소요 시간 계산
end_time = time.time()
total_time = end_time - start_time
print(f"\n총 소요 시간: {total_time:.2f}초")

print("\n" + "=" * 50)
print("튜토리얼 완료!")
print("=" * 50)

print("\n학습한 내용:")
print("1. Open3D를 사용한 포인트 클라우드 로드")
print("2. 복셀 다운샘플링으로 노이즈 감소")
print("3. 통계적 아웃라이어 제거")
print("4. 법선 벡터 계산")
print("5. DBSCAN 클러스터링으로 노이즈 제거")
print("6. ICP를 사용한 포인트 클라우드 정렬")
print("7. 거리 기반 성능 평가")
print("8. 결과 파일 저장") 