#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
포인트 클라우드 성능 평가기
========================
포인트 클라우드 간의 거리 계산 및 성능 평가를 수행합니다.
"""

import open3d as o3d
import numpy as np

def align_point_clouds(source_pcd, target_pcd, max_distance=0.1):
    """
    ICP 알고리즘을 사용하여 두 포인트 클라우드를 정렬합니다.
    
    Args:
        source_pcd: 소스 포인트 클라우드
        target_pcd: 타겟 포인트 클라우드
        max_distance (float): 최대 거리 임계값
    
    Returns:
        tuple: (정렬된 포인트 클라우드, 변환 행렬)
    """
    print(f"ICP 정렬 수행 (최대 거리: {max_distance})")
    
    # ICP 정렬 수행
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # 정렬된 포인트 클라우드 생성
    aligned_pcd = source_pcd.transform(result.transformation)
    
    print(f"ICP 피트니스: {result.fitness:.6f}")
    print(f"ICP RMSE: {result.inlier_rmse:.6f}")
    
    return aligned_pcd, result.transformation

def calculate_point_distances(source_points, target_points):
    """
    두 포인트 클라우드 간의 거리를 계산합니다.
    
    Args:
        source_points: 소스 포인트 배열
        target_points: 타겟 포인트 배열
    
    Returns:
        list: 각 소스 포인트에서 가장 가까운 타겟 포인트까지의 거리
    """
    print("포인트 간 거리 계산 중...")
    distances = []
    
    for point in source_points:
        # 모든 타겟 포인트까지의 거리 계산
        point_distances = np.linalg.norm(target_points - point, axis=1)
        # 가장 가까운 포인트까지의 거리
        min_dist = np.min(point_distances)
        distances.append(min_dist)
    
    return distances

def evaluate_denoising_performance(noisy_pcd, denoised_pcd, clean_pcd):
    """
    디노이징 성능을 평가합니다.
    
    Args:
        noisy_pcd: 노이즈가 있는 포인트 클라우드
        denoised_pcd: 디노이징된 포인트 클라우드
        clean_pcd: 깨끗한 원본 포인트 클라우드
    
    Returns:
        dict: 성능 평가 결과
    """
    print("\n" + "=" * 30)
    print("성능 평가")
    print("=" * 30)
    
    # numpy 배열로 변환
    clean_points_array = np.asarray(clean_pcd.points)
    
    # 1. 노이즈 버니와 깨끗한 버니 간 거리 계산
    print("1단계: 노이즈 버니와 깨끗한 버니 간 거리 계산")
    noisy_aligned, _ = align_point_clouds(noisy_pcd, clean_pcd)
    noisy_points_array = np.asarray(noisy_aligned.points)
    noisy_distances = calculate_point_distances(noisy_points_array, clean_points_array)
    noisy_to_clean_dist = np.mean(noisy_distances)
    print(f"노이즈 버니 → 깨끗한 버니 평균 거리: {noisy_to_clean_dist:.6f}")
    
    # 2. 디노이징된 버니와 깨끗한 버니 간 거리 계산
    print("2단계: 디노이징된 버니와 깨끗한 버니 간 거리 계산")
    denoised_aligned, _ = align_point_clouds(denoised_pcd, clean_pcd)
    denoised_points_array = np.asarray(denoised_aligned.points)
    denoised_distances = calculate_point_distances(denoised_points_array, clean_points_array)
    denoised_to_clean_dist = np.mean(denoised_distances)
    print(f"디노이징된 버니 → 깨끗한 버니 평균 거리: {denoised_to_clean_dist:.6f}")
    
    # 3. 개선률 계산
    print("3단계: 개선률 계산")
    improvement = ((noisy_to_clean_dist - denoised_to_clean_dist) / noisy_to_clean_dist) * 100
    print(f"개선률: {improvement:.2f}%")
    
    # 결과 반환
    results = {
        'noisy_to_clean_distance': noisy_to_clean_dist,
        'denoised_to_clean_distance': denoised_to_clean_dist,
        'improvement_percentage': improvement,
        'noisy_distances': noisy_distances,
        'denoised_distances': denoised_distances
    }
    
    return results

def print_performance_summary(results):
    """
    성능 평가 결과를 요약하여 출력합니다.
    
    Args:
        results (dict): 성능 평가 결과
    """
    print("\n" + "=" * 30)
    print("성능 평가 요약")
    print("=" * 30)
    print(f"노이즈 버니 → 깨끗한 버니 평균 거리: {results['noisy_to_clean_distance']:.6f}")
    print(f"디노이징된 버니 → 깨끗한 버니 평균 거리: {results['denoised_to_clean_distance']:.6f}")
    print(f"개선률: {results['improvement_percentage']:.2f}%")
    
    if results['improvement_percentage'] > 0:
        print("✅ 디노이징이 성공적으로 수행되었습니다!")
    else:
        print("❌ 디노이징 효과가 없거나 오히려 악화되었습니다.")

if __name__ == "__main__":
    # 테스트 실행
    from point_cloud_loader import load_bunny_data
    from denoising_processor import denoise_point_cloud
    
    # 데이터 로드
    noisy_points, noisy_pcd, clean_points, clean_pcd = load_bunny_data()
    
    if noisy_pcd is not None and clean_pcd is not None:
        # 디노이징 수행
        denoised_pcd = denoise_point_cloud(noisy_pcd)
        
        # 성능 평가
        results = evaluate_denoising_performance(noisy_pcd, denoised_pcd, clean_pcd)
        print_performance_summary(results)
    else:
        print("데이터 로드 실패!") 