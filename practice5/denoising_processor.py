#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
포인트 클라우드 디노이징 프로세서
==============================
포인트 클라우드에서 노이즈를 제거하는 다양한 방법을 제공합니다.
"""

import open3d as o3d
import numpy as np

def voxel_downsampling(pcd, voxel_size=0.05):
    """
    복셀 다운샘플링을 수행합니다.
    
    Args:
        pcd: Open3D 포인트 클라우드 객체
        voxel_size (float): 복셀 크기
    
    Returns:
        Open3D 포인트 클라우드 객체: 다운샘플링된 포인트 클라우드
    """
    print(f"복셀 다운샘플링 수행 (복셀 크기: {voxel_size})")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"다운샘플링 후 포인트 수: {len(pcd_down.points)}")
    return pcd_down

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    통계적 아웃라이어를 제거합니다.
    
    Args:
        pcd: Open3D 포인트 클라우드 객체
        nb_neighbors (int): 이웃 포인트 수
        std_ratio (float): 표준편차 비율
    
    Returns:
        tuple: (정제된 포인트 클라우드, 제거된 포인트 인덱스)
    """
    print(f"통계적 아웃라이어 제거 (이웃 수: {nb_neighbors}, 표준편차 비율: {std_ratio})")
    pcd_clean, outlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print(f"아웃라이어 제거 후 포인트 수: {len(pcd_clean.points)}")
    return pcd_clean, outlier_indices

def estimate_normals(pcd, radius=0.1, max_nn=30):
    """
    법선 벡터를 계산합니다.
    
    Args:
        pcd: Open3D 포인트 클라우드 객체
        radius (float): 검색 반경
        max_nn (int): 최대 이웃 포인트 수
    
    Returns:
        Open3D 포인트 클라우드 객체: 법선이 계산된 포인트 클라우드
    """
    print(f"법선 벡터 계산 (반경: {radius}, 최대 이웃 수: {max_nn})")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    print("법선 벡터 계산 완료")
    return pcd

def dbscan_clustering(pcd, eps=0.05, min_points=5):
    """
    DBSCAN 클러스터링을 수행하여 노이즈를 제거합니다.
    
    Args:
        pcd: Open3D 포인트 클라우드 객체
        eps (float): 이웃 포인트 간 최대 거리
        min_points (int): 클러스터로 인정할 최소 포인트 수
    
    Returns:
        Open3D 포인트 클라우드 객체: 클러스터링된 포인트 클라우드
    """
    print(f"DBSCAN 클러스터링 수행 (eps: {eps}, 최소 포인트: {min_points})")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    print(f"클러스터 수: {max_label + 1}")
    
    if max_label >= 0:
        # 가장 큰 클러스터만 유지
        cluster_sizes = np.bincount(labels[labels >= 0])
        largest_cluster_label = np.argmax(cluster_sizes)
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        denoised_pcd = pcd.select_by_index(largest_cluster_indices)
        print(f"최대 클러스터 포인트 수: {len(denoised_pcd.points)}")
        return denoised_pcd
    else:
        print("클러스터가 생성되지 않았습니다. 원본 데이터를 반환합니다.")
        return pcd

def denoise_point_cloud(pcd, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0, 
                       eps=0.05, min_points=5):
    """
    포인트 클라우드 디노이징의 전체 과정을 수행합니다.
    
    Args:
        pcd: Open3D 포인트 클라우드 객체
        voxel_size (float): 복셀 크기
        nb_neighbors (int): 이웃 포인트 수
        std_ratio (float): 표준편차 비율
        eps (float): DBSCAN eps 파라미터
        min_points (int): DBSCAN 최소 포인트 수
    
    Returns:
        Open3D 포인트 클라우드 객체: 디노이징된 포인트 클라우드
    """
    print("\n" + "=" * 30)
    print("포인트 클라우드 디노이징 과정")
    print("=" * 30)
    
    # 1단계: 복셀 다운샘플링
    pcd_down = voxel_downsampling(pcd, voxel_size)
    
    # 2단계: 통계적 아웃라이어 제거
    pcd_clean, _ = remove_statistical_outliers(pcd_down, nb_neighbors, std_ratio)
    
    # 3단계: 법선 벡터 계산
    pcd_clean = estimate_normals(pcd_clean)
    
    # 4단계: DBSCAN 클러스터링
    denoised_pcd = dbscan_clustering(pcd_clean, eps, min_points)
    
    print("\n디노이징 과정 완료!")
    return denoised_pcd

if __name__ == "__main__":
    # 테스트 실행
    from point_cloud_loader import load_bunny_data
    
    # 데이터 로드
    noisy_points, noisy_pcd, clean_points, clean_pcd = load_bunny_data()
    
    if noisy_pcd is not None:
        # 디노이징 수행
        denoised_pcd = denoise_point_cloud(noisy_pcd)
        print(f"\n최종 결과:")
        print(f"원본 포인트 수: {len(noisy_points)}")
        print(f"디노이징 후 포인트 수: {len(denoised_pcd.points)}")
    else:
        print("데이터 로드 실패!") 