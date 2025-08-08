#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
결과 저장 및 통계 출력기
=====================
포인트 클라우드 결과를 파일로 저장하고 통계 정보를 출력합니다.
"""

import numpy as np
import time
import os

def save_point_cloud_to_xyz(pcd, output_filename):
    """
    포인트 클라우드를 XYZ 파일로 저장합니다.
    
    Args:
        pcd: Open3D 포인트 클라우드 객체
        output_filename (str): 출력 파일명
    
    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # 포인트 좌표를 numpy 배열로 변환
        points_array = np.asarray(pcd.points)
        
        # XYZ 파일로 저장
        with open(output_filename, 'w') as f:
            for point in points_array:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"포인트 클라우드가 '{output_filename}' 파일로 저장되었습니다.")
        return True
        
    except Exception as e:
        print(f"오류: 파일 저장 실패 - {e}")
        return False

def print_statistics(noisy_points, denoised_pcd, clean_points, start_time):
    """
    처리 과정의 통계 정보를 출력합니다.
    
    Args:
        noisy_points: 원본 노이즈 포인트 리스트
        denoised_pcd: 디노이징된 포인트 클라우드
        clean_points: 깨끗한 원본 포인트 리스트
        start_time: 시작 시간
    """
    print("\n" + "=" * 30)
    print("통계 정보")
    print("=" * 30)
    
    # 포인트 수 통계
    print(f"원본 노이즈 포인트 수: {len(noisy_points)}")
    print(f"디노이징 후 포인트 수: {len(denoised_pcd.points)}")
    print(f"깨끗한 원본 포인트 수: {len(clean_points)}")
    
    # 포인트 수 변화율
    reduction_rate = ((len(noisy_points) - len(denoised_pcd.points)) / len(noisy_points)) * 100
    print(f"포인트 수 감소율: {reduction_rate:.2f}%")
    
    # 처리 시간
    end_time = time.time()
    total_time = end_time - start_time
    print(f"총 소요 시간: {total_time:.2f}초")
    
    # 처리 속도
    if total_time > 0:
        points_per_second = len(noisy_points) / total_time
        print(f"처리 속도: {points_per_second:.0f} 포인트/초")

if __name__ == "__main__":
    # 테스트 실행
    from point_cloud_loader import load_bunny_data
    from denoising_processor import denoise_point_cloud
    from performance_evaluator import evaluate_denoising_performance
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 데이터 로드 및 처리
    noisy_points, noisy_pcd, clean_points, clean_pcd = load_bunny_data()
    
    if noisy_pcd is not None and clean_pcd is not None:
        # 디노이징 수행
        denoised_pcd = denoise_point_cloud(
            noisy_pcd
        )
        
        # 성능 평가
        results = evaluate_denoising_performance(noisy_pcd, denoised_pcd, clean_pcd)
        
        # 결과 저장
        save_point_cloud_to_xyz(denoised_pcd, "data/denoised_bunny.xyz")
        
        # 통계 출력
        print_statistics(noisy_points, denoised_pcd, clean_points, start_time)
      
    else:
        print("데이터 로드 실패!") 