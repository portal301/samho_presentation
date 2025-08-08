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

def save_performance_report(results, output_filename="performance_report.txt"):
    """
    성능 평가 결과를 텍스트 파일로 저장합니다.
    
    Args:
        results (dict): 성능 평가 결과
        output_filename (str): 출력 파일명
    """
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("포인트 클라우드 디노이징 성능 평가 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 거리 측정 결과\n")
            f.write("-" * 20 + "\n")
            f.write(f"노이즈 버니 → 깨끗한 버니 평균 거리: {results['noisy_to_clean_distance']:.6f}\n")
            f.write(f"디노이징된 버니 → 깨끗한 버니 평균 거리: {results['denoised_to_clean_distance']:.6f}\n\n")
            
            f.write("2. 개선 효과\n")
            f.write("-" * 20 + "\n")
            f.write(f"개선률: {results['improvement_percentage']:.2f}%\n")
            
            if results['improvement_percentage'] > 0:
                f.write("결론: 디노이징이 성공적으로 수행되었습니다.\n")
            else:
                f.write("결론: 디노이징 효과가 없거나 오히려 악화되었습니다.\n")
            
            f.write("\n3. 상세 통계\n")
            f.write("-" * 20 + "\n")
            f.write(f"노이즈 거리 표준편차: {np.std(results['noisy_distances']):.6f}\n")
            f.write(f"디노이징 거리 표준편차: {np.std(results['denoised_distances']):.6f}\n")
            f.write(f"노이즈 거리 최대값: {np.max(results['noisy_distances']):.6f}\n")
            f.write(f"디노이징 거리 최대값: {np.max(results['denoised_distances']):.6f}\n")
        
        print(f"성능 평가 보고서가 '{output_filename}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"오류: 성능 보고서 저장 실패 - {e}")

def create_summary_report(noisy_points, denoised_pcd, clean_points, results, 
                         start_time, output_filename="summary_report.txt"):
    """
    전체 처리 과정의 요약 보고서를 생성합니다.
    
    Args:
        noisy_points: 원본 노이즈 포인트 리스트
        denoised_pcd: 디노이징된 포인트 클라우드
        clean_points: 깨끗한 원본 포인트 리스트
        results: 성능 평가 결과
        start_time: 시작 시간
        output_filename: 출력 파일명
    """
    try:
        end_time = time.time()
        total_time = end_time - start_time
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("포인트 클라우드 디노이징 요약 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 데이터 정보\n")
            f.write("-" * 20 + "\n")
            f.write(f"원본 노이즈 포인트 수: {len(noisy_points)}\n")
            f.write(f"디노이징 후 포인트 수: {len(denoised_pcd.points)}\n")
            f.write(f"깨끗한 원본 포인트 수: {len(clean_points)}\n")
            f.write(f"포인트 수 감소율: {((len(noisy_points) - len(denoised_pcd.points)) / len(noisy_points)) * 100:.2f}%\n\n")
            
            f.write("2. 성능 평가\n")
            f.write("-" * 20 + "\n")
            f.write(f"노이즈 버니 → 깨끗한 버니 평균 거리: {results['noisy_to_clean_distance']:.6f}\n")
            f.write(f"디노이징된 버니 → 깨끗한 버니 평균 거리: {results['denoised_to_clean_distance']:.6f}\n")
            f.write(f"개선률: {results['improvement_percentage']:.2f}%\n\n")
            
            f.write("3. 처리 정보\n")
            f.write("-" * 20 + "\n")
            f.write(f"총 소요 시간: {total_time:.2f}초\n")
            if total_time > 0:
                f.write(f"처리 속도: {len(noisy_points) / total_time:.0f} 포인트/초\n")
            
            f.write("\n4. 결론\n")
            f.write("-" * 20 + "\n")
            if results['improvement_percentage'] > 0:
                f.write("디노이징이 성공적으로 수행되어 포인트 클라우드의 품질이 향상되었습니다.\n")
            else:
                f.write("디노이징 효과가 제한적이거나 오히려 품질이 악화되었습니다.\n")
        
        print(f"요약 보고서가 '{output_filename}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"오류: 요약 보고서 저장 실패 - {e}")

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
        
        # 보고서 생성
        save_performance_report(results)
        create_summary_report(noisy_points, denoised_pcd, clean_points, results, start_time)
        
    else:
        print("데이터 로드 실패!") 