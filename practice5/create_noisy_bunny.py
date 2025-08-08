#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
랜덤 노이즈가 심한 버니 포인트 클라우드 생성기
===========================================
bunny.xyz 파일을 로드하여 다양한 강도의 랜덤 노이즈를 추가한 
bunny_noisy_made.xyz 파일을 생성합니다.
"""

import numpy as np
import open3d as o3d
import os

def load_xyz_file(file_path):
    """
    XYZ 파일을 로드합니다.
    
    Args:
        file_path (str): XYZ 파일 경로
    
    Returns:
        tuple: (points, pcd) - NumPy 배열과 Open3D 포인트 클라우드 객체
    """
    try:
        # XYZ 파일 로드
        points = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
        print(f"파일 로드 성공: {file_path}")
        print(f"포인트 수: {len(points)}")
        
        # Open3D 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return points, pcd
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return None, None

def add_random_noise(points, noise_intensity=0.1, noise_ratio=0.3):
    """
    포인트 클라우드에 랜덤 노이즈를 추가합니다.
    
    Args:
        points (np.ndarray): 원본 포인트 (N, 3)
        noise_intensity (float): 노이즈 강도 (기본값: 0.1)
        noise_ratio (float): 노이즈를 추가할 포인트의 비율 (기본값: 0.3)
    
    Returns:
        np.ndarray: 노이즈가 추가된 포인트
    """
    print(f"노이즈 추가 중... (강도: {noise_intensity}, 비율: {noise_ratio})")
    
    # 원본 포인트 복사
    noisy_points = points.copy()
    
    # 노이즈를 추가할 포인트 수 계산
    num_points = len(points)
    num_noise_points = int(num_points * noise_ratio)
    
    print(f"총 포인트 수: {num_points}")
    print(f"노이즈 추가할 포인트 수: {num_noise_points}")
    
    # 랜덤하게 선택된 포인트에 노이즈 추가
    noise_indices = np.random.choice(num_points, num_noise_points, replace=False)
    
    # 노이즈 생성 (정규분포 사용)
    noise = np.random.normal(0, noise_intensity, (num_noise_points, 3))
    
    # 선택된 포인트에 노이즈 추가
    noisy_points[noise_indices] += noise
    
    print(f"노이즈 추가 완료!")
    return noisy_points

def add_outlier_noise(points, outlier_ratio=0.05, outlier_distance=0.5):
    """
    포인트 클라우드에 아웃라이어 노이즈를 추가합니다.
    
    Args:
        points (np.ndarray): 원본 포인트 (N, 3)
        outlier_ratio (float): 아웃라이어 비율 (기본값: 0.05)
        outlier_distance (float): 아웃라이어 거리 (기본값: 0.5)
    
    Returns:
        np.ndarray: 아웃라이어가 추가된 포인트
    """
    print(f"아웃라이어 노이즈 추가 중... (비율: {outlier_ratio}, 거리: {outlier_distance})")
    
    # 원본 포인트 복사
    noisy_points = points.copy()
    
    # 아웃라이어 수 계산
    num_points = len(points)
    num_outliers = int(num_points * outlier_ratio)
    
    print(f"아웃라이어 추가할 포인트 수: {num_outliers}")
    
    # 포인트의 중심 계산
    center = np.mean(points, axis=0)
    
    # 아웃라이어 생성
    for i in range(num_outliers):
        # 랜덤한 방향으로 아웃라이어 생성
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        # 아웃라이어 거리 (랜덤)
        distance = outlier_distance + np.random.uniform(0, outlier_distance)
        
        # 아웃라이어 포인트 생성
        outlier_point = center + direction * distance
        
        # 기존 포인트에 추가
        noisy_points = np.vstack([noisy_points, outlier_point])
    
    print(f"아웃라이어 추가 완료!")
    return noisy_points

def create_noisy_bunny(input_file="data/bunny.xyz", 
                      output_file="data/bunny_noisy_made.xyz",
                      noise_intensity=0.15,
                      noise_ratio=0.4,
                      outlier_ratio=0.08,
                      outlier_distance=0.8):
    """
    랜덤 노이즈가 심한 버니 포인트 클라우드를 생성합니다.
    
    Args:
        input_file (str): 입력 파일 경로
        output_file (str): 출력 파일 경로
        noise_intensity (float): 노이즈 강도
        noise_ratio (float): 노이즈 비율
        outlier_ratio (float): 아웃라이어 비율
        outlier_distance (float): 아웃라이어 거리
    """
    print("=" * 50)
    print("랜덤 노이즈가 심한 버니 포인트 클라우드 생성")
    print("=" * 50)
    
    # 1. 원본 파일 로드
    print(f"\n1단계: 원본 파일 로드")
    points, pcd = load_xyz_file(input_file)
    
    if points is None:
        print("원본 파일 로드 실패!")
        return
    
    # 2. 랜덤 노이즈 추가
    print(f"\n2단계: 랜덤 노이즈 추가")
    noisy_points = add_random_noise(points, noise_intensity, noise_ratio)
    
    # 3. 아웃라이어 노이즈 추가
    print(f"\n3단계: 아웃라이어 노이즈 추가")
    noisy_points = add_outlier_noise(noisy_points, outlier_ratio, outlier_distance)
    
    # 4. 결과 저장
    print(f"\n4단계: 결과 저장")
    try:
        np.savetxt(output_file, noisy_points, delimiter=' ', fmt='%.6f')
        print(f"노이즈가 추가된 파일 저장 완료: {output_file}")
        print(f"최종 포인트 수: {len(noisy_points)}")
        
        # 통계 정보 출력
        print(f"\n통계 정보:")
        print(f"- 원본 포인트 수: {len(points)}")
        print(f"- 노이즈 추가 후 포인트 수: {len(noisy_points)}")
        print(f"- 추가된 포인트 수: {len(noisy_points) - len(points)}")
        print(f"- 노이즈 강도: {noise_intensity}")
        print(f"- 노이즈 비율: {noise_ratio}")
        print(f"- 아웃라이어 비율: {outlier_ratio}")
        print(f"- 아웃라이어 거리: {outlier_distance}")
        
    except Exception as e:
        print(f"파일 저장 실패: {e}")

def create_multiple_noisy_versions():
    """
    다양한 강도의 노이즈가 있는 버전들을 생성합니다.
    """
    print("=" * 50)
    print("다양한 강도의 노이즈 버전 생성")
    print("=" * 50)
    
    # 다양한 노이즈 설정
    noise_configs = [
        {"name": "light", "intensity": 0.05, "ratio": 0.2, "outlier_ratio": 0.03, "outlier_distance": 0.3},
        {"name": "medium", "intensity": 0.15, "ratio": 0.4, "outlier_ratio": 0.08, "outlier_distance": 0.8},
        {"name": "heavy", "intensity": 0.25, "ratio": 0.6, "outlier_ratio": 0.15, "outlier_distance": 1.2},
        {"name": "extreme", "intensity": 0.4, "ratio": 0.8, "outlier_ratio": 0.25, "outlier_distance": 2.0}
    ]
    
    for config in noise_configs:
        print(f"\n{config['name'].upper()} 노이즈 버전 생성 중...")
        output_file = f"data/bunny_noisy_{config['name']}.xyz"
        
        create_noisy_bunny(
            input_file="data/bunny.xyz",
            output_file=output_file,
            noise_intensity=config["intensity"],
            noise_ratio=config["ratio"],
            outlier_ratio=config["outlier_ratio"],
            outlier_distance=config["outlier_distance"]
        )

if __name__ == "__main__":
    # 기본 노이즈 버전 생성
    print("기본 노이즈 버전 생성")
    create_noisy_bunny()
    
    # 추가로 다양한 강도의 노이즈 버전도 생성할지 묻기
    print("\n" + "=" * 50)
    response = input("다양한 강도의 노이즈 버전도 생성하시겠습니까? (y/n): ")
    
    if response.lower() in ['y', 'yes', '예']:
        create_multiple_noisy_versions()
        print("\n모든 노이즈 버전 생성 완료!")
    else:
        print("기본 노이즈 버전만 생성되었습니다.")
    
    print("\n프로그램 종료") 