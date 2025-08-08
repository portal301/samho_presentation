#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
포인트 클라우드 데이터 로더
========================
XYZ 파일에서 포인트 클라우드 데이터를 로드하는 기능을 제공합니다.
"""

import open3d as o3d
import numpy as np

def load_xyz_file(filepath):
    """
    XYZ 파일에서 포인트 클라우드 데이터를 로드합니다.
    
    Args:
        filepath (str): XYZ 파일 경로
    
    Returns:
        tuple: (포인트 리스트, Open3D 포인트 클라우드 객체)
    """
    print(f"파일을 읽는 중: {filepath}")
    points = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                coords = line.strip().split()
                if len(coords) >= 3:
                    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                    points.append([x, y, z])
        
        # Open3D 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        
        print(f"포인트 수: {len(points)}")
        return points, pcd
        
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {filepath}")
        return None, None
    except Exception as e:
        print(f"오류: 파일 읽기 실패 - {e}")
        return None, None

def load_bunny_data():
    """
    버니 데이터셋을 로드합니다.
    
    Returns:
        tuple: (노이즈 포인트, 노이즈 PCD, 깨끗한 포인트, 깨끗한 PCD)
    """
    print("=" * 30)
    print("버니 데이터셋 로드")
    print("=" * 30)
    
    # 노이즈가 있는 버니 데이터 로드
    noisy_points, noisy_pcd = load_xyz_file("data/bunny_noisy_extreme.xyz")
    if noisy_points is None:
        return None, None, None, None
    
    # 깨끗한 버니 데이터 로드
    clean_points, clean_pcd = load_xyz_file("data/bunny.xyz")
    if clean_points is None:
        return None, None, None, None
    
    return noisy_points, noisy_pcd, clean_points, clean_pcd

if __name__ == "__main__":
    # 테스트 실행
    noisy_points, noisy_pcd, clean_points, clean_pcd = load_bunny_data()
    
    if noisy_pcd is not None and clean_pcd is not None:
        print("\n데이터 로드 성공!")
        print(f"노이즈 버니 포인트 수: {len(noisy_points)}")
        print(f"깨끗한 버니 포인트 수: {len(clean_points)}")
    else:
        print("\n데이터 로드 실패!") 