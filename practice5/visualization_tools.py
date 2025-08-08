#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice 5 - 3D Point Cloud Visualization Tool
============================================
3개의 창을 동시에 띄워서 각각 다른 포인트 클라우드를 시각화하는 도구
"""

import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time

class PointCloudVisualizer:
    def __init__(self):
        """
        3D 포인트 클라우드 시각화 도구 초기화
        """
        self.point_clouds = {}
        self.visualizers = {}
        self.is_running = False
        
        # GUI 초기화
        self.setup_gui()
        
        # 포인트 클라우드 로드
        self.load_point_clouds()
        
    def setup_gui(self):
        """
        GUI 설정
        """
        self.root = tk.Tk()
        self.root.title("3D Point Cloud Visualizer")
        self.root.geometry("400x200")
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="3D Point Cloud Visualization", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 설명
        desc_label = ttk.Label(main_frame, text="3개의 창이 열려서 각각 다른 포인트 클라우드를 표시합니다", 
                              font=("Arial", 10))
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # 시작 버튼
        start_button = ttk.Button(main_frame, text="3D 시각화 시작", 
                                 command=self.start_visualization)
        start_button.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        # 종료 버튼
        quit_button = ttk.Button(main_frame, text="종료", command=self.quit_app)
        quit_button.grid(row=3, column=0, columnspan=2, pady=(0, 0))
        
        # 창 크기 조정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def load_point_clouds(self):
        """
        포인트 클라우드 파일들을 로드합니다.
        """
        print("포인트 클라우드 로딩 중...")
        
        try:
            # 1. 원본 버니 (깨끗한 버전)
            print("원본 버니 로딩...")
            original_points, original_pcd = self.load_xyz_file("data/bunny.xyz")
            if original_pcd is not None:
                # 토끼를 올바른 방향으로 회전
                original_pcd = self.rotate_point_cloud(original_pcd)
                self.point_clouds['original'] = {
                    'pcd': original_pcd,
                    'name': 'Original Bunny',
                    'points': len(original_points)
                }
                print(f"원본 버니 로드 완료: {len(original_points)} 포인트")
            
            # 2. 노이즈 버니
            print("노이즈 버니 로딩...")
            noisy_points, noisy_pcd = self.load_xyz_file("data/bunny_noisy_extreme.xyz")
            if noisy_pcd is not None:
                # 토끼를 올바른 방향으로 회전
                noisy_pcd = self.rotate_point_cloud(noisy_pcd)
                self.point_clouds['noisy'] = {
                    'pcd': noisy_pcd,
                    'name': 'Noisy Bunny',
                    'points': len(noisy_points)
                }
                print(f"노이즈 버니 로드 완료: {len(noisy_points)} 포인트")
            
            # 3. 디노이징된 버니 (존재하는 경우)
            print("디노이징된 버니 로딩...")
            denoised_points, denoised_pcd = self.load_xyz_file("data/denoised_bunny.xyz")
            if denoised_pcd is not None:
                # 토끼를 올바른 방향으로 회전
                denoised_pcd = self.rotate_point_cloud(denoised_pcd)
                self.point_clouds['denoised'] = {
                    'pcd': denoised_pcd,
                    'name': 'Denoised Bunny',
                    'points': len(denoised_points)
                }
                print(f"디노이징된 버니 로드 완료: {len(denoised_points)} 포인트")
            else:
                # 디노이징된 파일이 없으면 노이즈 버니를 복사
                print("디노이징된 버니 파일이 없습니다. 노이즈 버니를 복사합니다.")
                denoised_pcd = noisy_pcd.clone()
                self.point_clouds['denoised'] = {
                    'pcd': denoised_pcd,
                    'name': 'Denoised Bunny (Copy)',
                    'points': len(noisy_points)
                }
            
            print("모든 포인트 클라우드 로딩 완료!")
            
        except Exception as e:
            print(f"포인트 클라우드 로딩 오류: {e}")
    
    def rotate_point_cloud(self, pcd):
        """
        포인트 클라우드를 올바른 방향으로 회전시킵니다.
        
        Args:
            pcd: Open3D 포인트 클라우드 객체
        
        Returns:
            Open3D 포인트 클라우드 객체: 회전된 포인트 클라우드
        """
        # X축을 중심으로 180도 회전 (토끼를 뒤집기)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        # 포인트 클라우드 회전
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
        
        return pcd
            
    def load_xyz_file(self, filepath):
        """
        XYZ 파일에서 포인트 클라우드 데이터를 로드합니다.
        
        Args:
            filepath (str): XYZ 파일 경로
        
        Returns:
            tuple: (포인트 리스트, Open3D 포인트 클라우드 객체)
        """
        try:
            points = []
            with open(filepath, 'r') as f:
                for line in f:
                    coords = line.strip().split()
                    if len(coords) >= 3:
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        points.append([x, y, z])
            
            # Open3D 포인트 클라우드 객체 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            
            return points, pcd
            
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {filepath}")
            return None, None
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return None, None
    
    def start_visualization(self):
        """
        시각화를 시작합니다.
        """
        if not self.point_clouds:
            print("로드된 포인트 클라우드가 없습니다.")
            return
        
        # 별도 스레드에서 시각화 실행
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self.visualize_all_point_clouds)
            thread.daemon = True
            thread.start()
    
    def visualize_all_point_clouds(self):
        """
        모든 포인트 클라우드를 별도 창에서 시각화합니다.
        """
        try:
            print("3D 시각화 창들을 생성 중...")
            
            # 각 포인트 클라우드에 대해 별도 창 생성
            window_positions = {
                'original': (50, 50),
                'noisy': (900, 50),
                'denoised': (1750, 50)
            }
            
            for pcd_type, pcd_info in self.point_clouds.items():
                try:
                    # 시각화 창 생성
                    vis = o3d.visualization.Visualizer()
                    window_name = f"3D Point Cloud - {pcd_info['name']}"
                    vis.create_window(window_name, width=800, height=600, 
                                    left=window_positions[pcd_type][0], 
                                    top=window_positions[pcd_type][1])
                    
                    # 포인트 클라우드 추가
                    vis.add_geometry(pcd_info['pcd'])
                    
                    # 뷰포인트 설정
                    ctr = vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_lookat([0, 0, 0])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.8)
                    
                    # 시각화 객체 저장
                    self.visualizers[pcd_type] = vis
                    
                    print(f"✓ {pcd_info['name']} 창 생성 완료")
                    
                except Exception as e:
                    print(f"✗ {pcd_info['name']} 창 생성 실패: {e}")
            
            print("\n3D 시각화 창들이 모두 열렸습니다!")
            print("사용법:")
            print("- 각 창에서 마우스 드래그: 회전")
            print("- 각 창에서 마우스 휠: 확대/축소")
            print("- 각 창에서 Shift + 드래그: 이동")
            print("- 창 닫기: 해당 창만 종료")
            print("- GUI 종료 버튼: 모든 창 종료")
            
            # 렌더링 루프
            while self.is_running:
                # 모든 창의 이벤트 처리
                all_closed = True
                for pcd_type, vis in self.visualizers.items():
                    if vis is not None:
                        if vis.poll_events():
                            vis.update_renderer()
                            all_closed = False
                        else:
                            # 창이 닫힌 경우
                            vis.destroy_window()
                            self.visualizers[pcd_type] = None
                
                # 모든 창이 닫혔으면 종료
                if all_closed:
                    break
                
                time.sleep(0.01)
            
            # 모든 창 정리
            for vis in self.visualizers.values():
                if vis is not None:
                    vis.destroy_window()
            
            self.visualizers.clear()
            
        except Exception as e:
            print(f"시각화 오류: {e}")
        finally:
            self.is_running = False
    
    def quit_app(self):
        """
        애플리케이션을 종료합니다.
        """
        self.is_running = False
        
        # 모든 시각화 창 닫기
        for vis in self.visualizers.values():
            if vis is not None:
                vis.destroy_window()
        
        self.visualizers.clear()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """
        애플리케이션을 실행합니다.
        """
        print("3D Point Cloud Visualizer 시작...")
        print("=" * 50)
        print("사용 가능한 포인트 클라우드:")
        for pcd_type, info in self.point_clouds.items():
            print(f"- {pcd_type}: {info['name']} ({info['points']} 포인트)")
        print("=" * 50)
        
        # GUI 실행
        self.root.mainloop()

def main():
    """
    메인 함수
    """
    print("3D Point Cloud Visualization Tool")
    print("=" * 40)
    
    # 시각화 도구 실행
    visualizer = PointCloudVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 