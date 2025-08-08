#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice 1 - Mission: Two Points and Distance
===========================================
미션: 두 점을 찍고, 두 점을 잇는 직선과 직선거리를 텍스트로 표시하기
"""

import cv2
import numpy as np
import math

class PointDistanceCalculator:
    def __init__(self, width=800, height=600):
        """
        초기화
        
        Args:
            width (int): 캔버스 너비
            height (int): 캔버스 높이
        """
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.points = []
        self.drawing = False
        
        # 창 생성 및 마우스 콜백 설정
        cv2.namedWindow('Two Points Distance Mission')
        cv2.setMouseCallback('Two Points Distance Mission', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        마우스 이벤트 콜백 함수
        
        Args:
            event: 마우스 이벤트 타입
            x, y: 마우스 좌표
            flags: 추가 플래그
            param: 추가 파라미터
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 왼쪽 버튼 클릭: 점 추가
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"점 {len(self.points)} 추가: ({x}, {y})")
                self.draw_points_and_line()
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 오른쪽 버튼 클릭: 초기화
            self.reset()
            
    def calculate_distance(self, point1, point2):
        """
        두 점 사이의 유클리드 거리를 계산합니다.
        
        Args:
            point1: 첫 번째 점 (x, y)
            point2: 두 번째 점 (x, y)
        
        Returns:
            float: 두 점 사이의 거리
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def draw_points_and_line(self):
        """
        점들과 직선, 거리를 그립니다.
        """
        # 캔버스 초기화
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # 제목 그리기
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.canvas, "Two Points Distance Mission", (50, 50), 
                    font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.canvas, "Left Click: Add Point, Right Click: Reset", (50, 80), 
                    font, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
        
        # 점 그리기
        for i, point in enumerate(self.points):
            x, y = point
            # 점 그리기 (빨간색 원)
            cv2.circle(self.canvas, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
            # 점 번호 표시
            cv2.putText(self.canvas, f"P{i+1}", (x+15, y-15), 
                        font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            # 좌표 표시
            cv2.putText(self.canvas, f"({x}, {y})", (x+15, y+10), 
                        font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        
        # 두 점이 모두 찍혔을 때 직선과 거리 표시
        if len(self.points) == 2:
            point1, point2 = self.points
            
            # 직선 그리기 (파란색)
            cv2.line(self.canvas, point1, point2, (255, 0, 0), 3, cv2.LINE_AA)
            
            # 거리 계산
            distance = self.calculate_distance(point1, point2)
            
            # 거리 텍스트 표시 (중점에)
            mid_x = (point1[0] + point2[0]) // 2
            mid_y = (point1[1] + point2[1]) // 2
            
            # 배경 박스 그리기
            text = f"Distance: {distance:.2f} pixels"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.8, 2)
            cv2.rectangle(self.canvas, 
                         (mid_x - text_width//2 - 10, mid_y - text_height - 10),
                         (mid_x + text_width//2 + 10, mid_y + baseline + 10),
                         (255, 255, 255), -1)
            
            # 거리 텍스트 그리기
            cv2.putText(self.canvas, text, 
                        (mid_x - text_width//2, mid_y + text_height//2), 
                        font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            
            # 결과 출력
            print(f"두 점 사이의 거리: {distance:.2f} 픽셀")
            
        # 현재 상태 표시
        status_text = f"Points: {len(self.points)}/2"
        cv2.putText(self.canvas, status_text, (50, self.height - 30), 
                    font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    def reset(self):
        """
        캔버스를 초기화합니다.
        """
        self.points = []
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        print("캔버스가 초기화되었습니다.")
        self.draw_points_and_line()
    
    def run(self):
        """
        메인 실행 루프
        """
        print("=" * 50)
        print("Two Points Distance Mission")
        print("=" * 50)
        print("사용법:")
        print("- 왼쪽 클릭: 점 추가 (최대 2개)")
        print("- 오른쪽 클릭: 초기화")
        print("- ESC: 종료")
        print("=" * 50)
        
        self.draw_points_and_line()
        
        while True:
            # 캔버스 표시
            cv2.imshow('Two Points Distance Mission', self.canvas)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # R 키로도 초기화 가능
                self.reset()
        
        cv2.destroyAllWindows()
        print("\n미션 완료!")

def main():
    """
    메인 함수
    """
    # 미션 실행
    calculator = PointDistanceCalculator(800, 600)
    calculator.run()

if __name__ == "__main__":
    main() 