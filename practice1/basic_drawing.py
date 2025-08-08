#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice 1 - Basic Drawing Example
================================
OpenCV를 사용한 기본적인 도형 그리기 예제
"""

import cv2
import numpy as np

def create_canvas(width=800, height=600):
    """
    빈 캔버스를 생성합니다.
    
    Args:
        width (int): 캔버스 너비
        height (int): 캔버스 높이
    
    Returns:
        numpy.ndarray: 흰색 배경의 캔버스
    """
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def draw_basic_shapes():
    """
    기본 도형들을 그리는 예제
    """
    print("기본 도형 그리기 예제")
    print("=" * 30)
    
    # 캔버스 생성
    canvas = create_canvas(800, 600)
    
    # 1. 원 그리기
    print("1. 원 그리기")
    # 빨간색 원 (안티앨리어싱 적용)
    cv2.circle(canvas, (200, 150), 50, (0, 0, 255), 3, cv2.LINE_AA)
    # 파란색 원 (채워진 원)
    cv2.circle(canvas, (400, 150), 50, (255, 0, 0), -1, cv2.LINE_AA)
    
    # 2. 직선 그리기
    print("2. 직선 그리기")
    # 초록색 직선 (안티앨리어싱 적용)
    cv2.line(canvas, (100, 300), (300, 300), (0, 255, 0), 5, cv2.LINE_AA)
    # 노란색 대각선
    cv2.line(canvas, (400, 250), (600, 350), (0, 255, 255), 3, cv2.LINE_AA)
    
    # 3. 사각형 그리기
    print("3. 사각형 그리기")
    # 보라색 사각형 (테두리만)
    cv2.rectangle(canvas, (100, 400), (250, 500), (255, 0, 255), 2, cv2.LINE_AA)
    # 주황색 사각형 (채워진 사각형)
    cv2.rectangle(canvas, (300, 400), (450, 500), (0, 165, 255), -1, cv2.LINE_AA)
    
    # 4. 다각형 그리기
    print("4. 다각형 그리기")
    # 삼각형
    triangle_points = np.array([[600, 400], [550, 500], [650, 500]], np.int32)
    cv2.polylines(canvas, [triangle_points], True, (255, 0, 0), 3, cv2.LINE_AA)
    
    # 육각형
    hexagon_points = np.array([
        [600, 150], [550, 200], [550, 250], 
        [600, 300], [650, 250], [650, 200]
    ], np.int32)
    cv2.polylines(canvas, [hexagon_points], True, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 5. 텍스트 그리기
    print("5. 텍스트 그리기")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # 제목
    cv2.putText(canvas, "Basic Drawing Example", (50, 50), 
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # 도형별 설명
    cv2.putText(canvas, "Circle", (150, 220), 
                font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Line", (150, 330), 
                font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Rectangle", (100, 530), 
                font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Polygon", (550, 530), 
                font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    return canvas

def main():
    """
    메인 함수
    """
    print("OpenCV 기본 도형 그리기 예제")
    print("=" * 40)
    
    # 기본 도형 그리기
    canvas = draw_basic_shapes()
    
    # 결과 표시
    cv2.imshow('Basic Drawing Example', canvas)
    print("\n창을 닫으려면 아무 키나 누르세요...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n예제 완료!")

if __name__ == "__main__":
    main() 