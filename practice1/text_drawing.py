#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice 1 - Text Drawing Example
================================
OpenCV를 사용한 텍스트 그리기 예제
"""

import cv2
import numpy as np

def create_text_canvas(width=800, height=600):
    """
    텍스트 그리기용 캔버스를 생성합니다.
    
    Args:
        width (int): 캔버스 너비
        height (int): 캔버스 높이
    
    Returns:
        numpy.ndarray: 흰색 배경의 캔버스
    """
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def draw_text_examples():
    """
    다양한 텍스트 그리기 예제
    """
    print("텍스트 그리기 예제")
    print("=" * 30)
    
    # 캔버스 생성
    canvas = create_text_canvas(800, 600)
    
    # 1. 기본 텍스트 그리기
    print("1. 기본 텍스트 그리기")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Hello, OpenCV!", (50, 100), 
                font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 2. 다양한 폰트 사용
    print("2. 다양한 폰트 사용")
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    ]
    
    font_names = [
        "SIMPLEX", "PLAIN", "DUPLEX", "COMPLEX",
        "TRIPLEX", "COMPLEX_SMALL", "SCRIPT_SIMPLEX", "SCRIPT_COMPLEX"
    ]
    
    for i, (font, name) in enumerate(zip(fonts, font_names)):
        y_position = 150 + i * 30
        cv2.putText(canvas, f"{name}: Sample Text", (50, y_position), 
                    font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    
    # 3. 다양한 색상과 크기
    print("3. 다양한 색상과 크기")
    colors = [
        (0, 0, 255),    # 빨간색
        (0, 255, 0),    # 초록색
        (255, 0, 0),    # 파란색
        (0, 255, 255),  # 노란색
        (255, 0, 255),  # 마젠타
        (255, 255, 0),  # 시안
    ]
    
    sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for i, (color, size) in enumerate(zip(colors, sizes)):
        x_position = 400
        y_position = 150 + i * 50
        cv2.putText(canvas, f"Size: {size}", (x_position, y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
    
    # 4. 텍스트 배경 박스
    print("4. 텍스트 배경 박스")
    text = "Background Box"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 배경 박스 그리기
    x, y = 50, 450
    cv2.rectangle(canvas, 
                 (x - 10, y - text_height - 10),
                 (x + text_width + 10, y + baseline + 10),
                 (200, 200, 200), -1)
    
    # 텍스트 그리기
    cv2.putText(canvas, text, (x, y), 
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # 5. 회전된 텍스트 (수동 구현)
    print("5. 회전된 텍스트")
    text = "Rotated Text"
    x, y = 400, 450
    
    # 회전 행렬 생성
    angle = 45  # 45도 회전
    center = (x, y)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.0, 2)
    
    # 텍스트를 이미지로 변환
    text_img = np.zeros((text_height + 20, text_width + 20, 3), dtype=np.uint8)
    cv2.putText(text_img, text, (10, text_height + 10), 
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 회전 적용
    rotated_text = cv2.warpAffine(text_img, rotation_matrix, 
                                 (canvas.shape[1], canvas.shape[0]))
    
    # 회전된 텍스트를 캔버스에 합성
    mask = rotated_text > 0
    canvas[mask] = rotated_text[mask]
    
    # 6. 다국어 텍스트 (한글은 제한적)
    print("6. 다국어 텍스트")
    cv2.putText(canvas, "OpenCV Text Drawing", (50, 520), 
                font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Computer Vision", (50, 550), 
                font, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
    
    return canvas

def main():
    """
    메인 함수
    """
    print("OpenCV 텍스트 그리기 예제")
    print("=" * 40)
    
    # 텍스트 그리기 예제
    canvas = draw_text_examples()
    
    # 결과 표시
    cv2.imshow('Text Drawing Example', canvas)
    print("\n창을 닫으려면 아무 키나 누르세요...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n예제 완료!")

if __name__ == "__main__":
    main() 