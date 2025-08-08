#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice 1 - Mouse Drawing Example
================================
마우스로 자유롭게 그리는 예제
"""

import cv2
import numpy as np

class MouseDrawing:
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
        self.drawing = False
        self.last_point = None
        self.color = (0, 0, 255)  # 빨간색
        self.thickness = 3
        
        # 창 생성 및 마우스 콜백 설정
        cv2.namedWindow('Mouse Drawing')
        cv2.setMouseCallback('Mouse Drawing', self.mouse_callback)
        
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
            # 왼쪽 버튼 클릭: 그리기 시작
            self.drawing = True
            self.last_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 마우스 이동: 그리기
            if self.drawing and self.last_point is not None:
                cv2.line(self.canvas, self.last_point, (x, y), 
                        self.color, self.thickness, cv2.LINE_AA)
                self.last_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 왼쪽 버튼 해제: 그리기 종료
            self.drawing = False
            self.last_point = None
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 오른쪽 버튼 클릭: 색상 변경
            self.change_color()
            
    def change_color(self):
        """
        색상을 변경합니다.
        """
        colors = [
            (0, 0, 255),    # 빨간색
            (0, 255, 0),    # 초록색
            (255, 0, 0),    # 파란색
            (0, 255, 255),  # 노란색
            (255, 0, 255),  # 마젠타
            (255, 255, 0),  # 시안
            (0, 0, 0),      # 검은색
        ]
        
        current_index = colors.index(self.color)
        next_index = (current_index + 1) % len(colors)
        self.color = colors[next_index]
        
        print(f"색상 변경: {self.get_color_name(self.color)}")
        
    def get_color_name(self, color):
        """
        색상 이름을 반환합니다.
        
        Args:
            color: BGR 색상 튜플
        
        Returns:
            str: 색상 이름
        """
        color_names = {
            (0, 0, 255): "red",
            (0, 255, 0): "green",
            (255, 0, 0): "blue",
            (0, 255, 255): "yellow",
            (255, 0, 255): "magenta",
            (255, 255, 0): "cyan",
            (0, 0, 0): "black",
        }
        return color_names.get(color, "알 수 없는 색상")
    
    def clear_canvas(self):
        """
        캔버스를 초기화합니다.
        """
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        print("캔버스가 초기화되었습니다.")
        
    def draw_instructions(self):
        """
        사용법을 캔버스에 그립니다.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        instructions = [
            "Mouse Drawing Example",
            "Left Click + Drag: drawing",
            "Right Click: change colr",
            "C: clear canvas",
            "ESC: exit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_position = 30 + i * 25
            cv2.putText(self.canvas, instruction, (10, y_position), 
                        font, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
        
        # 현재 색상 표시
        color_name = self.get_color_name(self.color)
        cv2.putText(self.canvas, f"current color: {color_name}", (10, self.height - 20), 
                    font, 0.7, self.color, 2, cv2.LINE_AA)
    
    def run(self):
        """
        메인 실행 루프
        """
        print("=" * 50)
        print("Mouse Drawing Example")
        print("=" * 50)
        print("사용법:")
        print("- 왼쪽 클릭 + 드래그: 그리기")
        print("- 오른쪽 클릭: 색상 변경")
        print("- C: 캔버스 초기화")
        print("- ESC: 종료")
        print("=" * 50)
        
        while True:
            # 사용법 그리기
            self.draw_instructions()
            
            # 캔버스 표시
            cv2.imshow('Mouse Drawing', self.canvas)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('c'):  # C 키
                self.clear_canvas()
        
        cv2.destroyAllWindows()
        print("\n그리기 완료!")

def main():
    """
    메인 함수
    """
    # 마우스 그리기 실행
    drawing = MouseDrawing(800, 600)
    drawing.run()

if __name__ == "__main__":
    main() 