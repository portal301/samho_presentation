#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice 4 - Filter Comparison Example
====================================
data 폴더의 노이즈 이미지에 다양한 OpenCV 필터를 적용하고 결과를 비교하는 예제
"""

import cv2
import numpy as np
import os

def load_noise_images():
    """
    data 폴더에서 노이즈 이미지들을 로드합니다.
    
    Returns:
        tuple: (noise_moon_1.png, noise_moon_2.png)
    """
    # 이미지 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(script_dir, "data/noise_moon_1.png")
    img2_path = os.path.join(script_dir, "data/noise_moon_2.png")

    # 이미지 로드
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        print(f"오류: {img1_path} 파일을 찾을 수 없습니다.")
        return None, None
    
    if img2 is None:
        print(f"오류: {img2_path} 파일을 찾을 수 없습니다.")
        return None, None
    
    print(f"✓ {img1_path} 로드 완료: {img1.shape}")
    print(f"✓ {img2_path} 로드 완료: {img2.shape}")
    
    return img1, img2

def apply_filter_to_image(image, filter_type):
    """
    이미지에 특정 필터를 적용합니다.
    
    Args:
        image: 입력 이미지
        filter_type: 필터 타입 (1-5)
    
    Returns:
        numpy.ndarray: 필터링된 이미지
    """
    if filter_type == 1:
        # 평균 필터 (Mean Filter)
        return cv2.blur(image, (5, 5))
    
    elif filter_type == 2:
        # 가우시안 필터 (Gaussian Filter)
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    elif filter_type == 3:
        # 중간값 필터 (Median Filter)
        return cv2.medianBlur(image, 5)
    
    elif filter_type == 4:
        # 양방향 필터 (Bilateral Filter)
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif filter_type == 5:
        # 모폴로지 필터 (Morphological Filter)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    else:
        # 필터 적용하지 않음 (원본 반환)
        return image.copy()

def get_filter_name(filter_type):
    """
    필터 타입에 따른 이름을 반환합니다.
    
    Args:
        filter_type: 필터 타입 (0-5)
    
    Returns:
        str: 필터 이름
    """
    filter_names = {
        0: "original image",
        1: "mean filter",
        2: "gaussian filter",
        3: "median filter",
        4: "bilateral filter",
        5: "morphological filter"
    }
    return filter_names.get(filter_type, "알 수 없는 필터")

def create_combined_image(img1, img2, filter_type=0):
    """
    두 이미지를 좌우로 나란히 배치하여 하나의 이미지로 만듭니다.
    
    Args:
        img1: 왼쪽 이미지
        img2: 오른쪽 이미지
        filter_type: 필터 타입 (0-5)
    
    Returns:
        numpy.ndarray: 결합된 이미지
    """
    # 이미지 크기 맞추기
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    
    # 더 큰 높이에 맞춰 리사이즈
    max_height = max(height1, height2)
    
    if height1 != max_height:
        img1 = cv2.resize(img1, (int(width1 * max_height / height1), max_height))
    if height2 != max_height:
        img2 = cv2.resize(img2, (int(width2 * max_height / height2), max_height))
    
    # 필터 적용
    if filter_type > 0:
        img1_filtered = apply_filter_to_image(img1, filter_type)
        img2_filtered = apply_filter_to_image(img2, filter_type)
    else:
        img1_filtered = img1.copy()
        img2_filtered = img2.copy()
    
    # 이미지들을 좌우로 결합
    combined = np.hstack([img1_filtered, img2_filtered])
    
    # 텍스트 추가
    filter_name = get_filter_name(filter_type)
    cv2.putText(combined, f"Left: noise_moon_1", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"Right: noise_moon_2", (img1_filtered.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 필터 정보 추가
    cv2.putText(combined, f"Filter: {filter_name}", (10, combined.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return combined

def print_usage_info():
    """
    사용법 정보를 출력합니다.
    """
    print("=" * 60)
    print("노이즈 제거 필터 비교 도구")
    print("=" * 60)
    print("사용법:")
    print("- '0': 원본 이미지 보기")
    print("- '1': 평균 필터 적용")
    print("- '2': 가우시안 필터 적용")
    print("- '3': 중간값 필터 적용")
    print("- '4': 양방향 필터 적용")
    print("- '5': 모폴로지 필터 적용")
    print("- ESC: 종료")
    print("=" * 60)

def print_filter_info():
    """
    각 필터에 대한 정보를 출력합니다.
    """
    print("\n" + "=" * 60)
    print("필터 정보")
    print("=" * 60)
    
    filter_info = {
        1: {
            "이름": "평균 필터 (Mean Filter)",
            "설명": "주변 픽셀들의 평균값으로 대체",
            "장점": "구현이 간단하고 빠름",
            "단점": "엣지가 블러되고 세부사항 손실",
            "적합한 노이즈": "가우시안 노이즈"
        },
        2: {
            "이름": "가우시안 필터 (Gaussian Filter)",
            "설명": "가우시안 가중치를 사용한 평균",
            "장점": "평균 필터보다 엣지 보존이 좋음",
            "단점": "여전히 일부 블러링 발생",
            "적합한 노이즈": "가우시안 노이즈"
        },
        3: {
            "이름": "중간값 필터 (Median Filter)",
            "설명": "주변 픽셀들의 중간값으로 대체",
            "장점": "임펄스 노이즈에 매우 효과적",
            "단점": "계산 비용이 높음",
            "적합한 노이즈": "솔트 앤 페퍼 노이즈"
        },
        4: {
            "이름": "양방향 필터 (Bilateral Filter)",
            "설명": "공간적 거리와 색상 차이를 모두 고려",
            "장점": "엣지를 보존하면서 노이즈 제거",
            "단점": "계산 비용이 매우 높음",
            "적합한 노이즈": "모든 노이즈 타입"
        },
        5: {
            "이름": "모폴로지 필터 (Morphological Filter)",
            "설명": "열기와 닫기 연산 조합",
            "장점": "구조적 노이즈 제거에 효과적",
            "단점": "이미지 구조에 의존적",
            "적합한 노이즈": "구조적 노이즈"
        }
    }
    
    for filter_type, info in filter_info.items():
        print(f"\n필터 {filter_type}: {info['이름']}")
        print(f"  설명: {info['설명']}")
        print(f"  장점: {info['장점']}")
        print(f"  단점: {info['단점']}")
        print(f"  적합한 노이즈: {info['적합한 노이즈']}")

def main():
    """
    메인 함수
    """
    print("노이즈 제거 필터 비교 예제")
    print("=" * 40)
    
    # 1. 노이즈 이미지 로드
    print("1단계: 노이즈 이미지 로드")
    img1, img2 = load_noise_images()
    
    if img1 is None or img2 is None:
        print("이미지 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 2. 필터 정보 출력
    print_filter_info()
    
    # 3. 사용법 정보 출력
    print_usage_info()
    
    # 4. OpenCV 창 생성
    window_name = "Noise Removal Filter Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    current_filter = 0  # 0: 원본, 1-5: 필터
    
    while True:
        # 현재 필터로 이미지 생성
        combined_image = create_combined_image(img1, img2, current_filter)
        
        # 이미지 표시
        cv2.imshow(window_name, combined_image)
        
        # 키 입력 처리
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('0'):
            current_filter = 0
            print("원본 이미지 표시")
        elif key == ord('1'):
            current_filter = 1
            print("평균 필터 적용")
        elif key == ord('2'):
            current_filter = 2
            print("가우시안 필터 적용")
        elif key == ord('3'):
            current_filter = 3
            print("중간값 필터 적용")
        elif key == ord('4'):
            current_filter = 4
            print("양방향 필터 적용")
        elif key == ord('5'):
            current_filter = 5
            print("모폴로지 필터 적용")
        elif key == ord('h') or key == ord('?'):
            print_usage_info()
    
    cv2.destroyAllWindows()
    print("\n프로그램 종료!")

if __name__ == "__main__":
    main() 