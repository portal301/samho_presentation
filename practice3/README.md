# Practice 3 - 색상 공간 및 HSV 처리

## 📋 개요
OpenCV를 사용한 색상 공간 변환과 HSV 색상 기반 객체 검출 연습

## 🎯 학습 목표
- 색상 공간 변환 방법 이해
- HSV 색상 공간의 특성 학습
- 색상 기반 객체 검출 구현
- 색상 마스크 생성 및 활용

## 📁 파일 구조
```
practice3/
├── README.md                    # 이 파일
├── hsv_example.py              # HSV 색상 처리 예제
└── data/                       # 색상별 대표 이미지
    └── candies.png             # 다양한 색상의 사탕 이미지
```

## 🚀 실행 방법

### HSV 색상 처리
```bash
python hsv_example.py
```

## 🎨 주요 기능

### 색상 공간 변환
- **BGR to HSV**: `cv2.cvtColor()` - BGR 색공간에서 HSV 색공간으로 변환
- **색상 공간 이해**: RGB, BGR, HSV 색공간의 차이점
- **색상 분리**: H(색조), S(채도), V(명도) 채널 분리

### HSV 색상 추출
- **색상 범위 설정**: 특정 색상의 HSV 범위 정의
- **마스크 생성**: `cv2.inRange()` - 색상 범위에 따른 마스크 생성
- **노이즈 제거**: 모폴로지 연산으로 마스크 정제

### 색상 기반 객체 검출
- **객체 검출**: 색상 마스크를 이용한 객체 영역 검출
- **윤곽선 검출**: `cv2.findContours()` - 객체 윤곽선 검출
- **객체 분석**: 면적, 중심점, 바운딩 박스 계산

### 실시간 색상 추적
- **웹캠 연동**: 실시간 비디오에서 색상 추적
- **동적 마스크**: 실시간으로 색상 마스크 적용
- **객체 추적**: 검출된 객체의 움직임 추적

## 🎮 사용법

### HSV 색상 처리
1. 프로그램 실행
2. 색상별 HSV 범위 확인
3. 색상 마스크 생성
4. 객체 검출 및 시각화

### 색상 범위 조정
- **빨간색**: H: 0-10, 160-180, S: 100-255, V: 100-255
- **초록색**: H: 35-85, S: 100-255, V: 100-255
- **파란색**: H: 100-130, S: 100-255, V: 100-255
- **노란색**: H: 20-35, S: 100-255, V: 100-255

## 🎯 실습 미션

### 미션 1: 기본 색상 검출
1. HSV 색상 공간 변환
2. 특정 색상 범위 설정
3. 색상 마스크 생성
4. 결과 시각화

### 미션 2: 다중 색상 검출
1. 여러 색상 범위 정의
2. 각 색상별 마스크 생성
3. 색상별 객체 분할
4. 통합 결과 시각화

### 미션 3: 실시간 색상 추적
1. 웹캠 연결
2. 실시간 색상 검출
3. 객체 추적 구현
4. 성능 최적화

## 🔧 주요 파라미터

### 색상 공간 변환
```python
hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
```

### 색상 범위 추출
```python
mask = cv2.inRange(hsv, lower_bound, upper_bound)
```

### 윤곽선 검출
```python
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 모폴로지 연산
```python
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

## 📊 예제 결과

### 색상 검출 결과
- **원본 이미지**: 다양한 색상의 사탕 이미지
- **색상별 마스크**: 각 색상에 대한 이진 마스크
- **검출된 객체**: 색상별로 검출된 객체 영역

### HSV 색상 공간
- **H (Hue)**: 색조 (0-179)
- **S (Saturation)**: 채도 (0-255)
- **V (Value)**: 명도 (0-255)

## 🛠️ 설치 요구사항

### 필수 패키지
```bash
pip install opencv-python numpy matplotlib
```

## 📚 참고 자료
- [OpenCV Color Spaces](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [OpenCV Color Detection](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) 