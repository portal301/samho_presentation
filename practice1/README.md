# Practice 1 - 기본 캔버스 그리기

## 📋 개요
OpenCV를 사용한 기본적인 캔버스 그리기 연습

## 🎯 학습 목표
- OpenCV의 기본 그리기 함수들 익히기
- 마우스 이벤트 처리 방법 학습
- 텍스트 그리기 및 폰트 활용
- Anti-aliasing 옵션 이해

## 📁 파일 구조
```
practice1/
├── README.md                    # 이 파일
├── basic_drawing.py            # 기본 도형 그리기 (원, 직선, 사각형)
├── mouse_drawing.py            # 마우스로 자유롭게 그리기
├── text_drawing.py             # 텍스트 그리기 및 폰트 활용
└── mission.py                  # 미션: 두 점을 잇는 직선과 거리 표시
```

## 🚀 실행 방법

### 기본 도형 그리기
```bash
python basic_drawing.py
```

### 마우스로 그리기
```bash
python mouse_drawing.py
```

### 텍스트 그리기
```bash
python text_drawing.py
```

### 미션 실행
```bash
python mission.py
```

## 🎨 주요 기능

### 기본 도형 그리기 (`basic_drawing.py`)
- **원 그리기**: `cv2.circle()` - 다양한 색상과 두께로 원 그리기
- **직선 그리기**: `cv2.line()` - 두 점을 잇는 직선 그리기
- **사각형 그리기**: `cv2.rectangle()` - 빈 사각형과 채워진 사각형
- **다각형 그리기**: `cv2.polylines()` - 다각형 그리기
- **Anti-aliasing**: 부드러운 선 그리기 옵션

### 마우스 그리기 (`mouse_drawing.py`)
- **마우스 이벤트**: 클릭, 드래그, 릴리즈 이벤트 처리
- **자유 그리기**: 마우스 드래그로 자유롭게 선 그리기
- **색상 변경**: 키보드로 색상 변경 (R, G, B, W, K)
- **지우개 기능**: 마우스 우클릭으로 지우기

### 텍스트 그리기 (`text_drawing.py`)
- **텍스트 그리기**: `cv2.putText()` - 다양한 폰트와 크기
- **폰트 종류**: FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN 등
- **텍스트 속성**: 크기, 두께, 색상 설정
- **배경 박스**: 텍스트 주변에 배경 박스 그리기

### 미션 (`mission.py`)
- **두 점 선택**: 마우스 클릭으로 두 점 선택
- **직선 그리기**: 선택된 두 점을 잇는 직선 그리기
- **거리 계산**: 유클리드 거리 계산
- **거리 표시**: 직선 위에 거리 텍스트 표시

## 🎮 사용법

### 기본 도형 그리기
1. 프로그램 실행
2. 다양한 도형들이 그려진 창 확인
3. ESC 키로 종료

### 마우스 그리기
1. 프로그램 실행
2. 마우스 드래그로 자유롭게 그리기
3. 키보드로 색상 변경:
   - 'R': 빨간색
   - 'G': 초록색
   - 'B': 파란색
   - 'W': 흰색
   - 'K': 검은색
4. 마우스 우클릭으로 지우기
5. ESC 키로 종료

### 텍스트 그리기
1. 프로그램 실행
2. 다양한 폰트와 크기로 그려진 텍스트 확인
3. ESC 키로 종료

### 미션
1. 프로그램 실행
2. 마우스 클릭으로 첫 번째 점 선택
3. 마우스 클릭으로 두 번째 점 선택
4. 두 점을 잇는 직선과 거리가 표시됨
5. ESC 키로 종료

## 🎯 실습 미션

### 미션 1: 기본 도형 그리기
1. `basic_drawing.py` 실행
2. 각 도형의 그리기 방법 이해
3. 색상과 두께 파라미터 실험

### 미션 2: 마우스 이벤트 처리
1. `mouse_drawing.py` 실행
2. 마우스 이벤트 콜백 함수 이해
3. 다양한 색상으로 그리기 실험

### 미션 3: 텍스트 그리기
1. `text_drawing.py` 실행
2. 다양한 폰트와 크기 실험
3. 텍스트 위치 조정 연습

### 미션 4: 종합 미션
1. `mission.py` 실행
2. 두 점 선택 및 직선 그리기
3. 거리 계산 및 표시

## 🔧 주요 파라미터

### 원 그리기
```python
cv2.circle(img, center, radius, color, thickness)
```

### 직선 그리기
```python
cv2.line(img, pt1, pt2, color, thickness)
```

### 사각형 그리기
```python
cv2.rectangle(img, pt1, pt2, color, thickness)
```

### 텍스트 그리기
```python
cv2.putText(img, text, org, font, fontScale, color, thickness)
```

## 🛠️ 설치 요구사항

### 필수 패키지
```bash
pip install opencv-python numpy
```

## 📚 참고 자료
- [OpenCV Drawing Functions](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html)
- [OpenCV Mouse Events](https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html) 