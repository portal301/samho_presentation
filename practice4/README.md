# Practice 4 - 이미지 노이즈 제거

## 📋 개요
OpenCV를 사용한 다양한 노이즈 제거 필터의 특성과 사용법 학습

## 🎯 학습 목표
- 다양한 노이즈 제거 필터의 특성 이해
- 노이즈 타입별 최적 필터 선택
- 실시간 필터 비교 및 시각화
- 필터 성능 평가 방법 학습

## 📁 파일 구조
```
practice4/
├── README.md                    # 이 파일
├── filter_comparison.py        # 다양한 필터 비교 예제
└── data/                       # 노이즈가 있는 연습용 이미지
    ├── noise_moon_1.png        # 노이즈가 있는 달 이미지 1
    └── noise_moon_2.png        # 노이즈가 있는 달 이미지 2
```

## 🚀 실행 방법

### 필터 비교 예제
```bash
python filter_comparison.py
```

### 필터별 단일 데모 (z: 원본, x: 필터, ESC: 종료)
```bash
cd practice4
python mean_filter_demo.py        # 평균 필터 (5x5)
python gaussian_filter_demo.py    # 가우시안 필터 (5x5, sigma=0)
python median_filter_demo.py      # 중간값 필터 (5x5)
python bilateral_filter_demo.py   # 양방향 필터 (d=9, sc=75, ss=75)
python morphology_filter_demo.py  # 모폴로지 (open->close, kernel=3x3)
```

## 🎨 주요 기능

### 노이즈 제거 필터
- **평균 필터 (Mean Filter)**: `cv2.blur()` - 주변 픽셀들의 평균값으로 대체
- **가우시안 필터 (Gaussian Filter)**: `cv2.GaussianBlur()` - 가우시안 가중치를 사용한 평균
- **중간값 필터 (Median Filter)**: `cv2.medianBlur()` - 주변 픽셀들의 중간값으로 대체
- **양방향 필터 (Bilateral Filter)**: `cv2.bilateralFilter()` - 공간적 거리와 색상 차이를 모두 고려
- **모폴로지 필터 (Morphological Filter)**: 열기와 닫기 연산 조합

### 실시간 필터 비교
- **키보드 조작**: 0-5번 키로 필터 전환
- **좌우 비교**: noise_moon_1.png와 noise_moon_2.png를 나란히 표시
- **필터 정보**: 현재 적용된 필터 이름 표시
- **실시간 전환**: 키를 누를 때마다 즉시 필터 적용

### 단일 데모 모드
- **z**: 원본 보기
- **x**: 해당 스크립트의 필터 적용 결과 보기
- **ESC**: 종료

### 노이즈 타입별 최적화
- **가우시안 노이즈**: 평균 필터, 가우시안 필터 효과적
- **솔트 앤 페퍼 노이즈**: 중간값 필터, 모폴로지 필터 효과적
- **혼합 노이즈**: 양방향 필터가 가장 효과적

## 🎮 사용법

### 필터 비교 실행
1. 프로그램 실행
2. 기본 화면 (0번): 원본 이미지들 좌우로 표시
3. 필터 적용 (1-5번):
   - **1번**: 평균 필터 적용
   - **2번**: 가우시안 필터 적용
   - **3번**: 중간값 필터 적용
   - **4번**: 양방향 필터 적용
   - **5번**: 모폴로지 필터 적용
4. ESC 키로 종료

### 키보드 조작
- **0**: 원본 이미지 보기
- **1**: 평균 필터 적용
- **2**: 가우시안 필터 적용
- **3**: 중간값 필터 적용
- **4**: 양방향 필터 적용
- **5**: 모폴로지 필터 적용
- **ESC**: 프로그램 종료

## 🎯 실습 미션

### 미션 1: 기본 필터 적용 및 비교
1. 각 필터의 특성 이해
2. 노이즈 타입별 효과 비교
3. 필터 파라미터 조정 실험
4. 결과 시각화 및 분석

### 미션 2: 노이즈 타입별 제거 방법
1. 가우시안 노이즈 제거
2. 솔트 앤 페퍼 노이즈 제거
3. 혼합 노이즈 제거
4. 최적 필터 선택

### 미션 3: 실시간 필터 전환 및 비교
1. 실시간 필터 전환
2. 필터별 성능 비교
3. 노이즈 제거 품질 평가
4. 최적 파라미터 도출

## 🔧 주요 파라미터

### 평균 필터
```python
result = cv2.blur(image, (5, 5))  # 5x5 커널
```

### 가우시안 필터
```python
result = cv2.GaussianBlur(image, (5, 5), 0)  # 5x5 커널, 표준편차 0
```

### 중간값 필터
```python
result = cv2.medianBlur(image, 5)  # 5x5 커널
```

### 양방향 필터
```python
result = cv2.bilateralFilter(image, 9, 75, 75)  # 지름, 색상 표준편차, 공간 표준편차
```

### 모폴로지 필터
```python
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
result = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
```

## 📊 예제 결과

### 필터별 성능 비교
- **평균 필터**: 구현이 간단하고 빠름, 엣지 블러링
- **가우시안 필터**: 평균 필터보다 엣지 보존이 좋음
- **중간값 필터**: 임펄스 노이즈에 매우 효과적
- **양방향 필터**: 엣지를 보존하면서 노이즈 제거
- **모폴로지 필터**: 구조적 노이즈 제거에 효과적

### 노이즈 제거 결과
- **원본 이미지**: 노이즈가 있는 달 이미지들
- **필터링된 이미지**: 각 필터를 적용한 결과
- **비교 시각화**: 좌우로 나란히 배치된 비교 화면

## 🛠️ 설치 요구사항

### 필수 패키지
```bash
pip install opencv-python numpy
```

## 📚 참고 자료
- [OpenCV Smoothing Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- [OpenCV Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) 