# Practice 2 - 카메라 캘리브레이션

## 📋 개요
OpenCV를 사용한 카메라 캘리브레이션 연습

## 🎯 학습 목표
- 카메라의 내부 파라미터 추정
- 이미지 왜곡 보정 방법 학습
- 체커보드 코너 검출 및 정밀화
- 리프로젝션 오차 계산

## 📁 파일 구조
```
practice2/
├── README.md                    # 이 파일
├── calib_example.py            # 카메라 캘리브레이션 예제
└── data/                       # 체커보드 이미지 및 샘플 데이터
    ├── checkerboard_sample.jpg # 캘리브레이션용 체커보드 이미지
    └── checkerboard_info.txt   # 체커보드 사용 가이드
```

## 🚀 실행 방법

### 기본 캘리브레이션
```bash
python calib_example.py
```

## 🎨 주요 기능

### 체커보드 코너 검출
- **코너 검출**: `cv2.findChessboardCorners()` - 체커보드 코너 위치 검출
- **코너 정밀화**: `cv2.cornerSubPix()` - 서브픽셀 정밀도로 코너 위치 정밀화
- **코너 그리기**: `cv2.drawChessboardCorners()` - 검출된 코너 시각화

### 카메라 캘리브레이션
- **카메라 캘리브레이션**: `cv2.calibrateCamera()` - 카메라 내부 파라미터 추정
- **카메라 매트릭스**: 초점 거리, 주점 좌표 추정
- **왜곡 계수**: 방사 왜곡, 접선 왜곡 계수 추정

### 왜곡 보정
- **왜곡 보정**: `cv2.undistort()` - 이미지 왜곡 보정
- **보정된 이미지**: 왜곡이 제거된 깨끗한 이미지 생성
- **전후 비교**: 원본 이미지와 보정된 이미지 비교

### 성능 평가
- **리프로젝션 오차**: 코너 검출 정확도 평가
- **오차 분석**: 평균 오차 및 표준편차 계산
- **결과 시각화**: 오차 분포 그래프 생성

## 🎮 사용법

### 캘리브레이션 실행
1. 프로그램 실행
2. 체커보드 이미지 로드
3. 코너 검출 및 정밀화
4. 카메라 캘리브레이션 수행
5. 왜곡 보정 적용
6. 결과 시각화 및 저장

### 체커보드 준비
- **크기**: 9x6 내부 코너 (10x7 전체)
- **품질**: 선명하고 왜곡이 적은 이미지
- **각도**: 다양한 각도에서 촬영된 이미지들

## 🎯 실습 미션

### 미션 1: 기본 캘리브레이션
1. 체커보드 이미지 준비
2. 코너 검출 및 정밀화
3. 카메라 내부 파라미터 추정
4. 결과 분석

### 미션 2: 왜곡 보정
1. 원본 이미지 로드
2. 왜곡 보정 적용
3. 전후 비교 시각화
4. 보정된 이미지 저장

### 미션 3: 성능 평가
1. 리프로젝션 오차 계산
2. 오차 분포 분석
3. 캘리브레이션 품질 평가
4. 개선 방안 도출

## 🔧 주요 파라미터

### 체커보드 검출
```python
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
```

### 코너 정밀화
```python
cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
```

### 카메라 캘리브레이션
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

### 왜곡 보정
```python
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

## 📊 예제 결과

### 캘리브레이션 결과
- **카메라 매트릭스**: 초점 거리, 주점 좌표
- **왜곡 계수**: 방사 왜곡, 접선 왜곡
- **리프로젝션 오차**: 평균 오차 및 표준편차

### 왜곡 보정 결과
- **원본 이미지**: 왜곡이 있는 체커보드 이미지
- **보정된 이미지**: 왜곡이 제거된 깨끗한 이미지
- **비교 시각화**: 전후 비교를 위한 나란히 배치

## 🛠️ 설치 요구사항

### 필수 패키지
```bash
pip install opencv-python numpy matplotlib
```

## 📚 참고 자료
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) 