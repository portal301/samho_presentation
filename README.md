# 컴퓨터비전 튜토리얼

컴퓨터 비전 및 3D 포인트 클라우드 처리 예제 모음

## 🎯 프로젝트 개요

이 프로젝트는 컴퓨터 비전과 3D 포인트 클라우드 처리를 단계별로 학습할 수 있도록 구성된 교육용 예제 모음입니다. 각 practice는 독립적으로 학습할 수 있으며, 기초부터 고급까지 체계적으로 구성되어 있습니다.

## 🐍 호환 가능한 Python 버전

- **Python 3.9** 이상 권장
- **Python 3.10-3.13** 완전 지원

## 📦 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd samho_presentation
```

2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

3. 패키지 설치
```bash
pip install -r requirements.txt
```

## 📁 프로젝트 구조 및 학습 가이드

```
samho_presentation/
├── 📂 practice1/                    # 🎨 기본 캔버스 그리기
│   ├── basic_drawing.py            # 기본 도형 그리기 (원, 직선, 사각형)
│   ├── mouse_drawing.py            # 마우스로 자유롭게 그리기
│   ├── text_drawing.py             # 텍스트 그리기 및 폰트 활용
│   ├── mission.py                  # 미션: 두 점을 잇는 직선과 거리 표시
│   └── README.md                   # 상세한 학습 가이드
│
├── 📂 practice2/                    # 📷 카메라 캘리브레이션
│   ├── data/                       # 체커보드 이미지 및 샘플 데이터
│   │   ├── checkerboard_sample.jpg # 캘리브레이션용 체커보드 이미지
│   │   └── checkerboard_info.txt   # 체커보드 사용 가이드
│   └── README.md                   # 캘리브레이션 학습 목표 및 예제 구조
│
├── 📂 practice3/                    # 🌈 색상 공간 및 HSV 처리
│   ├── data/                       # 색상별 대표 이미지
│   │   └── candies.png             # 다양한 색상의 사탕 이미지
│   ├── hsv_example.py              # HSV 색상 처리 예제
│   └── README.md                   # HSV 색상 처리 학습 가이드
│
├── 📂 practice4/                    # 🔧 이미지 노이즈 제거
│   ├── data/                       # 노이즈가 있는 연습용 이미지
│   │   ├── noise_moon_1.png        # 노이즈가 있는 달 이미지 1
│   │   └── noise_moon_2.png        # 노이즈가 있는 달 이미지 2
│   ├── filter_comparison.py        # 다양한 필터 비교 예제
│   ├── mean_filter_demo.py         # 평균 필터 단일 데모
│   ├── gaussian_filter_demo.py     # 가우시안 필터 단일 데모
│   ├── median_filter_demo.py       # 중간값 필터 단일 데모
│   ├── bilateral_filter_demo.py    # 양방향 필터 단일 데모
│   └── README.md                   # 노이즈 제거 필터 학습 가이드
│
├── 📂 practice5/                    # 🌐 3D 포인트 클라우드 처리
│   ├── data/                       # 3D 포인트 클라우드 데이터
│   │   ├── bunny.xyz               # 깨끗한 버니 포인트 클라우드
│   │   ├── bunny_noisy.xyz         # 노이즈가 있는 버니 포인트 클라우드
│   │   └── denoised_bunny.xyz      # 디노이징된 버니 포인트 클라우드
│   ├── point_cloud_loader.py       # 포인트 클라우드 로더
│   ├── denoising_processor.py      # 디노이징 프로세서
│   ├── visualization_tools.py      # 3D 시각화 도구 (3개 창 동시 표시)
│   ├── rotate_bunny_axes.py        # 버니 좌표축 회전 (예: z축 45도)
│   ├── shift_bunny_z.py            # 버니 z축 이동/중앙정렬
│   ├── icp_demo.py                 # 전역정합+ICP 정렬 데모
│   └── README.md                   # 3D 포인트 클라우드 처리 가이드
│
├── 📂 practice6/                    # 🧭 특징점과 기하 정렬
│   ├── feature_matching_orb.py     # ORB 특징점 매칭 데모 
│   └── README.md                   # Practice 6 가이드
│
├── 📂 data/                         # 공통 데이터 파일
│   ├── bunny.xyz                   # 깨끗한 버니 포인트 클라우드
│   ├── bunny_noisy.xyz             # 노이즈가 있는 버니 포인트 클라우드
│   ├── sample.jpg                  # 캘리브레이션용 샘플 이미지
│   ├── candies.png                 # HSV 예제용 이미지
│   ├── checkerboard_7x5.png        # 7x5 체커보드
│   ├── checkerboard_8x6.png        # 8x6 체커보드
│   └── checkerboard_9x6.png        # 9x6 체커보드
│
├── requirements.txt                 # Python 패키지 목록
└── README.md                       # 프로젝트 설명서 (이 파일)
```

## 🎓 학습 순서 및 내용

### 📂 Practice 1 - 기본 캔버스 그리기 🎨
**학습 목표**: OpenCV의 기본 그리기 함수들을 익히고 마우스 이벤트 처리 방법을 학습합니다.

**주요 내용**:
- `cv2.circle()`, `cv2.line()`, `cv2.rectangle()` 등 기본 도형 그리기
- `cv2.putText()` 텍스트 그리기 및 다양한 폰트 활용
- 마우스 이벤트 콜백 함수 구현
- Anti-aliasing 옵션 이해

**실습 미션**: 
- 두 점을 찍고, 두 점을 잇는 직선과 직선거리를 텍스트로 표시하기

**실행 방법**:
```bash
cd practice1
python basic_drawing.py    # 기본 도형 그리기
python mouse_drawing.py    # 마우스로 그리기
python text_drawing.py     # 텍스트 그리기
python mission.py          # 미션 실행
```

---

### 📂 Practice 2 - 카메라 캘리브레이션 📷
**학습 목표**: 카메라의 내부 파라미터를 추정하고 이미지 왜곡을 보정하는 방법을 학습합니다.

**주요 내용**:
- 체커보드 코너 검출 (`cv2.findChessboardCorners()`)
- 코너 정밀화 (`cv2.cornerSubPix()`)
- 카메라 캘리브레이션 (`cv2.calibrateCamera()`)
- 왜곡 보정 (`cv2.undistort()`)
- 리프로젝션 검증

**실습 미션**:
- 단일 이미지 캘리브레이션 수행
- 왜곡 보정 전후 비교
- 리프로젝션 오차 계산

**실행 방법**:
```bash
cd practice2
python show_detected_corners.py   # 체커보드 코너 검출
python show_reprojected_points.py # 리프로젝션 포인트 시각화
python show_undistorted_image.py  # 왜곡 보정 결과
```

---

### 📂 Practice 3 - 색상 공간 및 HSV 처리 🌈
**학습 목표**: 색상 공간 변환과 HSV를 이용한 색상 기반 객체 검출을 학습합니다.

**주요 내용**:
- BGR에서 HSV 색공간 변환 (`cv2.cvtColor()`)
- 특정 색상 범위 추출 (`cv2.inRange()`)
- 색상 마스크 생성 및 적용
- 색상 기반 객체 검출

**실습 미션**:
- 기본 색상 검출 구현
- 다중 색상 검출 및 분할
- 실시간 색상 추적

**실행 방법**:
```bash
cd practice3
python hsv_example.py      # HSV 색상 추출
```

---

### 📂 Practice 4 - 이미지 노이즈 제거 🔧
**학습 목표**: 다양한 노이즈 제거 필터의 특성과 사용법을 학습합니다.

**주요 내용**:
- 평균 필터, 가우시안 필터, 중간값 필터
- 양방향 필터, 모폴로지 필터
- 노이즈 타입별 최적 필터 선택
- 실시간 필터 비교 및 시각화

**실습 미션**:
- 기본 필터 적용 및 비교
- 노이즈 타입별 제거 방법
- 실시간 필터 전환 및 비교

**실행 방법**:
```bash
cd practice4
python filter_comparison.py    # 필터 비교 예제
```

---

### 📂 Practice 5 - 3D 포인트 클라우드 처리 🌐
**학습 목표**: Open3D를 사용한 3D 포인트 클라우드 처리와 디노이징을 학습합니다.

**주요 내용**:
- 포인트 클라우드 로드 및 시각화
- 복셀 다운샘플링
- 통계적 아웃라이어 제거
- DBSCAN 클러스터링
- ICP 정렬 및 성능 평가
- 3개 창 동시 시각화

**실습 미션**:
- 기본 포인트 클라우드 처리
- 디노이징 파이프라인 구현
- 성능 평가 시스템 구축
- 3D 시각화 및 비교

**실행 방법**:
```bash
cd practice5
python visualization_tools.py  # 3D 시각화 도구 (3개 창)
python point_cloud_loader.py   # 포인트 클라우드 로더
python denoising_processor.py  # 디노이징 프로세서
python performance_evaluator.py # 성능 평가기
python result_saver.py         # 결과 저장기
```

## 🚀 빠른 시작

### 전체 프로젝트 실행
```bash
# 3D 포인트 클라우드 시각화 (3개 창)
cd practice5 && python visualization_tools.py

# 카메라 캘리브레이션 (분리된 단일 데모)
cd practice2 && python show_detected_corners.py
cd practice2 && python show_reprojected_points.py
cd practice2 && python show_undistorted_image.py

# HSV 색상 추출
cd practice3 && python hsv_example.py

# 노이즈 제거 필터 비교
cd practice4 && python filter_comparison.py

# 3D ICP 정렬 데모 (전역정합→ICP)
cd practice5 && python icp_demo.py

# ORB 특징점 매칭
cd practice6 && python feature_matching_orb.py
```

### 개별 Practice 실행
```bash
# Practice 1: 기본 그리기
cd practice1 && python mission.py

# Practice 2: 캘리브레이션 (단일 데모)
cd practice2 && python show_detected_corners.py
cd practice2 && python show_reprojected_points.py
cd practice2 && python show_undistorted_image.py

# Practice 3: 색상 처리
cd practice3 && python hsv_example.py

# Practice 4: 노이즈 제거
cd practice4 && python filter_comparison.py

# Practice 5: 3D 포인트 클라우드 및 ICP 데모
cd practice5 && python visualization_tools.py
cd practice5 && python icp_demo.py

# Practice 6: ORB 특징점 매칭
cd practice6 && python feature_matching_orb.py
```

## 🔧 주요 기능

### 3D 포인트 클라우드 디노이징
- **복셀 다운샘플링**: 노이즈 감소
- **통계적 아웃라이어 제거**: 이상치 포인트 제거
- **법선 벡터 계산**: 표면 방향 추정
- **DBSCAN 클러스터링**: 노이즈 제거
- **성능 평가**: ICP 정렬 및 거리 계산
- **3개 창 동시 시각화**: 원본, 노이즈, 디노이징 결과 비교

### 카메라 캘리브레이션
- 체커보드 코너 검출
- 카메라 내부 파라미터 추정
- 왜곡 보정 (Undistortion)
- 리프로젝션 검증

### HSV 색상 추출
- BGR에서 HSV 색공간 변환
- 특정 색상 범위 추출
- 마스크 생성

### 노이즈 제거 필터
- 평균, 가우시안, 중간값 필터
- 양방향, 모폴로지 필터
- 실시간 필터 전환 및 비교
- 키보드 조작으로 필터 적용

## 📊 예제 결과

### 디노이징 성능
- 원본 노이즈 포인트 → 디노이징된 포인트
- 개선률 계산 및 시각화
- 처리 시간 및 통계 정보
- 3개 창 동시 비교 시각화

### 캘리브레이션 결과
- 카메라 매트릭스 출력
- 왜곡 계수 표시
- 보정된 이미지 저장

### 노이즈 제거 결과
- 다양한 필터 적용 결과 비교
- 실시간 필터 전환
- 노이즈 타입별 최적 필터 확인

## 🛠️ 개발 환경

- **IDE**: VS Code, PyCharm 권장
- **Python**: 3.8+ (가상환경 사용 권장)
- **주요 라이브러리**: Open3D, OpenCV, NumPy, Matplotlib

## 📚 추가 학습 자료

각 practice 폴더의 README.md 파일에서 더 상세한 학습 가이드와 참고 자료를 확인할 수 있습니다.

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.
