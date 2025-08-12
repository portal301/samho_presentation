# Practice 5 - 3D Point Cloud Processing

## 📋 개요
Open3D를 사용한 3D 포인트 클라우드 처리 및 디노이징 연습

## 🎯 학습 목표
- 3D 포인트 클라우드 로드 및 시각화
- 포인트 클라우드 디노이징 파이프라인 구현
- **점 분포 균일화 및 최적화** ⭐
- 성능 평가 및 결과 분석
- 3개 창 동시 시각화를 통한 비교

## 📁 파일 구조
```
practice5/
├── README.md                    # 이 파일
├── data/                        # 3D 포인트 클라우드 데이터
│   ├── bunny.xyz                # 깨끗한 버니 포인트 클라우드
│   ├── bunny_noisy.xyz          # 노이즈가 있는 버니 포인트 클라우드
│   └── denoised_bunny.xyz       # 디노이징된 버니 포인트 클라우드
├── point_cloud_loader.py        # 포인트 클라우드 로더
├── denoising_processor.py       # 디노이징 프로세서 (점 분포 균일화 포함) ⭐
├── performance_evaluator.py     # 성능 평가기
├── main.py              # 디노이즈 메인 실행 파일
└── visualization_tools.py       # 3D 시각화 도구 (3개 창 동시 표시)
```

## 🚀 실행 방법

### 3D 시각화 도구 (추천) ⭐
```bash
cd practice5
python visualization_tools.py
```

### 개별 모듈 실행
```bash
cd practice5
python point_cloud_loader.py      # 포인트 클라우드 로더
python denoising_processor.py     # 디노이징 프로세서 (균일화 포함)
python performance_evaluator.py   # 성능 평가기
python visualization_tools.py     # 3개 창 시각화
python rotate_bunny_axes.py       # 버니 좌표축 회전 (예: z축 45도)
python shift_bunny_z.py           # 버니 z축 이동/중앙정렬
python icp_demo.py                # 전역정합→ICP 정렬 데모
```

## 🎨 주요 기능

### 3D 시각화 도구 (`visualization_tools.py`) ⭐
- **3개 창 동시 표시**: 원본, 노이즈, 디노이징된 버니를 별도 창에 표시
- **창 위치 최적화**: 화면에 겹치지 않도록 배치
- **독립적 조작**: 각 창을 독립적으로 회전, 확대/축소, 이동
- **실시간 비교**: 3개 창을 동시에 조작하여 비교

### 포인트 클라우드 로더 (`point_cloud_loader.py`)
- **XYZ 파일 로드**: `.xyz` 형식 포인트 클라우드 파일 로드
- **데이터 검증**: 파일 존재 여부 및 포인트 수 확인
- **Open3D 변환**: NumPy 배열을 Open3D 포인트 클라우드 객체로 변환
- **오류 처리**: 파일 로드 실패 시 적절한 오류 메시지

### 디노이징 프로세서 (`denoising_processor.py`) ⭐
- **복셀 다운샘플링**: `voxel_downsampling()` - 점 분포 균일화 및 노이즈 감소
- **균일 다운샘플링**: `uniform_downsampling()` - 일정한 간격으로 포인트 선택
- **통계적 아웃라이어 제거**: `remove_statistical_outliers()` - 이상치 포인트 제거
- **법선 벡터 계산**: `estimate_normals()` - 표면 방향 추정
- **DBSCAN 클러스터링**: `dbscan_clustering()` - 노이즈 제거
- **균일 점 분포**: `uniform_point_distribution()` - 목표 밀도에 맞춘 균일 분포 생성
- **통합 디노이징**: `denoise_point_cloud()` - 전체 파이프라인 실행 (균일화 포함)

### 성능 평가기 (`performance_evaluator.py`)
- **ICP 정렬**: `align_point_clouds()` - 포인트 클라우드 정렬
- **거리 계산**: `calculate_point_distances()` - 포인트 간 거리 측정
- **성능 평가**: `evaluate_denoising_performance()` - 개선률 계산
- **결과 요약**: `print_performance_summary()` - 성능 지표 출력

### 결과 저장기 (`result_saver.py`)
- **XYZ 파일 저장**: `save_point_cloud_to_xyz()` - 결과를 XYZ 파일로 저장
- **통계 정보**: `print_statistics()` - 포인트 수 및 처리 시간 출력
- **성능 보고서**: `save_performance_report()` - 상세한 성능 보고서 생성
- **요약 보고서**: `create_summary_report()` - 종합적인 요약 보고서

## 🎮 사용법

### 3D 시각화 도구 실행
1. **프로그램 실행**: GUI 창이 열립니다
2. **"3D 시각화 시작" 버튼 클릭**: 3개의 3D 창이 동시에 열립니다
3. **각 창 독립 조작**:
   - **마우스 드래그**: 3D 모델 회전
   - **마우스 휠**: 확대/축소
   - **Shift + 드래그**: 이동
4. **창 관리**:
   - 개별 창 닫기: 해당 창만 종료
   - GUI 종료 버튼: 모든 창 동시 종료

### 창 배치
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Original Bunny │  │   Noisy Bunny   │  │ Denoised Bunny  │
│                 │  │                 │  │                 │
│   (왼쪽 창)     │  │   (중앙 창)     │  │   (오른쪽 창)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🎯 실습 미션

### 미션 1: 3D 시각화 및 비교
1. 3D 시각화 도구 실행
2. 3가지 포인트 클라우드 전환하며 비교
3. 시각적 차이점 관찰
4. 포인트 수 변화 확인

### 미션 2: 기본 포인트 클라우드 처리
1. 포인트 클라우드 데이터 로드
2. 기본 통계 정보 출력
3. 3D 시각화
4. 결과 저장

### 미션 3: 디노이징 파이프라인 구현
1. 복셀 다운샘플링 적용 (점 분포 균일화)
2. 아웃라이어 제거
3. 법선 벡터 계산
4. 클러스터링 기반 노이즈 제거

### 미션 4: 점 분포 균일화 실험 ⭐
1. 균일 다운샘플링 적용
2. 목표 밀도 설정 및 조정
3. 균일 분포 결과 확인
4. 성능 비교 분석

### 미션 5: 성능 평가 시스템
1. ICP를 사용한 포인트 클라우드 정렬
2. 거리 기반 성능 측정
3. 개선률 계산
4. 결과 보고서 생성

## 🔧 주요 파라미터

### 복셀 다운샘플링 (점 분포 균일화)
```python
voxel_size = 0.05  # 복셀 크기 (작을수록 더 많은 포인트 유지)
```

### 균일 다운샘플링
```python
every_k_points = 5  # k개마다 하나의 포인트 선택
```

### 균일 점 분포
```python
target_density = 1000  # 목표 포인트 밀도
```

### 아웃라이어 제거
```python
nb_neighbors = 20  # 이웃 포인트 수
std_ratio = 2.0    # 표준편차 비율
```

### DBSCAN 클러스터링
```python
eps = 0.05         # 이웃 포인트 간 최대 거리
min_points = 5     # 클러스터로 인정할 최소 포인트 수
```

### ICP 정렬
```python
max_distance = 0.1  # 최대 거리 임계값
```

## 📈 성능 메트릭

### 개선률 계산
```python
improvement = ((original_distance - denoised_distance) / original_distance) * 100
```

### 포인트 수 변화
```python
reduction_rate = ((original_points - denoised_points) / original_points) * 100
```

### 점 분포 균일성
```python
uniformity_score = calculate_uniformity(denoised_pcd)
```

## 📊 예제 결과

### 3D 시각화 결과
- **원본 버니**: 깨끗하고 매끄러운 표면
- **노이즈 버니**: 점들이 불규칙하게 분포
- **디노이징된 버니**: 노이즈가 제거되고 균일하게 분포된 매끄러운 표면

### 디노이징 성능
- 원본 노이즈 포인트 → 디노이징된 포인트
- 개선률 계산 및 시각화
- 처리 시간 및 통계 정보
- 점 분포 균일성 향상

### 점 분포 균일화 효과 ⭐
- **복셀 다운샘플링**: 밀도가 높은 영역의 포인트를 줄여서 균일한 분포 생성
- **균일 다운샘플링**: 일정한 간격으로 포인트를 선택하여 균일한 분포 생성
- **목표 밀도 조정**: 사용자가 원하는 포인트 밀도로 조정 가능

## 🛠️ 설치 요구사항

### 필수 패키지
```bash
pip install open3d numpy matplotlib
```

## 📚 참고 자료
- [Open3D Documentation](http://www.open3d.org/docs/)
- [Open3D Python API](http://www.open3d.org/docs/release/python_api/)
- [Point Cloud Processing Tutorial](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) 