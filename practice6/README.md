# Practice 6 - 특징점과 기하 정렬

## 📋 개요
한 번 실행으로 바로 결과를 보여주는 심화 데모를 제공합니다.

- `feature_matching_orb.py`: ORB 특징점 매칭 데모
- (참고) ICP 정렬 데모는 `practice5/icp_demo.py`를 확인하세요.

## 📁 파일 구조
```
practice6/
├── README.md                 # 이 파일
├── feature_matching_orb.py   # ORB 특징점 매칭 데모
└── data/                     # 데모용 보조 데이터 (선택)
```

## 🚀 실행 방법

### ORB 특징점 매칭
```bash
cd practice6
python feature_matching_orb.py
```

## 🎨 주요 내용

### ORB 특징점 매칭 (`feature_matching_orb.py`)
- 입력 이미지(`data/sample.jpg`가 없으면 `practice3/data/candies.png` 사용)를 한 장 로드
- 스케일/회전/시프트 변환으로 두 번째 이미지를 생성
- ORB로 키포인트/디스크립터 추출 후 BFMatcher(Hamming) + Lowe ratio test로 좋은 매치 선별
- 원본/변환 이미지 사이의 매칭 결과를 한 창으로 시각화

## 🛠️ 요구 사항
`requirements.txt`에 포함된 OpenCV/NumPy로 실행됩니다.

