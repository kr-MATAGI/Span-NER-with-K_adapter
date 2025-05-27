# Span-NER with K-adapter

한국어 Span 기반 개체명 인식을 위한 K-adapter 구현 프로젝트입니다.

## 프로젝트 소개

이 프로젝트는 KoELECTRA를 기반으로 한 한국어 개체명 인식(Named Entity Recognition) 모델입니다. Span 기반의 접근 방식과 K-adapter 구조를 활용하여 더 효과적인 개체명 인식을 수행합니다.

### 주요 특징

- KoELECTRA 기반 모델 사용
- Span 기반의 개체명 인식 접근
- K-adapter 구조 활용
- Mecab 토크나이저 지원

## 환경 설정

### 요구사항

- Python 3.6+
- PyTorch
- Transformers
- numpy
- tqdm
- attrdict

### 설치 방법

```bash
git clone https://github.com/[username]/Span-NER-with-K_adapter.git
cd Span-NER-with-K_adapter
pip install -r requirements.txt
```

## 사용 방법

### 학습

1. 설정 파일 수정
   `CONFIG.json` 파일에서 필요한 파라미터를 설정합니다.

2. 학습 실행
   ```bash
   python run_dp.py
   ```

### 주요 설정 파라미터

- `model_type`: "monologg/koelectra-base-v3-discriminator"
- `max_seq_len`: 128
- `train_batch_size`: 32
- `eval_batch_size`: 128
- `learning_rate`: 5e-5
- `num_train_epochs`: 10

## 프로젝트 구조

```
.
├── model/              # 모델 관련 코드
├── token_utils/        # 토큰 처리 유틸리티
├── definition/         # 태그 정의 파일
├── CONFIG.json         # 설정 파일
├── run_dp.py          # 메인 실행 파일
├── run_rc.py          # 관계 분류 실행 파일
├── datasets.py        # 데이터셋 처리
├── tag_def.py         # 태그 정의
└── utils.py           # 유틸리티 함수
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 공개되어 있습니다. 