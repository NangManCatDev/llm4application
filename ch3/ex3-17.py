## 모델 학습에 사용할 KLUE YNAT 데이터셋 다운로드
from datasets import load_dataset

# KLUE YNAT 데이터셋 로드
# train: 훈련 데이터셋
# validation: 검증 데이터셋
klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

# 데이터셋 정보 출력
print("훈련 데이터셋 샘플 수: ")
print(klue_tc_train)
print("검증 데이터셋 샘플 수: ")
print(klue_tc_eval)

print("훈련 데이터셋 첫 번째 샘플: ")
print(klue_tc_train[0])


## 사용하지 않는 불필요한 칼럼 제거
# guid: 고유 식별자
# url: 뉴스 기사 URL
# date: 뉴스 기사 날짜
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])

print("제거 후 훈련 데이터셋 샘플 정보: ")
print(klue_tc_train)
# 출력 결과
# Dataset({
#     features: ['id', 'title', 'text', 'label', 'label_names'],
#     num_rows: 10000
# })


## 카테고리를 문자로 표기한 label_str 칼럼 추가
# 레이블 정보 확인
klue_tc_train.features['label']
# ClassLabel(names=['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']), id=None)

# 레이블 인덱스를 문자열로 변환 예시
klue_tc_train.features['label'].int2str(1)
# '경제'

# 레이블 변환을 위한 레이블 객체 저장
klue_tc_label = klue_tc_train.features['label']

# 레이블을 문자열로 변환하는 함수 정의
def make_str_label(batch):
    batch['label_str'] = klue_tc_label.int2str(batch['label'])
    return batch

# 훈련 데이터셋에 문자열 레이블 추가
# batched=True: 배치 단위로 처리
# batch_size=1000: 한 번에 처리할 배치 크기
klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)

print("카테고리 문자로 표기한 label_str 칼럼 추가 후 훈련 데이터셋 첫 번째 샘플: ")
print(klue_tc_train[0])
# 출력 결과
# {'title': '유튜브 내달 2일까지 크리에이터 지원 공간 운영', 'label': 3, 'label_str': '생활문화'}


## 학습/검증/테스트 데이터셋 분할
# 훈련 데이터셋에서 10,000개의 샘플을 테스트 데이터로 분할
# shuffle=True: 데이터를 무작위로 섞음
# seed=42: 무작위성을 고정하여 재현 가능한 결과를 얻음
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']

# 검증 데이터셋을 1,000개의 테스트 데이터와 나머지로 분할
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']

# 나머지 검증 데이터셋에서 다시 1,000개의 샘플을 검증 데이터로 분할
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']


## Trainer API를 사용한 학습: 준비
# 필요한 라이브러리 임포트
import torch
import numpy as np
from transformers import (
    Trainer,              # 모델 학습을 위한 Trainer 클래스
    TrainingArguments,    # 학습 관련 설정을 위한 클래스
    AutoModelForSequenceClassification,  # 텍스트 분류를 위한 자동 모델 클래스
    AutoTokenizer         # 토크나이저 자동 로드 클래스
)

# 텍스트를 토큰으로 변환하는 함수 정의
# padding="max_length": 모든 시퀀스를 최대 길이로 패딩
# truncation=True: 최대 길이를 초과하는 시퀀스를 잘라냄
def tokenize_function(examples):
    return tokenizer(examples["title"], padding="max_length", truncation=True)


# KLUE RoBERTa 모델과 토크나이저 로드
model_id = "klue/roberta-base"
# num_labels: 분류할 클래스의 수 (레이블의 개수)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.features['label'].names))
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 각 데이터셋에 토크나이저 적용
# batched=True: 배치 단위로 처리하여 성능 향상
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


## Trainer API를 사용한 학습: 학습 인자와 평가 함수 정의
# 학습 관련 설정 정의
training_args = TrainingArguments(
    output_dir='./results',          # 모델과 체크포인트 저장 디렉토리
    num_train_epochs=1,              # 전체 데이터셋을 학습하는 횟수
    per_device_train_batch_size=8,   # 학습 시 각 디바이스의 배치 크기
    per_device_eval_batch_size=8,    # 평가 시 각 디바이스의 배치 크기
    evaluation_strategy="epoch",     # 각 에포크마다 모델 평가 수행
    learning_rate=5e-5,              # 학습률 설정
    push_to_hub=False                # Hugging Face Hub에 모델 업로드 여부
)

# 모델 평가를 위한 메트릭 계산 함수 정의
def compute_metrics(eval_pred):
    # eval_pred는 (logits, labels) 형태의 튜플
    logits, labels = eval_pred
    # logits에서 가장 높은 값을 가진 인덱스를 예측값으로 선택
    predictions = np.argmax(logits, axis=-1)
    # 예측값과 실제 레이블이 일치하는 비율을 정확도로 계산
    return {"accuracy": (predictions == labels).mean()}


## Trainer API를 사용한 학습: 학습 진행
# Trainer 객체 생성
# model: 학습할 모델
# args: 학습 설정
# train_dataset: 훈련 데이터셋
# eval_dataset: 검증 데이터셋
# tokenizer: 토크나이저
# compute_metrics: 평가 메트릭 계산 함수
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 모델 학습 시작
trainer.train()

# 테스트 데이터셋으로 모델 평가
result = trainer.evaluate(test_dataset)

# 정확도 출력
print("테스트 데이터셋 정확도: ", result['eval_accuracy'])  # 0.859