## Trainder를 사용하지 않는 학습: (1) 학습을 위한 모델과 토크나이저 준비
import torch
import numpy as np  # 평가 함수에서 numpy를 사용하므로 추가
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer   # 필요한 클래스 추가
from datasets import load_dataset


## Trainer를 사용하지 않는 학습: (2) 데이터셋 준비
# KLUE YNAT 데이터셋 로드
klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

# 불필요한 칼럼 제거
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])

# 훈련/검증/테스트 데이터셋 분할
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']


## Trainder를 사용하지 않는 학습: (3) 모델과 토크나이저 준비
# 모델과 토크나이저 불러오기
# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "klue/roberta-base"

# 텍스트를 토큰으로 변환하는 함수 정의
# padding="max_length": 모든 시퀀스를 최대 길이로 패딩
# truncation=True: 최대 길이를 초과하는 시퀀스를 잘라냄
def tokenize_function(examples):    # 제목(title) 칼럼에 대한 토큰화
    return tokenizer(examples["title"], padding="max_length", truncation=True)

# KLUE RoBERTa 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)

# KLUE RoBERTa 모델 로드
# num_labels: 분류할 클래스의 수 (레이블의 개수)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.features['label'].names))
# 모델을 GPU로 이동
model.to(device)


## Trainer를 사용하지 않는 학습: (4) 학습을 위한 데이터 준비
# 데이터 로더 생성 함수 정의
# dataset: 데이터셋
# batch_size: 배치 크기
# shuffle: 데이터 셔플 여부 (True: 데이터 순서를 무작위로 섞음, False: 원래 순서 유지)
def make_dataloader(dataset, batch_size, shuffle=True):
    # map: 데이터셋의 각 샘플에 함수를 적용
    # batched=True: 배치 단위로 처리하여 성능 향상
    # with_format("torch"): 데이터를 PyTorch 텐서 형식으로 변환
    dataset = dataset.map(tokenize_function, batched=True).with_format("torch")
    # 컬럼 이름 변경 (label -> labels)
    dataset = dataset.rename_column("label", "labels")
    # 불필요한 컬럼 제거
    dataset = dataset.remove_columns(column_names=['title'])
    # DataLoader 생성
    # shuffle=True: 각 에포크마다 데이터 순서를 무작위로 섞음
    # shuffle=False: 데이터 순서를 그대로 유지
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 데이터 로더 만들기
# 훈련 데이터: 셔플 적용 (과적합 방지와 일반화 성능 향상을 위해)
train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
# 검증/테스트 데이터: 셔플 미적용 (일관된 평가를 위해)
valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)


## Trainer를 사용하지 않는 학습: (5) 학습을 위한 함수 정의
# 한 에포크(전체 데이터셋을 한 번 학습하는 과정) 동안 모델을 학습시키는 함수
# 입력:
#   - model: 학습할 모델
#   - data_loader: 학습 데이터를 제공하는 DataLoader
#   - optimizer: 모델 파라미터를 업데이트하는 최적화 알고리즘
def train_epoch(model, data_loader, optimizer):
    # 모델을 학습 모드로 설정 (드롭아웃, 배치 정규화 등이 학습용으로 동작)
    model.train()
    total_loss = 0
    # tqdm: 진행 상황을 시각적으로 표시하는 진행 막대
    for batch in tqdm(data_loader):
        # 이전 배치에서 계산된 그래디언트를 초기화
        # 이는 각 배치마다 독립적으로 그래디언트를 계산하기 위함
        optimizer.zero_grad()
        
        # 입력 데이터를 GPU/CPU 장치로 이동
        input_ids = batch['input_ids'].to(device)   # 모델에 입력할 토큰 아이디
        attention_mask = batch['attention_mask'].to(device)  # 모델에 입력할 어텐션 마스크
        labels = batch['labels'].to(device) # 모델에 입력할 레이블
        
        # 모델에 입력을 전달하여 순전파(forward pass) 수행
        # 이 과정에서 모델은 예측을 생성하고 손실을 계산
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # 손실(loss): 모델의 예측과 실제 레이블 간의 차이를 측정하는 값
        # 낮은 손실 = 더 정확한 예측, 높은 손실 = 부정확한 예측
        # 텍스트 분류에서는 주로 교차 엔트로피 손실(cross-entropy loss)을 사용
        loss = outputs.loss
        
        # 손실에 대한 역전파(backpropagation) 수행
        # 이는 손실 함수를 모델의 각 파라미터에 대해 미분하여
        # 각 파라미터가 손실에 얼마나 기여했는지 계산하는 과정
        # 이 계산된 기울기(그래디언트)는 모델 내부에 저장됨
        loss.backward()
        
        # 모델 업데이트(optimization step)
        # 계산된 그래디언트를 사용하여 모델의 파라미터를 업데이트
        # 이 과정에서 optimizer는 학습률(learning rate)을 적용하고,
        # 그래디언트 방향으로 파라미터를 조정하여 손실을 감소시키는 방향으로 이동
        # AdamW 최적화기는 적응적 학습률과 가중치 감소를 사용하여 더 효과적인 학습을 가능하게 함
        optimizer.step()
        
        # 전체 손실 합계에 현재 배치의 손실 추가
        total_loss += loss.item()
    
    # 전체 에포크의 평균 손실 계산
    # 이 값을 통해 학습이 얼마나 잘 진행되고 있는지 모니터링 가능
    avg_loss = total_loss / len(data_loader)
    return avg_loss


## Trainer를 사용하지 않는 학습: (6) 평가를 위한 함수 정의
# 모델의 성능을 평가하는 함수
# 입력:
#   - model: 평가할 모델
#   - data_loader: 평가 데이터를 제공하는 DataLoader
# 출력:
#   - avg_loss: 평균 손실
#   - accuracy: 정확도 (정확히 예측한 샘플의 비율)
def evaluate(model, data_loader):
    # 모델을 평가 모드로 설정
    # 이 모드에서는 드롭아웃이 적용되지 않고, 배치 정규화가 다르게 동작함
    # 즉, 학습 시의 무작위성을 제거하여 일관된 결과를 얻기 위함
    model.eval()
    total_loss = 0
    predictions = []  # 모델의 예측값을 저장할 리스트
    true_labels = []  # 실제 레이블을 저장할 리스트
    
    # torch.no_grad() 컨텍스트:
    # 평가 과정에서는 역전파가 필요 없으므로 그래디언트 계산을 비활성화
    # 이는 메모리 사용량을 줄이고 계산 속도를 높이는 효과가 있음
    with torch.no_grad():
        # 각 배치에 대한 처리
        for batch in tqdm(data_loader):
            # 입력 데이터를 GPU/CPU 장치로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 모델에 입력을 전달하여 출력을 계산
            # 학습과 동일하게 forward pass를 수행하지만, backward pass는 수행하지 않음
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # logits: 모델이 각 클래스에 대해 예측한 확률 분포(소프트맥스 적용 전)
            # 각 클래스에 대한 점수를 나타내며, 값이 높을수록 해당 클래스에 속할 가능성이 높음
            logits = outputs.logits
            
            # 손실 계산 (학습 과정과 동일하지만, 파라미터 업데이트에는 사용되지 않음)
            loss = outputs.loss
            total_loss += loss.item()
            
            # 예측값 계산: logits에서 가장 큰 값을 가진 인덱스가 예측 클래스
            # dim=-1: 마지막 차원을 따라 최대값을 찾음 (클래스 차원)
            preds = torch.argmax(logits, dim=-1)
            
            # 예측값과 실제 레이블을 CPU로 이동 후 NumPy 배열로 변환하여 저장
            # extend: 리스트에 항목들을 추가
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # 평균 손실 계산
    avg_loss = total_loss / len(data_loader)
    
    # 정확도 계산: 예측값과 실제 레이블이 일치하는 비율
    # np.mean은 배열의 평균을 계산 (True = 1, False = 0 으로 취급)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    return avg_loss, accuracy
            


## Trainer를 사용하지 않는 학습: (7) 학습 수행
# 학습 설정
num_epochs = 1  # 전체 데이터셋을 1회 학습
# AdamW 옵티마이저 초기화
# model.parameters(): 모델의 학습 가능한 모든 파라미터
# lr=5e-5: 학습률 설정 (작은 값으로 설정하여 안정적인 학습)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 학습 루프
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    # 현재 에포크의 학습 수행 및 손실 계산
    train_loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Train Loss: {train_loss:.4f}")
    
    # 검증 데이터를 사용한 모델 평가
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
    print(f"Validation loss: {valid_loss:.4f}")
    print(f"Validation accuracy: {valid_accuracy:.4f}")

# 테스트 데이터를 사용한 최종 모델 평가
# 최종 학습된 모델의 실제 성능 확인
_, test_accuracy = evaluate(model, test_dataloader)
print(f"Test accuracy: {test_accuracy:.4f}")





