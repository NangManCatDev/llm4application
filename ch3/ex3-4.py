## 분류 헤드가 포함된 모델 불러오기
from transformers import AutoModelForSequenceClassification

model_id = 'SamLowe/roberta-base-go_emotions'
classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
