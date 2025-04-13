## 모델 아이디로 모델 불러오기
from transformers import AutoModel

model_id = 'klue/roberta-base'
model = AutoModel.from_pretrained(model_id)
