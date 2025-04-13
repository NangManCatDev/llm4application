## 토크나이저 불러오기
from transformers import AutoTokenizer

model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
