## BERT와 GPT-2 모델을 활용할 때 허깅페이스 트랜스포머 코드 비교
from transformers import AutoModel, AutoTokenizer

text = "What is the huggingface transformer library?"

# BERT 모델 로드
bert_model = AutoModel.from_pretrained("bert-base-uncased") # 모델 불러오기
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # 토크나이저 불러오기
encoded_input = bert_tokenizer(text, return_tensors="pt") # 입력 토큰화
bert_outputs = bert_model(**encoded_input) # 모델 출력

# GPT-2 모델 로드
gpt2_model = AutoModel.from_pretrained("gpt2") # 모델 불러오기
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2") # 토크나이저 불러오기
encoded_input = gpt2_tokenizer(text, return_tensors="pt") # 입력 토큰화
gpt2_outputs = gpt2_model(**encoded_input) # 모델 출력

# 모델 출력 비교
print("BERT 모델 출력:")
print(bert_outputs)
print("\nGPT-2 모델 출력:")
print(gpt2_outputs)


