## 토크나이저 불러오기
from transformers import AutoTokenizer

model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)


## 토크나이저 사용하기
tokenized = tokenizer("토크나이저는 텍스트를 토크 단위로 나눈다")
print(tokenized)
# {'input_ids': [0, 9157, 7461, 2190, 2259, 8509, 2138, 9157, 5385, 2200, 20950, 2],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))
# ['[CLS]', '토크', '##나이', '##저', '##는', '텍스트', '##를', '토크', '단위', '##로', '나눈다', '[SEP]']

print(tokenizer.decode(tokenized['input_ids']))
# [CLS] 토크나이저는 텍스트를 토크 단위로 나눈다 [SEP]

print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))
# 토크나이저는 텍스트를 토크 단위로 나눈다



## 토크나이저에 여러 문장 넣기
tokenized = tokenizer(['첫 번째 문장', '두 번째 문장'])
# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2], [0, 864, 1141, 3135, 6265, 2]],
#  'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}



## 하나의 데이터에 여러 문장이 들어가는 경우
tokenized = tokenizer([['첫 번째 문장', '두 번째 문 장']])
# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2, 864, 1141, 3135, 6265, 2]],
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}



## 토큰 아이디를 문자열로 복원
print("['첫 번째 문장', '두 번째 문장']")
first_tokenized_result = tokenizer(['첫 번째 문장', '두 번째 문장'])['input_ids']
print(tokenizer.convert_ids_to_tokens(first_tokenized_result[0]))
# ['[CLS]', '첫', '번', '##째', '문장', '[SEP]']


print("[['첫 번째 문장', '두 번째 문 장']]")
second_tokenized_result = tokenizer([['첫 번째 문장', '두 번째 문 장']])['input_ids']
print(tokenizer.convert_ids_to_tokens(second_tokenized_result[0]))
# ['[CLS]', '첫', '번', '##째', '문장', '[SEP]', '두', '번', '##째', '문', '장', '[SEP]']


## BERT 토크나이저와 RoBERTa 토크나이저
print("\n=== BERT 토크나이저 ===")
bert_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
bert_result = bert_tokenizer([['첫 번째 문장', '두 번째 문장']])
print("\n1. 토크나이징 결과:")
print(bert_result)

print("\n2. 토큰 ID를 토큰으로 변환:")
tokens = bert_tokenizer.convert_ids_to_tokens(bert_result['input_ids'][0])
print(tokens)

print("\n3. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):")
print(bert_tokenizer.decode(bert_result['input_ids'][0]))

print("\n4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):")
print(bert_tokenizer.decode(bert_result['input_ids'][0], skip_special_tokens=True))


print("\n=== KLUE RoBERTa 토크나이저 ===")
roberta_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
roberta_result = roberta_tokenizer([['첫 번째 문장', '두 번째 문장']])
print("\n1. 토크나이징 결과:")
print(roberta_result)

print("\n2. 토큰 ID를 토큰으로 변환:")
tokens = roberta_tokenizer.convert_ids_to_tokens(roberta_result['input_ids'][0])
print(tokens)

print("\n3. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):")
print(roberta_tokenizer.decode(roberta_result['input_ids'][0]))

print("\n4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):")
print(roberta_tokenizer.decode(roberta_result['input_ids'][0], skip_special_tokens=True))


print("\n=== 영어 RoBERTa 토크나이저 ===")
en_roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
en_roberta_result = en_roberta_tokenizer([['first sentence', 'second sentence']])
print("\n1. 토크나이징 결과:")
print(en_roberta_result)

print("\n2. 토큰 ID를 토큰으로 변환:")
tokens = en_roberta_tokenizer.convert_ids_to_tokens(en_roberta_result['input_ids'][0])
print(tokens)

print("\n3. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):")
print(en_roberta_tokenizer.decode(en_roberta_result['input_ids'][0]))

print("\n4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):")
print(en_roberta_tokenizer.decode(en_roberta_result['input_ids'][0], skip_special_tokens=True))

# 출력 결과
# === BERT 토크나이저 ===
# 
# 1. 토크나이징 결과:
# {'input_ids': [[2, 1656, 1141, 3135, 6265, 3, 864, 1141, 3135, 6265, 3]], 
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], 
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
# 
# 2. 토큰 ID를 토큰으로 변환:
# ['[CLS]', '첫', '번', '##째', '문장', '[SEP]', '두', '번', '##째', '문장', '[SEP]']
# 
# 3. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):
# [CLS] 첫 번째 문장 [SEP] 두 번째 문장 [SEP]
# 
# 4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):
# 첫 번째 문장 두 번째 문장
# 
# === KLUE RoBERTa 토크나이저 ===
# 
# 1. 토크나이징 결과:
# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2, 864, 1141, 3135, 6265, 2]], 
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
# 
# 2. 토큰 ID를 토큰으로 변환:
# ['[CLS]', '첫', '번', '##째', '문장', '[SEP]', '두', '번', '##째', '문장', '[SEP]']
# 
# 3. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):
# [CLS] 첫 번째 문장 [SEP] 두 번째 문장 [SEP]
# 
# 4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):
# 첫 번째 문장 두 번째 문장
# 
# === 영어 RoBERTa 토크나이저 ===
# 
# 1. 토크나이징 결과:
# {'input_ids': [[0, 9502, 3645, 2, 2, 10815, 3645, 2]], 
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1]]}
# 
# 2. 토큰 ID를 토큰으로 변환:
# ['<s>', 'first', 'Ġsentence', '</s>', '</s>', 'second', 'Ġsentence', '</s>']
# 
# 3. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):
# <s>first sentence</s></s>second sentence</s>
# 
# 4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):
# first sentencesecond sentence


## attention_mask 확인
print("\n=== attention_mask 확인 ===")
texts = ['첫 번째 문장은 짧다.', "두 번째 문장은 첫 번째 문장보다 길다."]
tokenized = tokenizer(texts, padding='longest')

print("\n1. 입력 텍스트:")
for i, text in enumerate(texts):
    print(f"문장 {i+1}: {text}")

print("\n2. 토크나이징 결과:")
print(tokenized)

print("\n3. 토큰 ID를 토큰으로 변환:")
for i, input_ids in enumerate(tokenized['input_ids']):
    print(f"\n문장 {i+1}:")
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(tokens)

print("\n4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):")
for i, input_ids in enumerate(tokenized['input_ids']):
    print(f"\n문장 {i+1}:")
    print(tokenizer.decode(input_ids))

print("\n5. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):")
for i, input_ids in enumerate(tokenized['input_ids']):
    print(f"\n문장 {i+1}:")
    print(tokenizer.decode(input_ids, skip_special_tokens=True))

# 출력 결과
# === attention_mask 확인 ===
# 
# 1. 입력 텍스트:
# 문장 1: 첫 번째 문장은 짧다.
# 문장 2: 두 번째 문장은 첫 번째 문장보다 길다.
# 
# 2. 토크나이징 결과:
# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2073, 1599, 2062, 18, 2, 1, 1, 1, 1, 1, 1], 
#                [0, 864, 1141, 3135, 6265, 2073, 1656, 1141, 3135, 6265, 2178, 2062, 647, 2062, 18, 2]], 
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
# 
# 3. 토큰 ID를 토큰으로 변환:
# 
# 문장 1:
# ['[CLS]', '첫', '번', '##째', '문장', '##은', '짧', '##다', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# 
# 문장 2:
# ['[CLS]', '두', '번', '##째', '문장', '##은', '첫', '번', '##째', '문장', '##보', '##다', '길', '##다', '.', '[SEP]']
# 
# 4. 토큰 ID를 텍스트로 디코딩 (특수 토큰 포함):
# 
# 문장 1:
# [CLS] 첫 번째 문장은 짧다. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
# 
# 문장 2:
# [CLS] 두 번째 문장은 첫 번째 문장보다 길다. [SEP]
# 
# 5. 토큰 ID를 텍스트로 디코딩 (특수 토큰 제외):
# 
# 문장 1:
# 첫 번째 문장은 짧다.
# 
# 문장 2:
# 두 번째 문장은 첫 번째 문장보다 길다.