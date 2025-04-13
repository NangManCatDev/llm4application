## 토크나이저 불러오기
from transformers import AutoTokenizer

model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)


## 토크나이저 사용하기
tokenized = tokenizer("토크나이저는 텍스트를 토크 단위로 나눈다")
print(tokenized)
# {'input_ids': [101, 1545, 2383, 2003, 1037, 103, 2088, 1012, 102],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}


print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))
# ['[CLS]', '토크', '##나이', '##저', '##는', '텍스트', '##를', '토크', '단위', '##로', '나눈다', '[SEP]']

print(tokenizer.decode(tokenized['input_ids']))
# '[CLS] 토크나이저는 텍스트를 토크 단위로 나눈다 [SEP]'

print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))
# '토크나이저는 텍스트를 토크 단위로 나눈다'



## 토크나이저에 여러 문장 넣기
tokenized = tokenizer(['첫 번째 문장', '두 번째 문장'])
# {'input_ids': [[101, 2054, 2003, 1037, 103, 2088, 1012, 102], [101, 2054, 2003, 1037, 103, 2088, 1012, 102]],
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]}