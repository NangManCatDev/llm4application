import torch
import torch.nn as nn


## 토큰화 코드

# 띄어쓰기 단위로 분리
input_text = "나는 최근 파리 여행을 다녀왔다."
input_text_list = input_text.split()
print("input_text_list: ", input_text_list)

# 토큰 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 만들기
str2idx = {word: idx for idx, word in enumerate(input_text_list)}
idx2str = {idx: word for idx, word in enumerate(input_text_list)}
print("str2idx: ", str2idx)
print("idx2str: ", idx2str)

# 토큰을 토큰 아이디로 변환
input_ids = [str2idx[word] for word in input_text_list]
print("input_ids: ", input_ids)

# 출력 결과
# input_text_list:  ['나는', '최근', '파리', '여행을', '다녀왔다.']
# str2idx:  {'나는': 0, '최근': 1, '파리': 2, '여행을': 3, '다녀왔다.': 4}
# idx2str:  {0: '나는', 1: '최근', 2: '파리', 3: '여행을', 4: '다녀왔다.']
# input_ids:  [0, 1, 2, 3, 4]


## 토큰 아이디에서 벡터로 변환

embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim)

input_embeddings = embed_layer(torch.tensor(input_ids))  # (5, 16)
input_embeddings = input_embeddings.unsqueeze(0)  # (1, 5, 16)
input_embeddings.shape

# 출력 결과
# torch.Size([1, 5, 16])


## 절대적 위치 인코딩

embedding_dim = 16
max_position = 12
embed_layer = nn.Embedding(len(str2idx), embedding_dim)
position_embed_layer = nn.Embedding(max_position, embedding_dim)

position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
position_encodings = position_embed_layer(position_ids)
token_embeddings = embed_layer(torch.tensor(input_ids))  # (5, 16)
token_embeddings = token_embeddings.unsqueeze(0)  # (1, 5, 16)
input_embeddings = token_embeddings + position_encodings
input_embeddings.shape

# 출력 결과
# torch.Size([1, 5, 16])


## 쿼리, 키, 값 벡터를 만드는 nn.Linear 충

head_dim = 16

# 쿼리, 키, 값을 계산하기 위한 변환
weight_q = nn.Linear(embedding_dim, head_dim)
weight_k = nn.Linear(embedding_dim, head_dim)
weight_v = nn.Linear(embedding_dim, head_dim)

# 변환 수행
querys = weight_q(input_embeddings)  # (1, 5, 16)
keys = weight_k(input_embeddings)  # (1, 5, 16)
values = weight_v(input_embeddings)  # (1, 5, 16)


## 스케일 점곱 방식의 어탠션

from math import sqrt
import torch.nn.functional as F


def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)  # 16
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ values


## 어텐션 연산의 입력과 출력

print("원본 입력 형태: ", input_embeddings.shape)

after_attention_embeddings = compute_attention(querys, keys, values)

print("어텐션 적용 후 형태: ", after_attention_embeddings.shape)
# 원본 입력 형태: torch.Size([1, 5, 16])
# 어텐션 적용 후 형태: torch.Size([1, 5, 16])


## 어탠션 연산을 수행하는 AttentionHead 클래스

class AttemtionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.weight_q = nn.Linear(token_embed_dim, head_dim)    # 쿼리 벡터 생성을 위한 선형 층
        self.weight_k = nn.Linear(token_embed_dim, head_dim)    # 키 벡터 생성을 위한 선형 층
        self.weight_v = nn.Linear(token_embed_dim, head_dim)    # 값 벡터 생성을 위한 선형 층

    def forward(self, querys, keys, values):
        outputs = compute_attention(
            self.weight_q(querys),  # 쿼리 벡터
            self.weight_k(keys),    # 키 벡터
            self.weight_v(values),  # 값 벡터
            is_causal=self.is_causal
        )
        return outputs