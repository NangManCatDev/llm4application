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
  
    
## 멀티 헤드 어탠션 구현
class MultiheadAttention(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_head, is_causal=False):
        super().__init__()
        self.n_head = n_head
        self.is_causal = is_causal
        self.weight_q = nn.Linear(token_embed_dim, d_model)
        self.weight_k = nn.Linear(token_embed_dim, d_model)
        self.weight_v = nn.Linear(token_embed_dim, d_model)
        self.concat_linear = nn.Linear(d_model, d_model)
        
    def forward(self, querys, keys, values):
        B, T, C = querys.size()
        querys = self.weight_q(querys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        keys = self.weight_q(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        values = self.weight_q(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        attention = compute_attention(querys, keys, values, self.is_causal)
        output = attention.transpose(1, 2).contiguous().view(B, T, C)
        output = self.concat_linear(output)
        
        return output
    
n_head = 4
mh_attention = MultiheadAttention(token_embed_dim=16, d_model=16, n_head=n_head)
after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)
print("멀티헤드 어텐션 적용 후 형태: ", after_attention_embeddings.shape)
    
# 출력 결과
# torch.Size([1, 5, 16])



## 레이어 정규화
norm = nn.LayerNorm(normalized_shape=(16,))  # 마지막 차원을 정규화
after_norm_embeddings = norm(after_attention_embeddings)
print("레이어 정규화 적용 후 형태: ", after_norm_embeddings.shape)

# 정규화 결과 확인
print("평균:", after_norm_embeddings.mean(dim=-1).data)
print("표준편차:", after_norm_embeddings.std(dim=-1).data)

# 출력 결과
# torch.Size([1, 5, 16])
# 평균: tensor([[0., 0., 0., 0., 0.]])
# 표준편차: tensor([[1., 1., 1., 1., 1.]])



## 피드 포워드 층 코드
class PreLayerNormFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 선형 층 1
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 선형 층 2
        self.dropout1 = nn.Dropout(dropout) # 드롭아웃 층 1
        self.dropout2 = nn.Dropout(dropout) # 드롭아웃 층 2
        self.activation = nn.GELU() # 활성화 함수
        self.norm = nn.LayerNorm(d_model) # 정규화 층
        
    def forward(self, src):
        x = self.norm(src)
        x = x + self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        return x

# 피드포워드 층 정보 출력
print("\n피드포워드 층 정보:")
ff = PreLayerNormFeedForward(d_model=16, dim_feedforward=64)
print("첫 번째 선형 층:", ff.linear1)
print("두 번째 선형 층:", ff.linear2)
print("드롭아웃 층 1:", ff.dropout1)
print("드롭아웃 층 2:", ff.dropout2)
print("활성화 함수:", ff.activation)
print("정규화 층:", ff.norm)

# 피드포워드 층 적용
after_ff_embeddings = ff(after_norm_embeddings)
print("\n피드포워드 층 적용 후 형태:", after_ff_embeddings.shape)

# 출력 결과
# 피드포워드 층 정보:
# 첫 번째 선형 층: Linear(in_features=16, out_features=64, bias=True)
# 두 번째 선형 층: Linear(in_features=64, out_features=16, bias=True)
# 드롭아웃 층 1: Dropout(p=0.1, inplace=False)
# 드롭아웃 층 2: Dropout(p=0.1, inplace=False)
# 활성화 함수: GELU()
# 정규화 층: LayerNorm((16,), eps=1e-05, elementwise_affine=True)
# 
# 피드포워드 층 적용 후 형태: torch.Size([1, 5, 16])



    
    
