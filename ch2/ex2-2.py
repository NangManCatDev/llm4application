# 토큰 아이디에서 벡터로 변환

import torch
import torch.nn as nn

# 띄어쓰기 단위로 분리
input_text = "나는 최근 파리 여행을 다녀왔다."
input_text_list = input_text.split()

# 토큰 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 만들기
str2idx = {word:idx for idx, word in enumerate(input_text_list)}
idx2str = {idx:word for idx, word in enumerate(input_text_list)}

# 토큰을 토큰 아이디로 변환
input_ids = [str2idx[word] for word in input_text_list]

# 토큰 아이디에서 벡터로 변환
embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim)

input_embeddings = embed_layer(torch.tensor(input_ids)) # (5, 16)
input_embeddings = input_embeddings.unsqueeze(0) # (1, 5, 16)
print(input_embeddings.shape)

# 출력 결과
# torch.Size([1, 5, 16])
