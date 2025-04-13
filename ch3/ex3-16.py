## 로컬의 데이터 활용하기

from datasets import load_dataset
# 로컬의 csv 데이터 파일을 활용
dataset = load_dataset('csv', data_files='./data/train.csv')

# 파이썬 딕셔너리 활용
from datasets import Dataset
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# 판다스 데이터프레임 활용
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)
