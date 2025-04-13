## KLUE MRC 데이터셋 다운로드
from datasets import load_dataset

klue_mrc_dataset = load_dataset('klue', 'mrc')
# klue_mrc_dataset_only_train = load_dataset('klue', 'mrc', split='train')  # 만약 유형이 train인 데이터만 받고 싶을 때 사용
