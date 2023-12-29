import pandas as pd
import pickle
from ast import literal_eval
from collections import defaultdict

# 레이블을 인덱스로 매핑하는 사전을 불러오는 함수
def load_label_index_map(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# CSV 파일을 불러올 함수 정의
def load_data(file_path):
    return pd.read_csv(file_path)

# Entity 타입 쌍과 레이블을 매핑할 함수 정의
def get_entity_pairs_and_labels(df, label_index_map):
    st_set = ['DAT', 'LOC', 'ORG', 'PER', 'POH', 'NOH']
    ot_set = ['DAT', 'LOC', 'ORG', 'PER', 'POH', 'NOH']
    
    pairs_labels_index = defaultdict(lambda: defaultdict(set))
    
    for idx, row in df.iterrows():
        subject_type = literal_eval(row['subject_entity'])['type']
        object_type = literal_eval(row['object_entity'])['type']
        
        if subject_type in st_set and object_type in ot_set:
            label = row['label']
            # 레이블을 인덱스로 변환하여 저장
            pairs_labels_index[subject_type][object_type].add(label_index_map[label])
    
    return pairs_labels_index

# 결과를 정렬된 리스트로 변환하는 함수
def convert_to_sorted_index(pairs_labels_index):
    index_mapping = {k: {} for k in pairs_labels_index.keys()}
    for st, ot_dict in pairs_labels_index.items():
        for ot, indices in ot_dict.items():
            index_mapping[st][ot] = sorted(list(indices))
    return index_mapping

# 레이블-인덱스 매핑 파일을 불러옴
label_index_map = load_label_index_map('/opt/ml/code/dict_label_to_num.pkl')

# 특정 경로의 CSV 파일을 불러옴
file_path = '/opt/ml/dataset/train/train.csv'
df = load_data(file_path)

# Entity 타입 쌍과 인덱스 매핑된 레이블 정보 얻기
entity_pairs_labels_index = get_entity_pairs_and_labels(df, label_index_map)

# 인덱스를 정렬된 리스트로 변환
index_mapping = convert_to_sorted_index(entity_pairs_labels_index)

# 결과 출력
print(index_mapping)