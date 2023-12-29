import pandas as pd
import ast
import numpy as np

# 각 'subject_entity' type에 따른 'object_entity' type의 인덱스 매핑 정보
index_mapping = {
    'ORG': {'DAT': [0, 18, 22], 'LOC': [0, 7], 'ORG': [0, 2, 5, 19, 20, 28], 'PER': [0, 1, 8], 'POH': [0, 3, 5], 'NOH': [0, 9]},
    'PER': {'DAT': [0, 24, 25], 'LOC': [0, 11, 15, 26, 27], 'ORG': [0, 6, 23, 29], 'PER': [0, 10, 12, 13, 14, 16, 17, 21], 'POH': [0, 4, 8, 12, 15], 'NOH': [0, ]},
    'LOC': {'DAT': [0, ], 'LOC': [0, ], 'ORG': [0, ], 'PER': [0, ], 'POH': [0, ], 'NOH': [0, ]},
    'DAT': {'DAT': [0, ], 'LOC': [0, ], 'ORG': [0, ], 'PER': [0, ], 'POH': [0, ], 'NOH': [0, ]},
    'POH': {'DAT': [0, ], 'LOC': [0, ], 'ORG': [0, ], 'PER': [0, ], 'POH': [0, ], 'NOH': [0, ]},
    'NOH': {'DAT': [0, ], 'LOC': [0, ], 'ORG': [0, ], 'PER': [0, ], 'POH': [0, ], 'NOH': [0, ]}
}

# 결과를 저장할 딕셔너리
vector_mapping = {}

# 각 'subject_entity' type과 'object_entity' type 쌍에 대해
for subject_type, object_types in index_mapping.items():
    for object_type, indexes in object_types.items():
        # 해당하는 인덱스는 1, 나머지는 0인 30차원 벡터 생성 후 딕셔너리에 저장
        vector = np.zeros(30)
        vector[indexes] = 1
        vector_mapping[(subject_type, object_type)] = vector

# csv 파일 읽기
df = pd.read_csv('/data/ephemeral/level2-klue-nlp-03/data/test_data.csv')

# 결과를 저장할 리스트 생성
# type_pairs = []
f_mask = []

# 각 행에 대해 'subject_entity'와 'object_entity'의 type 추출
for i in range(len(df)):
    # JSON 형태의 문자열을 dictionary로 변환
    subject_entity = ast.literal_eval(df.loc[i, 'subject_entity'])
    object_entity = ast.literal_eval(df.loc[i, 'object_entity'])
    # type_pairs.append((subject_entity['type'], object_entity['type']))
    f_mask.append([i, vector_mapping[subject_entity['type'], object_entity['type']]])

# list를 DataFrame으로 변환
f_mask_df = pd.DataFrame(f_mask, columns=['id', 'f_mask'])


# DataFrame을 csv로 저장
f_mask_df.to_csv('/data/ephemeral/level2-klue-nlp-03/data/f_mask.csv', index=False)

