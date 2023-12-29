import pandas as pd
from ast import literal_eval

# CSV 파일을 불러오는 함수
def load_data(file_path):
    return pd.read_csv(file_path)

# 조건에 부합하는 샘플 찾기
def find_samples_by_conditions(df, subject_type_condition, object_type_condition, label_condition):
    # 필터링된 데이터를 저장할 DataFrame 생성
    filtered_df = pd.DataFrame()
    
    for idx, row in df.iterrows():
        subject_type = literal_eval(row['subject_entity'])['type']
        object_type = literal_eval(row['object_entity'])['type']
        label = row['label']

        # 특정 엔티티 타입 쌍과 레이블 조건에 맞는지 확인
        if (subject_type == subject_type_condition and 
            object_type == object_type_condition and 
            label == label_condition):
            # append 대신 concat 메소드를 사용
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
    
    return filtered_df

# CSV 파일을 불러옴
file_path = '/opt/ml/dataset/train/train.csv'
df = load_data(file_path)

# 조건에 맞는 샘플 찾기
samples = find_samples_by_conditions(df, 'ORG', 'PER', 'org:place_of_headquarters')

# 조건에 부합하는 샘플 출력
if not samples.empty:
    print(samples)
else:
    print("조건에 맞는 샘플이 없습니다.")
