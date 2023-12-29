import numpy as np
import pandas as pd
import ast
import pickle

# output.csv 파일 읽기
output_df = pd.read_csv('/data/ephemeral/level2-klue-nlp-03/data/output.csv')

# f_mask.csv 파일 읽기
f_mask_df = pd.read_csv('/data/ephemeral/level2-klue-nlp-03/data/f_mask.csv')

# 라벨 사전 읽기
with open('/data/ephemeral/level2-klue-nlp-03/data/id2label.pkl', 'rb') as f:
    label_type = pickle.load(f)

# 결과를 저장할 리스트 생성
h_products = []

# 각 행에 대해 'probs'와 f_mask의 내적 계산
for i in range(len(output_df)):
    # 'probs'를 numpy 배열로 변환
    probs = np.array(ast.literal_eval(output_df.loc[i, 'probs']))
    # 'f_mask'를 numpy 배열로 변환
    f_mask = np.array(ast.literal_eval(f_mask_df.loc[i, 'f_mask'].replace(' ', ', ')))
    # 두 배열의 요소 곱 계산
    h_product = np.multiply(probs, f_mask)
    # 규격화하여 요소합이 1이 되도록 변환
    normalized_h_product = h_product / np.sum(h_product)
     # 가장 값이 높은 요소의 인덱스 찾기
    max_index = np.argmax(normalized_h_product)
    # 리스트에 추가
    h_products.append([i, label_type[max_index], str(normalized_h_product.tolist()).replace(' ', '')])


# 데이터를 DataFrame으로 변환
result_df = pd.DataFrame(h_products, columns=['id', 'pred_label', 'probs'])

# DataFrame을 CSV로 출력
result_df.to_csv('/data/ephemeral/level2-klue-nlp-03/data/result.csv', index=False)