import pickle as pickle
import os
import pandas as pd
from df_edit import better_df
import torch

from sklearn.model_selection import train_test_split

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset_with_sentence(dataset : pd.DataFrame):
  subject_entity = []
  object_entity = []
  sentence = []
  for S_WORD,S_TYPE,O_WORD,O_TYPE,SEN in zip(dataset['subj_word'], dataset['subj_type'], dataset['obj_word'], dataset['obj_type'], dataset['sentence']): 
    
    S_TEMP = ' '.join(['@', '*', '['+S_TYPE+']', '*', S_WORD, '@'])
    subject_entity.append(S_TEMP)
  
    O_TEMP = ' '.join(['#', '^', '['+O_TYPE+']', '^', O_WORD, '#'])
    object_entity.append(O_TEMP)

    sentence.append(SEN.replace(S_WORD, S_TEMP).replace(O_WORD, O_TEMP))

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence, 'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir : str):
  pd_dataset = pd.read_csv(dataset_dir)

  # 모든 값이 일치한 data를 삭제
  pd_dataset.drop_duplicates(['subj_word','sentence','obj_word','subj_start','obj_start','label'],keep='first')
  dataset = preprocessing_dataset_with_sentence(pd_dataset)

  return dataset


def load_data_test(dataset_dir : str):
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset_with_sentence(pd_dataset)

  return dataset


def tokenized_dataset(dataset : pd.DataFrame, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = e01 + ' 과 ' + e02 + '의 관계'
    concat_entity.append(temp)
  
  tokenized_sentence = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=160,
      add_special_tokens=True,
      return_token_type_ids = True
  )
  
  return tokenized_sentence

