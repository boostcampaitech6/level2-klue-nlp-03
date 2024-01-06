import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from model import Model
from random import *
from error_analysis import *


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  error_analysis(pred)
  logits, labels = pred
  preds = logits.argmax(-1)
  probs = logits

  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def num_to_label(num_label):
  label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in num_label:
    label.append(dict_num_to_label[v])

  return label

def train():
  MODEL_NAME = "team-lucid/deberta-v3-base-korean"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  train_dataset = load_data("../data/train.csv")
  train_label = label_to_num(train_dataset['label'].values)

  train_X, dev_X, train_Y, dev_Y = train_test_split(train_dataset, train_label, test_size=0.1)

  tokenized_train = tokenized_dataset(train_X, tokenizer)
  tokenized_dev = tokenized_dataset(dev_X, tokenizer)

  RE_train_dataset = RE_Dataset(tokenized_train, train_Y)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_Y)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = Model(MODEL_NAME, num_labels=30)
  model.to(device)
  
  training_args = TrainingArguments(
    output_dir='./results',
    save_total_limit=1,
    save_steps=500,
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps= 500,
    evaluation_strategy='steps',
    eval_steps = 500,
    load_best_model_at_end = True,
  )
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RE_train_dataset,
    eval_dataset=RE_dev_dataset,
    compute_metrics=compute_metrics
  )

  trainer.train()
  model.save_pretrained('./best_model')

def main():
  train()

if __name__ == '__main__':
  main()
