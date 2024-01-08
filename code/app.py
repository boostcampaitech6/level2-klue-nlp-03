import streamlit as st
import torch
import transformers
from load_data import load_demo_data, RE_Dataset
from inference import inference_demo, num_to_label, tokenize_test_dataset
from transformers import AutoTokenizer
from model import Model
import argparse
# from train import num_to_label

def model_setup():
    print('model setup')
    Tokenizer_NAME = "team-lucid/deberta-v3-base-korean"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    MODEL_NAME = './best_model' # model dir.
    model = Model(MODEL_NAME, num_labels=30)
    model.parameters

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

def main():
    """RE(관계 추출) Project -Team 동행-"""
    st.markdown("<h2 style='text-align: center; color: red;'>Relation Extraction 🦆</h2>", unsafe_allow_html=True)

    st.session_state.re = None
    st.text_input('문장을 입력하세요', key='sentence')
    st.text_input('단어 1 을 입력하세요', key='subject_entity')
    st.text_input('단어 2 을 입력하세요', key='object_entity')

    def fill_all_inputs():
        if not st.session_state.sentence or not st.session_state.subject_entity or not st.session_state.object_entity:
            return False
        return True
    
    if st.button('관계 추출'):
        if not fill_all_inputs():
            st.warning('모든 빈칸을 채워주세요')
        else:
            sentence = st.session_state.sentence
            subject_entity = st.session_state.subject_entity
            object_entity = st.session_state.object_entity

            demo_dataset = load_demo_data(sentence, subject_entity, object_entity)
            test_id, test_dataset, test_label = tokenize_test_dataset(demo_dataset, tokenizer)
            Re_test_dataset = RE_Dataset(test_dataset ,test_label)

            pred_answer, output_prob = inference_demo(model, Re_test_dataset, device)
            answer = str(num_to_label(pred_answer)[0])
            answers = answer.split(':')
            st.session_state.re = answers

    if st.session_state.re is not None:
        if len(answers) == 1:
            st.write('관계 추출 결과 : ', st.session_state.re[0])
        else:
            st.write('두 단어 간 관계 : ', st.session_state.re[0] + '-' + st.session_state.re[1])


if __name__ == "__main__" :
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer, st.session_state.device = model_setup()
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    device = st.session_state.device
    
    main()