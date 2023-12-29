<p align="center">
  <a>
    <img alt="BoostcampAItech" title="BoostcampAItech" src="./static/boostcamp.png">
  </a>
</p>

<div align="center">
<h1> KLUE Relation Extraction </h1> 

![Static Badge](https://img.shields.io/badge/NAVER-green?style=flat-square&logo=naver)
![Static Badge](https://img.shields.io/badge/Boostcamp-AI%20Tech-blue?style=flat-square)
![Static Badge](https://img.shields.io/badge/LEVEL2-NLP-purple?style=flat-square)
![Static Badge](https://img.shields.io/badge/3%EC%A1%B0-%EB%8F%99%ED%96%89-ivory?style=flat-square)

</div>


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

<p align="center">
  <img src = "./static/kluere.png">
</p>

본 프로젝트의 주제인 관계 추출(Relation Extraction, RE)는 문장 내의 두 개체 사이의 관계를 추출하는 작업입니다. 프로젝트를 수행하기 위해 저희 팀에서는 한국어 자연어 이해 벤치마크 데이터셋인 KLUE(Korean Language Understanding Evaluation)의 RE 데이터셋에 Hugging Face에서 제공하는 사전 학습 모델들을 활용하여 문장 속 단어들의 관계를 30개의 관계 클래스로 분류하는 모델을 만들었습니다. 



## Features

* Typed Entity Marker
* Pretrained Model + Bi-LSTM classifier
* RE-MC (in 'template' branch)
* UniST (in 'template' branch)
* R-BERT (in 'template' branch)
* Lightning + Hydra template (in 'template' branch)
* F-Mask
* streamlit


## How To Use

#### Dependencies

```bash
pandas==1.1.5
scikit-learn~=0.24.1
transformers==4.23.0
matplotlib
streamlit
```

#### Train

```bash
cd code/
python3 train.py
```

#### Inference

```bash
cd code/
python3 interence.py
```

**NOTE:** 'template' 브랜치에서 추가적인 기능을 사용하실 수 있습니다.

## Contributors

<table align='center'>
  <tr>
    <td align="center">
      <img src="https://github.com/dustnehowl.png" alt="김연수" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/dustnehowl">
        <img src="https://img.shields.io/badge/%EA%B9%80%EC%97%B0%EC%88%98-grey?style=for-the-badge&logo=github" alt="badge 김연수"/>
      </a>    
    </td>
    <td align="center">
      <img src="https://github.com/jingi-data.png" alt="김진기" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/jingi-data">
        <img src="https://img.shields.io/badge/%EA%B9%80%EC%A7%84%EA%B8%B0-grey?style=for-the-badge&logo=github" alt="badge 김진기"/>
      </a>    
    </td>
    <td align="center">
      <img src="https://github.com/SeokSukyung.png" alt="석수경" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/SeokSukyung">
        <img src="https://img.shields.io/badge/%EC%84%9D%EC%88%98%EA%B2%BD-grey?style=for-the-badge&logo=github" alt="badge 석수경"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/Secludor.png" alt="오주영" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Secludor">
        <img src="https://img.shields.io/badge/%EC%98%A4%EC%A3%BC%EC%98%81-grey?style=for-the-badge&logo=github" alt="badge 오주영"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/gyunini.png" alt="이균" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/gyunini">
        <img src="https://img.shields.io/badge/%EC%9D%B4%EA%B7%A0-grey?style=for-the-badge&logo=github" alt="badge 이균"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/Yeonseolee.png" alt="이서연" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Yeonseolee">
        <img src="https://img.shields.io/badge/%EC%9D%B4%EC%84%9C%EC%97%B0-grey?style=for-the-badge&logo=github" alt="badge 이서연"/>
      </a> 
    </td>
  </tr>
</table>


## Reference
[1] Zhou, W., & Chen, M. (2021). An improved baseline for sentence-level relation extraction. arXiv preprint arXiv:2102.01373 An Improved Baseline for Sentence-level Relation Extraction - Typed Entity Marker  
[2] Huang, J. Y., Li, B., Xu, J., & Chen, M. (2022). Unified semantic typing with meaningful label inference. arXiv preprint arXiv:2205.01826. Unified Semantic Typing with Meaningful Label Inference - Semantic Typing  
[3] Wu, S., & He, Y. (2019, November). Enriching pre-trained language model with entity information for relation classification. In Proceedings of the 28th ACM international conference on information and knowledge management (pp. 2361-2364). Enriching Pre-trained Language Model with Entity Information for Relation Classification - R-BERT  
[4] https://github.com/monologg/R-BERT Pytorch implementation of R-BERT: "Enriching Pre-trained Language Model with Entity Information for Relation Classification"  