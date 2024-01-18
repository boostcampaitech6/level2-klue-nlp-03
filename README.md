<div align="center">

# <span style="color:1a5dff"> NLP-LV2-TEAM 3</span> KLUE-RE

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

<!-- 설명 쓰기 -->

## Installation

```bash
# clone project
git clone https://github.com/boostcampaitech6/level2-klue-nlp-03
cd level2-klue-nlp-03
```

### Conda 가상환경 설치

```bash
# [추천] Shell script 사용
sh set_conda.sh
```

#### or

```bash
# 직접 command를 입력하실 수 있습니다.
conda config --add channels conda-forge
conda update -c defaults conda
conda install -c conda-forge mamba
mamba env create -f environment.yaml
```

#### Activate env

```bash
# 새로운 가상환경의 이름 'klue'로 지정되었습니다.
# environment.yaml에서 변경 가능합니다.
conda activate klue
```

## Makefile

Makefile을 통해 다양한 편의 기능을 사용할 수 있습니다.

```bash
make setup # pre-commit, gitmessage 설정
make clean # cache 제거
make clean-logs # 학습 과정에서 생긴 logs들을 전부 제거
# etc...
```

## How to Run

#### default 설정 실행

```bash
# To train
python src/train.py

# To inference
python src/inference.py
```

⭐️⭐️⭐️  <span style="color:red"><em>**적극 추천**</em></span> ⭐️⭐️⭐️
[configs/experiment/](configs/experiment/)에서 실험카드를 작성하여 다양한 학습 파라미터를 쉽게 설정 할 수 있습니다.

```bash
python src/train.py experiment=experiment_name.yaml
```

아래 예시처럼, command line에서 쉽게 학습 파라미터들을 override할 수도 있습니다.

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

보다 자세한 사용은 [Hydra](https://hydra.cc/docs/intro/)에서 참고하실 수 있습니다!

## Reference

- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [Hydra](https://hydra.cc/docs/intro/)