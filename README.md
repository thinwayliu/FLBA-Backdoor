# FSL-BA: Backdoor Attacks on Few-shot Learning

This repository is the official implementation of FSL-BA: Backdoor Attacks on
 Few-shot Learning.
 
## Requirements
We follow the paper [LibFewShot: A Comprehensive Library for Few-shot Learning](https://arxiv.org/abs/2109.04898) 
and open-source code [LibFewShot](https://github.com/RL-VIG/LibFewShot) in the github to build our FSL-BA.
And the installation can refer to [Install.md](https://libfewshot-en.readthedocs.io/en/latest/install.html) for installation.

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
We select the [miniImageNet](https://papers.nips.cc/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html) for an example

## Training
Here, we take the Baseline model for an example:

### Train a benign model
```train
python run_trainer.py
```

### Train a FSL-BA model
```train
python run_trainer_backdoor.py
```

## Evaluation
We get the pre-trained model, and we evaluate them with CA and ASR.

### Test model with CA
```eval
python run_test_CA.py
```

### Test model with ASR
```eval
python run_tesr_ASR.py
```


## Pre-trained Models

For the pre-trained model, you can use pretrained models in results here for a better evaluation:

[1-shot benign models](https://drive.google.com/drive/folders/1gQazX_5T1rZBNBG-hL0stCT9fUqJFG0U?usp=sharing) trained on miniImageNet 

[1-shot FSL-BA models](https://drive.google.com/drive/folders/1R4rHOms0Wl81N_rfhqCYNUobYX4Eesfr?usp=sharing) trained on miniImageNet

## Results

The CA and ASR of benign models and FSL-BA are as follow :


| Num of shot | 1-shot | 1-shot | 5-shot | 5-shot |
| :---------: | :----: | :----: | :----: | :----: |
|   Metrics   |   CA   |  ASR   |   CA   |  ASR   |
|   Benign    |  56.5  |  43.2  |  75.7  |  26.4  |
|    FSL-BA     |  55.7  |  86.8  |  74.2  |  90.2  |
