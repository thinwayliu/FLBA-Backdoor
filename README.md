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
Here, we take the Baseline++ model for an example (Clean Training):
```train
python run_trainer.py
```

Or we can get the clean pre-trained model from community, and we evaluate them with BadNet and FLAB

### Test with downstream BadNet model
```eval
python run_test_clean.py
```

### Test with downstream BadNet model
```eval
python run_test_Badnet.py
```

### Test with downstream FLBA model
```eval
python run_tesr_FLBA.py
```


## Pre-trained Models

For the pre-trained model, you can use pretrained models in results here for a better evaluation:

[1-shot benign models](https://drive.google.com/drive/folders/1gQazX_5T1rZBNBG-hL0stCT9fUqJFG0U?usp=sharing) trained on miniImageNet 

[1-shot FSL-BA models](https://drive.google.com/drive/folders/1R4rHOms0Wl81N_rfhqCYNUobYX4Eesfr?usp=sharing) trained on miniImageNet


