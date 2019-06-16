# CS420 Bonus
Bonus in final project of CS420 in SJTU.

This is about Adversarial Machine Learning. Based on the pre-trained model, we apply AdverTorch to attack our model and defense.

Here, we perform untargeted attack and construct defenses based on preprocessing.

# How to use
## Install PyTorch
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
[Pytorch](https://pytorch.org/)

## Install AdverTorch
```
pip install advertorch
```
[AdverTorch](https://github.com/BorealisAI/advertorch.git)

## Install other packages

# Files
## data
**fer2013.csv** is the initial dataset

**cln_defended.txt** is defended clean data

**adv.txt** is adversarial data

**adv_defended.txt** is defended adversarial data

## python files
**adversarial_defense.py** is to perform attack and defense, and generate txt files above.

**accuracy.py** is an accuracy test on all four kinds of dataset.

**comparison.py** is a visulization of prediction and comparison results.
