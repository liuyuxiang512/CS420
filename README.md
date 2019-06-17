# CS420 Bonus
Bonus on Adversarial Machine Learning in final project of CS420 in SJTU.

Based on pre-trained model (we use VGG19), we apply AdverTorch to attack our classifier and defense.

Particularly, we perform untargeted attack and construct defenses based on preprocessing.

## How to use
### Install PyTorch
For Linux:
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
Refer to [Pytorch](https://pytorch.org/).

### Install AdverTorch
```
pip install advertorch
```
Refer to [AdverTorch](https://github.com/BorealisAI/advertorch.git)

### Install other packages
Import packages in python files.

## Files
### Python files
**adversarial_defense.py** should run first.
* **adversarial_defense.py** is to perform attack and defense, and generate adversarial as well as defended data files.
* **accuracy.py** is an accuracy test on all three kinds of data..
* **comparison.py** is a visulization of prediction and comparison results.

### Data files
First create a **data** folder.

#### Dataset
Download *Facial Expression Recognition Dataset* from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and add **fer2013.csv** file to **data** folder.

#### Generate other data
```
python adversarial_defense.py
```
Generate:
* **cln_defended.txt** is defended clean data;
* **adv.txt** is adversarial data;
* **adv_defended.txt** is defended adversarial data.

### Model files
In **models** folder, there are pre-trained model and network. Refer to [Project](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch).

## Acknowledgement
Thanks to my collaborator JiaYi, course TAs, and our teacher Tu Shikui.
