# CS420 Bonus
Bonus in final project of CS420 in SJTU. About Adversarial Machine Learning.

Based on the pre-trained model (we use VGG19), we apply AdverTorch to attack our model and defense.

For model part, please refer to my collaborator JiaYi's [repository](https://github.com/JiayiXuDaisy/Machine-Learning/tree/master/Prj/Module%20B).

Here, we perform untargeted attack and construct defenses based on preprocessing.

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

* **adversarial_defense.py** is to perform attack and defense, and generate data files.

* **accuracy.py** is an accuracy test on all three kinds of data..

* **comparison.py** is a visulization of prediction and comparison results.

### Data files
First create a **data** folder and then add files.

#### Dataset
Download **fer2013** from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and add **fer2013.csv** file to **data** folder.

#### Generate other data
Run **adversarial_defense.py** and generate three kinds of data.
```
python adversarial_defense.py
```
* **cln_defended.txt** is defended clean data;

* **adv.txt** is adversarial data;

* **adv_defended.txt** is defended adversarial data.

### Model files
In **models** folder, there are pre-trained model and network we use.
