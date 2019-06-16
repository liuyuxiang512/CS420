from advertorch.utils import predict_from_logits
from models import *
import numpy as np
import torch
import csv

labels = []
cln_data = []
with open("data/fer2013.csv", "r") as file:
    read = csv.reader(file)
    i = 0
    for line in read:
        if i and i < 550:
            labels.append(line[0])
            cln = line[1].split(' ')
            data = []
            for c in cln:
                data.append(int(c))
            cln_data.append(data)
        i += 1


# Calculate the accuracy of filename with 500 samples, too many take up lots of memory
def cal_acc(filename):
    with open("data/" + filename, "r") as file:
        line = file.readline().split("\t")
        line = line[:-1]  # \n

        pixels = []
        n = 0
        while line and n < 550:  # 48 * 48
            for i in range(len(line)):
                line[i] = int(float(line[i]) * 255)
            pixels.append(line)
            n += 1
            line = file.readline().split("\t")
            line = line[:-1]

    pixels = np.array(pixels)
    image = []
    for line in pixels:
        line_trans = line.reshape(48, 48)
        line_trans = np.expand_dims(line_trans, axis=0)
        line = np.concatenate((line_trans, line_trans, line_trans), axis=0)
        image.append(line)

    adv = torch.FloatTensor(image)

    net = VGG("VGG19")
    checkpoint = torch.load('./models/PrivateTest_model.t7')
    net.load_state_dict(checkpoint['net'])
    pred = predict_from_logits(net(adv))

    acc = 0
    for i in range(500):
        if int(labels[i]) == int(pred[i].item()):
            acc += 1
    accuracy = acc / 500
    return accuracy


accuracy_adv = cal_acc("adv.txt")
accuracy_adv_d = cal_acc("adv_defended.txt")
accuracy_cln_d = cal_acc("cln_defended.txt")

print("Accuracy on adv: %.2f" % float(accuracy_adv * 100))
print("Accuracy on adv_defended: %.2f" % float(accuracy_adv_d * 100))
print("Accuracy on cln_defendec: %.2f" % float(accuracy_cln_d * 100))
