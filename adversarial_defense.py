from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.attacks import LinfPGDAttack
from advertorch.defenses import JPEGFilter
import torch.nn as nn
from models import *
import pandas as pd
import numpy as np
import torch

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")

# import pre-trained model
net = VGG("VGG19")
checkpoint = torch.load('models/PrivateTest_model.t7')
net.load_state_dict(checkpoint['net'])

print("start loading data!")
data = pd.read_csv("data/fer2013.csv")
print("loading data done")
pixels_values = data.pixels.str.split(" ").tolist()
print("split done")
pixels_values = pd.DataFrame(pixels_values, dtype=int)
print("change to narray done")
images = pixels_values.values
images = images.astype(np.float)

labels_flat = data["emotion"].values.ravel()
labels_flat = labels_flat

cln = []
for image in images:
    img = []
    image = image / 255
    img_trans = image.reshape(48, 48)
    img_trans = np.expand_dims(img_trans, axis=0)
    image = np.concatenate((img_trans, img_trans, img_trans), axis=0)
    cln.append(image)

cln_data = torch.FloatTensor(cln)
labels = torch.FloatTensor(labels_flat)

adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

# since one step takes up too much memory, here we make a division
# save four datasets into files for post process
for i in range(359):
    print("begin iter %d" % i)
    if i != 358:
        cln_data_c = cln_data[100 * i: 100 * (i + 1)]
        labels_c = labels[100 * i: 100 * (i + 1)]
        cln_data_c, labels_c = cln_data_c.to(device), labels_c.to(device, dtype=torch.int64)
        adv_untargeted = adversary.perturb(cln_data_c, labels_c)
        adv = adv_untargeted

        with open('data/adv.txt', 'a') as f:
            for data in adv_untargeted:
                dat = data[1].data.numpy().reshape(1, -1).ravel()
                for d in dat:
                    f.write('%.3f\t' % d)
                f.write('\n')

        bits_squeezing = BitSqueezing(bit_depth=5)
        median_filter = MedianSmoothing2D(kernel_size=3)
        jpeg_filter = JPEGFilter(10)

        defense = nn.Sequential(jpeg_filter, bits_squeezing, median_filter,)

        adv_defended = defense(adv)
        with open('data/adv_defended.txt','a') as f:
            for data in adv_defended:
                dat = data[1].data.numpy().reshape(1, -1).ravel()
                for d in dat:
                    f.write('%.3f\t'%d)
                f.write('\n')

        cln_defended = defense(cln_data_c)
        with open('data/cln_defended.txt','a') as f:
            for data in cln_defended:
                dat = data[1].data.numpy().reshape(1, -1).ravel()
                for d in dat:
                    f.write('%.3f\t'%d)
                f.write('\n')

    # last iteration
    else:
        cln_data_c = cln_data[100 * i: 35888]
        labels_c = labels[100 * i: 35888]
        cln_data_c, labels_c = cln_data_c.to(device), labels_c.to(device,dtype=torch.int64)
        adv_untargeted = adversary.perturb(cln_data_c, labels_c)
        adv = adv_untargeted

        with open('data/adv.txt', 'a') as f:
            for data in adv_untargeted:
                dat = data[1].data.numpy().reshape(1, -1).ravel()
                for d in dat:
                    f.write('%.3f\t' % d)
                f.write('\n')

        bits_squeezing = BitSqueezing(bit_depth=5)
        median_filter = MedianSmoothing2D(kernel_size=3)
        jpeg_filter = JPEGFilter(10)

        defense = nn.Sequential(jpeg_filter, bits_squeezing, median_filter, )

        adv_defended = defense(adv)
        with open('data/adv_defended.txt', 'a') as f:
            for data in adv_defended:
                dat = data[1].data.numpy().reshape(1, -1).ravel()
                # print(dat.shape)
                for d in dat:
                    f.write('%.3f\t' % d)
                f.write('\n')

        cln_defended = defense(cln_data_c)
        with open('data/cln_defended.txt', 'a') as f:
            for data in cln_defended:
                dat = data[1].data.numpy().reshape(1, -1).ravel()
                for d in dat:
                    f.write('%.3f\t' % d)
                f.write('\n')
