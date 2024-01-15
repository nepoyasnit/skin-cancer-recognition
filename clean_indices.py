import os
import pickle
import numpy as np
import pandas as pd
import random
from itertools import permutations 


REMOVED_INDICES = [17257, 23523, 1609, 8458, 8535, 25291, 23971, 19824, 24269, 18154, 23511, 14954, 14568, 13745, 24712,
 18476, 21588, 20376, 22221, 20866, 24558, 17816, 25279, 17535, 24431, 20931, 22229, 18479, 18295, 24581,
 22164, 18627, 22720, 19282, 25177, 25229, 23034, 19012, 23191, 19699, 19664, 23954, 24430, 20807, 24787,
 25161, 20520, 21751, 21988, 24025, 23919, 24281]

with open('/home/konovalyuk/data/isic/indices_isic2019.pkl','rb') as f:
    indices = pickle.load(f)

indices['trainIndCV'][0] = np.setdiff1d(indices['trainIndCV'][0], REMOVED_INDICES)
print(indices['trainIndCV'][0][0])


sdir = 'isic2019/images/official/'
flist = os.listdir(sdir)

for i in range(len(flist)):
    flist[i] = flist[i][:-4]


flist = pd.Series(flist)
labels = pd.read_csv('isic2019/labels/official/binary_labels2019.csv')

print(len(flist), len(labels))

indices = list(labels.index)

random.shuffle(indices)
indices = np.array(indices)
train, test = indices[int(len(indices)*0.2):], indices[:int(len(indices)*0.2)]
print(len(train), len(test), len(indices))
train = np.array_split(train, 5)

print(len(train))