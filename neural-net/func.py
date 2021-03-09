from scipy import linalg
import numpy as np
from numpy import loadtxt
from numpy import random
import math
from sklearn.metrics import accuracy_score
from main import *

f=open("ecoli.txt","r+")

dataframe=f.readlines()

makeCol = makeColumn(dataframe)

classSet = list(dict.fromkeys(makeCol[8]))

poplist = []

for i in range(len(makeCol[8])):
    if makeCol[8][i] == 'imL' or makeCol[8][i] == 'imS':
        poplist.insert(0,i)

for p in poplist:
    for j in range(9):
        makeCol[j].pop(p)

makeCol.pop(4)

makeCol[7] = changeName(makeCol[7])

nn = init_net(6,10,6)

dataset = []
for i in range(len(makeCol[7])):
    dataset.append(get_data(makeCol,i))

copy = dataset

fold = []
ds = len(dataset)
size = int(len(dataset) / 10.0)
for i in range(10):
    x = []
    for s in range(size):
        index = random.randint(ds - 1)
        d = dataset.pop(index)
        x.append(d)
        ds -= 1
    fold.append(x)

score = []

for i in range(10):
    ts = fold[i]
    tn = []
    for j in range(10):
        if j != i:
            for k in range(len(fold[j])):
                tn.append(fold[j][k])
    train(nn,tn,0.8,20,6)

    a = []
    p = []

    for i in ts:
        pre = predict(nn,i)
        p.append(pre)
        a.append(i[-1])

    acc = accuracy(a,p)
    score.append(acc)

print(score)
print("Mean = %.3f" % (sum(score)/float(len(score))))