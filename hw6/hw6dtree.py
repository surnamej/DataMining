from scipy import linalg
import numpy as np
from numpy import loadtxt
import math
from sklearn.metrics import accuracy_score
from hw6func import *
f=open("ecoli.txt","r+")
dataframe=f.readlines()
#print(type(X))
#print(type(X[0]))
#print(X)
makeCol = makeColumn(dataframe)
#for j in makeCol:
#    print(checkMissValue(j))
classSet = list(dict.fromkeys(makeCol[8]))
for i in [1,2,3,4,5,6,7]:
    print(i)
    print("Gain = ",getGain(i,makeCol))

def extraction(mcg,gvh,lip,chg,aac,alm1,alm2):
    classP = "NA"
    if float(alm1) <= 0.5:
        if float(mcg) <= 0.5:
            classP = "cp"     
        else:
            if float(gvh) <= 0.5:
                classP = "cp"
            else:
                if float(aac) <= 0.5:
                    if float(lip) <= 0.5:
                        classP = "pp"
                    else:
                        classP = "imL"
                else:
                    classP = "om"
    else:
        if float(alm2) <= 0.5:
            if float(aac) <= 0.5:
                if float(mcg) <= 0.5:
                    classP = "im"
                else:
                    if float(lip) <= 0.5:
                        classP = "pp"
                    else:
                        classP = "omL"
            else:
                if float(lip) <= 0.5:
                    classP = "om"
                else:
                    classP = "omL"
        else:
            if float(mcg) <= 0.5:
                if float(lip) <= 0.5:
                    classP = "im"
                else:
                    classP = "imU"
            else:
                if float(lip) <= 0.5:
                    if float(aac) <= 0.5:
                        if float(gvh) <= 0.5:
                            classP = "imU"
                        else:
                            classP = "pp"
                    else:
                        if float(gvh) <= 0.5:
                            classP = "imU"
                        else:
                            classP = "im"
                else:
                    classP = "im"
    print("Value is ",classP)
    return classP

count = 0
true = []
pred = []
for i in range(len(makeCol[8])):
    value = makeCol[8][i]
    pre = extraction(
        makeCol[1][i], 
        makeCol[2][i], 
        makeCol[3][i],
        makeCol[4][i],
        makeCol[5][i],
        makeCol[6][i],
        makeCol[7][i])
    true.append(value)
    pred.append(pre)
    if value == pre:
        count+=1


acc = count/len(makeCol[8])
print(count)
print("Accurancy is ",acc*100,"%")