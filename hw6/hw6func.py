from scipy import linalg
import numpy as np
import random
import math

def makeColumn(mValue):
    vaCol = [[],[],[],[],[],[],[],[],[]]
    for mVaStr in mValue:
        vaStr = mVaStr.split()
        for i in range(9):
            vaCol[i].append(vaStr[i])
    return vaCol
            
def checkMissValue(varCol):
    return varCol.count('NA')

def count(cName,col,mCol,num):
    more = 0
    for i in range(len(mCol[col])):
        if mCol[8][i] == cName:
            if float(mCol[col][i]) > num:
                more+=1
    return more

def count2(cName,col,mCol,num):
    more = 0
    for i in range(len(mCol[col])):
        if mCol[8][i] == cName:
            if float(mCol[col][i]) <= num:
                more+=1
    return more

def getGain(col,columnList):
    return entropy(columnList) - info(col,columnList)

def getGainfil(col,columnList):
    return entropy(filColumn(columnList)) - info(col,filColumn(columnList))

def filColumn(columnList):
    new = [[],[],[],[],[],[],[],[],[]]
    for i in range(len(columnList[8])):
        if ((float(columnList[6][i]) > 0.5) ) :
            for j in range(len(new)):
                new[j].append(columnList[j][i])
    return new

def entropy(columnList):
    length = len(columnList[8])
    kList = []
    for k in list(dict.fromkeys(columnList[8])):
        counter = len(list(filter(lambda x:x == k,columnList[8])))
        kList.append(counter)
    return etp(kList)

def info(col,columnList):
    length = len(columnList[8])
    a = []
    b = []
    more = 0
    less = 0
    for k in list(dict.fromkeys(columnList[8])):
        add = count(k,col,columnList,0.5)
        counter = len(list(filter(lambda x:x == k,columnList[8])))
        a.append(add)
        b.append(counter - add)
    ca = (sum(a)/len(columnList[col]))
    cb = (sum(b)/len(columnList[col]))
    ans = (ca*etp(a)) + (cb*etp(b))
    return ans

def etp(l):
    s = sum(l)
    ans = 0
    for i in l:
        if i != 0:
            c = i/s
            m = math.log(c,2)
            ans+= -(c*m)
    return ans    



