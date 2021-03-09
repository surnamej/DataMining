from scipy import linalg
import numpy as np
from numpy import loadtxt
from numpy import random
import math

def makeColumn(mValue):
    vaCol = [[],[],[],[],[],[],[],[],[]]
    for mVaStr in mValue:
        vaStr = mVaStr.split()
        for i in range(9):
            vaCol[i].append(vaStr[i])
    return vaCol

def changeName(col):
    new_col = []
    classSet = list(dict.fromkeys(col))
    for i in range(len(col)):
        new_col.append(classSet.index(col[i]))
    return new_col

def init_net(input,hidden,output):
    network = list()
    hidden_layer = [{'weights':[random.rand() for i in range(input + 1)]} for i in range(hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random.rand() for i in range(hidden + 1)]} for i in range(output)]
    network.append(output_layer)
    return network

def init_net2(input,h1,h2,output):
    network = list()
    hidden1_layer = [{'weights':[random.rand() for i in range(input + 1)]} for i in range(h1)]
    network.append(hidden1_layer)
    hidden2_layer = [{'weights':[random.rand() for i in range(h1 + 1)]} for i in range(h2)]
    network.append(hidden2_layer)
    output_layer = [{'weights':[random.rand() for i in range(h2 + 1)]} for i in range(output)]
    network.append(output_layer)
    return network

def sigmoid(v):
    s = 1/(1+math.exp(-v))
    return(s)

def Nout(x,w):
    act = w[-1]
    for i in range(len(w) - 1):
        act += w[i] * x[i]
    return act
    
def f_propagate(nn,x):
    inputs = x
    for layer in nn:
        new_inputs = []
        for n in layer:
            act = Nout(n['weights'],inputs)
            n['output'] = sigmoid(act)
            new_inputs.append(n['output'])
        inputs = new_inputs
    return inputs

def get_data(table,index):
    x = []
    for i in [1,2,3,4,5,6]:
        x.append(float(table[i][index]))
    x.append(int(table[7][index]))
    return x

def tranfer_d(o):
    return o * (1 - o)

def b_propagate(nn,expected):
    for i in reversed(range(len(nn))):
        layer = nn[i]
        errs = list()
        if i != len(nn) - 1:
            for j in range(len(layer)):
                err = 0.0
                for n in nn[i + 1]:
                    err += (n['weights'][j] * n['error'])
                errs.append(err)
        else:
            for j in range(len(layer)):
                n = layer[j]
                errs.append(expected[j] - n['output'])
        for j in range(len(layer)):
            n = layer[j]
            n['error'] = errs[j] * tranfer_d(n['output'])

def up_weights(nn,x,l):
    for i in range(len(nn)):
        inputs = x[:-1]
        if i != 0:
            inputs = [n['output'] for n in nn[i-1]]
        for n in nn[i]:
            for j in range(len(inputs)):
                n['weights'][j] += l * n['error'] * inputs[j]
            n['weights'][-1] += l * n['error']

def train(nn,train_data,l,epoch,output_num):
    for e in range(epoch):
        sum_e = 0.0
        for x in train_data:
            outputs = f_propagate(nn,x)
            expected = [0 for i in range(output_num)]
            expected[x[-1]] = 1
            sum_e += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            b_propagate(nn,expected)
            up_weights(nn,x,l)
        print('epoch = %d' %e)
        print('learning rate = %.3f' %l)
        print('error = %.3f' %sum_e)
        print('-----------------------------------------')

def predict(nn,x):
    outputs = f_propagate(nn,x)
    return outputs.index(max(outputs))

def accuracy(actual,predict):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def fold_validate(data):
    split = list()
    copy = data
    fold_size = int(len(data) / 10.0)
    for i in range(10):
        fold = list()
        while len(fold) < fold_size:
            index = random.randint(len(copy))
            a = copy.pop(index)
            fold.append(a)
        split.append(fold)
    return split

def evaluate(data, n_fold,l,e,o):
    folds = fold_validate(data)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            copy = list(row)
            test_set.append(copy)
            copy[-1] = None
        predicted = b_propagate2(train_set,test_set,l,e,o)
        actual = [row[-1] for row in fold]
        acc = accuracy(actual,predicted)
        scores.append(acc)
    return scores

def b_propagate2(t, test, l, epoch, hidden):
    nn = init_net(6,hidden,6)
    train(nn,t,l,epoch,6)
    prediction = list()
    for row in test:
        pre = predict(nn, row)
        prediction.append(pre)
    return(prediction)


