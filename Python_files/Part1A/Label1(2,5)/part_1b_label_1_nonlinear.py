from libsvm.svmutil import *
import numpy as np
from numpy import linalg
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix,solvers

df = pd.read_csv('C:/Users/surya/Downloads/2019EE10481/2019EE10481.csv',header = None,prefix='Column') 
labelsAll= df['Column25'].unique() #unique labels
labels=labelsAll[0:2] # labels for Binary Clasification

df1=df[(df.Column25==labels[0]) | (df.Column25==labels[1])]

df3=df1.Column25
label=np.array(df3)
label_train=label[0:177]
label_test=label[177:577]

df1=df1.iloc[:,0:10:1]
data = df1.values #To convert data frame into numpy array
train_data=data[0:177]
test_data=data[177:577]

prob = svm_problem(label_train, train_data)

Accuracy=[]
max=0
j=0
j1=0
for k in range(-10,10):
    for i in range(-10,10):
        param = svm_parameter('-t 2 -g {} -c {} -b 0'.format(2**k,2**i))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
        Accuracy.append(p_acc[0])
        if(max<p_acc[0]):
            max=p_acc[0]
            j=i
            j1=k
    

print('Value of coarse log2(c), for maximum accuracy: {}'.format(j))
print('Value of coarse log2(gamma), for maximum accuracy: {}'.format(j1))
print('The percentage of predicted values')

prob = svm_problem(label_train, train_data)

Accuracy=[]
max=0
j=0
j1=0
k=-4
while(k<-2):
    i=-1
    while(i<1):
        param = svm_parameter('-t 2 -g {} -c {} -b 0'.format(2**k,2**i))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
        Accuracy.append(p_acc[0])
        if(max<p_acc[0]):
            max=p_acc[0]
            j=i
            j1=k
        i+=0.1
    k+=0.1
    

print('Value of coarse log2(c), for maximum accuracy: {}'.format(j))
print('Value of coarse log2(gamma), for maximum accuracy: {}'.format(j1))
print('The percentage of predicted values is equal to {} %'.format(max))