from libsvm.svmutil import *
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix,solvers

df = pd.read_csv('C:/Users/surya/Downloads/2019EE10481/2019EE10481.csv',header = None,prefix='Column') 

df1=df.Column25
label=np.array(df1)
label_train=label[0:2000]
label_test=label[2000:3000]

df=df.iloc[:,0:25:1]
data = df.values #To convert data frame into numpy array
train_data=data[0:2000]
test_data=data[2000:3000]

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

prob = svm_problem(label_train, train_data)

Accuracy=[]
max=0
j=0
j1=0
k=-5
while(k<-3):
    i=0
    while(i<2):
        param = svm_parameter('-t 2 -g {} -c {} -b 0'.format(2**k,2**i))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
        Accuracy.append(p_acc[0])
        if(max<p_acc[0]):
            max=p_acc[0]
            j=i
            j1=k
        i+=1
    k+=0.01
    

print('Value of coarse log2(c), for maximum accuracy: {}'.format(j))
print('Value of coarse log2(gamma), for maximum accuracy: {}'.format(j1))
print('The percentage of predicted values is equal to {} %'.format(max))