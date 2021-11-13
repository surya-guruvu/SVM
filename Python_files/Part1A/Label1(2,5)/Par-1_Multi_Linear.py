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
c0=[]
Accuracy0=[]
max0=0
c=[]
Accuracy=[]
max=0
j=0
j0=0
for i in range(-20,15):
    c0.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_train, train_data, m) 
    Accuracy0.append(p_acc[0])

    if(max0<p_acc[0]):
        max0=p_acc[0]
        j0=i
for i in range(-20,15):
    c.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
    Accuracy.append(p_acc[0])

    if(max<p_acc[0]):
        max=p_acc[0]
        j=i

plt.plot(c0,Accuracy0)
plt.plot(c,Accuracy)

print('Value of coarse log2(c), for maximum accuracy: {}'.format(j))

prob = svm_problem(label_train, train_data)
c=[]
Accuracy=[]
max=0
j=0
i=-6
while(i<-4):
    c.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
    Accuracy.append(p_acc[0])

    if(max<p_acc[0]):
        max=p_acc[0]
        j=i

    i+=0.1
print('Value of fine log2(c), for maximum accuracy: {}'.format(j))
print('Percentage of Samples for which output is predicted correctly: {}'.format(max))

plt.show()