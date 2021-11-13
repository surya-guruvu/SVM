from libsvm.svmutil import *
import numpy as np
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

#Labels used are 2.0 and 5.0

df3=df1.Column25
label=np.array(df3)
label_train=label[0:177]
label_test=label[177:577]

df1=df1.iloc[:,0:10:1]
data = df1.values #To convert data frame into numpy array
train_data=data[0:177]
test_data=data[177:577]

prob = svm_problem(label_train, train_data)
c=[]
Accuracy=[]
max=0
j=0
for i in range(-20,20):
    c.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_train, train_data, m) 
    Accuracy.append(p_acc[0])

    if(max<p_acc[0]):
        max=p_acc[0]
        j=i
    
plt.plot(c,Accuracy,label='train')


c=[]
Accuracy=[]
max=0
j=0
for i in range(-20,20):
    c.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
    Accuracy.append(p_acc[0])

    if(max<p_acc[0]):
        max=p_acc[0]
        j=i
    
plt.plot(c,Accuracy,label='test')
plt.xlabel('log2(c)');
plt.ylabel('Accuracy');
plt.legend(loc='best')
print('Value of coarse log2(c), for maximum accuracy: {}'.format(j))

c=[]
Accuracy=[]
max=0
j=0
i=-8
while(i<-6):
    c.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
#svm_type = m.get_svm_type()
    p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
    Accuracy.append(p_acc[0])
    #print(svm_type)
    #print(m)
    if(max<p_acc[0]):
        max=p_acc[0]
        j=i
    i+=0.001
print('Value of fine log2(c), for maximum accuracy: {}'.format(j))
print('The percentage of predicted values: {}'.format(max))



plt.show()
