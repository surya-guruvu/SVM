from libsvm.svmutil import *
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix,solvers

df = pd.read_csv('C:/Users/surya/Downloads/2019EE10481/train_set.csv',header = None,prefix='Column') 

df1=df.Column25
label=np.array(df1)
label_train=label[0:6000]
label_test=label[6000:8000]

df=df.iloc[:,0:25:1]
data = df.values #To convert data frame into numpy array
train_data=data[0:6000]
test_data=data[6000:8000]

prob = svm_problem(label_train, train_data)
c0=[]
Accuracy0=[]
max0=0
c=[]
Accuracy=[]
max=0
j=0
j0=0
for i in range(-10,10):
    c0.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_train, train_data, m) 
    Accuracy0.append(p_acc[0])

    if(max0<p_acc[0]):
        max0=p_acc[0]
        j0=i
for i in range(-10,10):
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
i=-3
while(i<-1):
    c.append(i)
    param = svm_parameter('-t 0 -c {} -b 0'.format(2**i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(label_test, test_data, m) 
    Accuracy.append(p_acc[0])

    if(max<p_acc[0]):
        max=p_acc[0]
        j=i

    i+=0.1


print('Value of coarse log2(c), for maximum accuracy: {}'.format(j))
print('Percentage of Samples for which output is predicted correctly: {}'.format(max))

df1 = pd.read_csv('C:/Users/surya/Downloads/409-Ass2/test_set.csv',header = None,prefix='Column')

label_dummy=np.ones(2000)

df=df1.iloc[:,0:25:1]
data1 = df.values #To convert data frame into numpy array

prob = svm_problem(label_train, train_data)
param = svm_parameter('-t 0 -c {} -b 0'.format(2**(-1.8)))
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(label_dummy, data1, m) 

p_label=np.array(p_label)
print(p_label)

arr=np.arange(2001)[1:2001]

arr = arr.astype('str')
print(arr)

L={"Id":arr,"Class":p_label}
df_out=pd.DataFrame(L)
print(df_out)

with open('foo.csv','w+') as f:
    f.write('Id,Class\n')
    for i in range(p_label.shape[0]):
        if (i+1<1000):
            f.write('{},{:d}\n'.format(str(i+1),int(p_label[i])))
        else:
            f.write('\"{:01d},{:03d}\",{:d}\n'.format((i+1)//1000,(i+1)%1000,int(p_label[i])))
            #f.write('\"{:01d},{:03d}\",{:d}\n'.format((i+1)//1000, (i+1)%1000, int(p_label[I])))

plt.show()