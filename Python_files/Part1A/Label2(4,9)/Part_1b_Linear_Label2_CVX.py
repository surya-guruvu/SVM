df = pd.read_csv('C:/Users/surya/Downloads/2019EE10481/2019EE10481.csv',header = None,prefix='Column') 
labelsAll= df['Column25'].unique() #unique labels
labels=labelsAll[0:2] # labels for Binary Clasification

df1=df[(df.Column25==labels[0]) | (df.Column25==labels[1])]

#Labels used are 2.0 and 5.0

df3=df1.Column25
label=np.array(df3)

for i in range(len(label)):
    if(label[i]==4.0):
        label[i]=1.0
    else:
        label[i]=-1.0
        
label_train=label[0:180]
label_test=label[180:580]

df1=df1.iloc[:,0:10:1]
data = df1.values #To convert data frame into numpy array
train_data=data[0:180]
test_data=data[180:580]

N=len(label_train)
X=train_data
y=label_train

K = np.zeros((N,N))

# Generating Kernel Function (Gram-Matrix)
for i in range(N):
    for j in range(N):
        K[i,j] = np.dot(X[i], X[j])
        
# Generating P and q
P=cvxopt.matrix(np.outer(y,y) * K)
q = cvxopt.matrix(np.ones(N) * -1)


C=2**(0.88)

#Generating conditions (i.e., o<=lambd_i<=c)
G1 = np.diag(np.ones(N) * -1)
G2 = np.identity(N)
G = cvxopt.matrix(np.vstack((G1, G2)))

h1 = np.zeros(N)
h2 = np.ones(N) * C
h = cvxopt.matrix(np.hstack((h1, h2)))


#Generating conditions (i.e., sigma())
A=cvxopt.matrix(y, (1,N))
b = cvxopt.matrix(0.0)

solution = cvxopt.solvers.qp(P, q, G, h,A,b)

a = np.ravel(solution['x'])
sv= a>1e-4
sv_index = np.arange(len(a))[sv]
sv_lambda= a[sv]  # Represent lambdas of all support vectors
sv_fea= X[sv]     # Represent all support vectors
sv_label = y[sv]


# Refer SVM.pdf for more clarity
#Validation

for n in range(len(sv_lambda)):
    b += sv_label[n]
    b -= np.sum(sv_lambda * sv_label * K[sv_index[n],sv])
b /= len(sv_lambda)


y_predict = np.zeros(len(test_data))
for i in range(len(test_data)):
    s = 0
    for a, sv_y, sv in zip(sv_lambda, sv_label, sv_fea):
        s += a * sv_y * np.dot(test_data[i], sv)
    y_predict[i] = s
y_predict=np.array(y_predict)+b


y_predict=np.sign(y_predict)

label_test=np.array(label_test)

Accuracy_table=np.ravel(label_test==y_predict)

count=0
for i in range(len(label_test)):
    if(Accuracy_table[i]==True):
        count+=1
        
print('No of Samples for which output is predicted correctly: {}'.format(count/4))