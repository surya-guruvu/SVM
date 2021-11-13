from libsvm.svmutil import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix,solvers
import random
import time

#C: regularization parameter
#tol: numerical tolerance
#max passes: max # of times to iterate over α’s without changing

def lin_ker(Xi,Xj):
    return np.dot(Xi,Xj)

def lin_class(lam, x, X, Y, b):
    sum=0
    for i in range(len(Y)):
        sum+=lam[i] * Y[i] * (np.dot(X[i], x))
    sum+=b
    return sum

def findL(Yi,Yj,lami,lamj,C):
    if Yi!= Yj:
        return max(0, lamj - lami)
    else:
        return max(0, lami + lamj - C)

def findH(Yi,Yj,lami,lamj,C):
    if(Yi==Yj):
        return min(C,lami+lamj)
    else:
        return min(C,C-lami+lamj)

def neta(Xi,Xj):
    return 2 * (np.inner(Xi, Xj)) - np.inner(Xi, Xi) - np.inner(Xj, Xj)

def lin_error(Xi, Yi, lam, X,Y,b):
    fxi = lin_class(lam, Xi, X, Y, b)
    Ei = fxi - Yi
    
    return Ei

def compute_lamj(Yj,lamj, Ei, Ej, neta):
    return lamj - ((Yj * (Ei - Ej)) / neta)

def clip_lamj(lamj,L,H):
    if(lamj>H):
        return H
    elif(lamj<L):
        return L
    else:
        
        return lamj

def compute_lami(lami, Yi, Yj, lamj_old, lamj):
    return lami + (Yi * Yj) * (lamj_old - lamj)

def findbs(b, Xi, Yi, lami, lamj, Ei, Ej, Xj, Yj, lami_old, lamj_old):
    b1 = (b - Ei)-(Yi*(lami - lami_old) * np.inner(Xi, Xi)) - (Yj * (lamj - lamj_old) * np.inner(Xi, Xj)) 
    b2 = (b - Ej)-(Yi*(lami - lami_old) * np.inner(Xi, Xj)) - (Yj * (lamj - lamj_old) * np.inner(Xj, Xj)) 

    return b1, b2

def compute_b(b1, b2, lami, lamj, C):
    if lami > 0 and lami < C:
        return b1
    elif lamj > 0 and lamj < C:
        return b2
    else:
        return (b1 + b2) / 2


def gen_j(m, i):
    random.seed(time.time())
    
    j = random.randint(0, m-1)

    while i == j:
        j = random.randint(0, m-1)

    return j

def SMO(C, tol, max_passes, X, Y):
    m = len(Y)    
    b = 0
    lams = np.zeros(m)
    lams_old = np.zeros(m)
    passes = 0

    while(passes < max_passes):
        changedLam=0
        for i in range(m):
            Ei =lin_error(X[i], Y[i], lams, X,Y,b) #calculate_error(X[i], y[i], alphas[i], X, b)

            #print(Ei)
        
            if ((Y[i] * Ei < (-1 * tol) and lams[i] < C) or (Y[i] * Ei > tol and lams[i] > 0)):
                j = gen_j(m, i)
            
                Ej = lin_error(X[j], Y[j], lams, X,Y,b)  #calculate_error(X[j], y[j], alphas[j], X, b)
                #print(Ej)
                
                lams_old[i], lams_old[j] = lams[i], lams[j]
                
                L = findL(Y[i],Y[j],lams[i],lams[j],C)
                H = findH(Y[i],Y[j],lams[i],lams[j],C) 
                
                if L == H:
                    continue

                eta = neta(X[i], X[j])

                if eta >= 0:
                    continue
                    
                lam1=compute_lamj(Y[j],lams[j], Ei, Ej, eta)

                lams[j] = clip_lamj(lam1, L, H)
                

                if abs(lams[j] - lams_old[j]) < 1e-3:
                    continue
                
                lams[i] = compute_lami(lams[i], Y[i], Y[j], lams_old[j], lams[j])
                
                b1,b2=findbs(b, X[i], Y[i], lams[i], lams[j], Ei, Ej, X[j], Y[j], lams_old[i], lams_old[j])
            
                b = compute_b(b1, b2, lams[i], lams[j], C)
        
                changedLam += 1 
        
        if changedLam == 0:
            passes += 1
        else:
            passes = 0

	return lams, b

lam,b=SMO(1/(2**(-5.63)), 1e-7, 25, train_data, label_train)
y_predict=predict(lam,b,test_data)
y_predict=np.sign(y_predict)
label_test=np.array(label_test)

Accuracy_table=np.ravel(label_test==y_predict)
count=0
for k in range(len(label_test)):
    if(Accuracy_table[k]==True):
        count+=1
print(count/4)