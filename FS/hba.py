import numpy as np
from numpy.random import rand
from FS.functionHo import Fun
import random
import time
import math
import sys
from numpy import linalg as LA

def fun(X):
    output = sum(np.square(X))
    return output

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='integer')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X

# Calculate fitness values for each Honey Badger.
def CaculateFitness1(X,Fun):
    fitness = Fun(X)
    return fitness

# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

# Sort the position of the Honey Badger according to fitness.
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

# Boundary detection Function.
def BorderCheck1(X,lb,ub,dim):
        for j in range(dim):
            if X[j]<lb[j]:
                X[j] = ub[j]
            elif X[j]>ub[j]:
                X[j] = lb[j]
        return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin

def Intensity(N,GbestPositon,X):
  epsilon = 0.00000000000000022204
  di = np.zeros(N)
  S = np.zeros(N)
  I = np.zeros(N)
  for j in range(N):
    if (j <= N):
      di[j]=LA.norm([[X[j,:]-GbestPositon+epsilon]])
      S[j]= LA.norm([X[j,:]-X[j+1,:]+epsilon])
      di[j] = np.power(di[j], 2)
      S[j]= np.power(S[j], 2)
    else:
      di[j]=[ LA.norm[[X[N,:]-GbestPositon+epsilon]]]
      S[j]=[LA.norm[[X[N,:]-X[1,:]+epsilon]]]
      di[j] = np.power(di[j], 2)
      S[j]= np.power(S[j], 2)    
  
    for i in range(N):
      n = random.random()
      I[i] = n*S[i]/[4*math.pi*di[i]]
    return I

def jfs(xtrain, ytrain, opts):
    
    # Parameters
    N        = opts['N']
    Max_iter = opts['T']
    fl    = -10
    ul    = 10
    thres = 0.5
    
    
    # Dimension
    dim = np.size(xtrain, 1)
    lb = fl*np.ones([dim, 1])
    ub = ul*np.ones([dim, 1])
    #if np.size(lb) == 1:
     #   ub = ub * np.ones([1, dim], dtype='float')
     #  lb = lb * np.ones([1, dim], dtype='float')
   
    # Initialize position & velocity
    X = init_position(N,dim,lb,ub)                    # Initialize the number of honey badgers

    #PRE
    Xgb   = np.zeros([1, dim], dtype='float')
    fitness = np.zeros([N, 1])
    Curve = np.zeros([Max_iter, 1])
    Xnew = np.zeros([N, dim])
    C = 2                                          # constant in Eq. (3)
    beta = 6   

    while t < Max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
    
        for i in range(N):
            fitness[i] = CaculateFitness1(X[i, :], fun)
            fitness, sortIndex = SortFitness(fitness)       # Sort the fitness values of honey badger.
            X = SortPosition(X, sortIndex)                  # Sort the honey badger.
            GbestScore = fitness[0]                         # The optimal value for the current iteration.
            GbestPositon = np.zeros([1, dim])
            GbestPositon[0, :] = X[0, :]
        
    # Store result
    Curve[0,t] = GbestPositon.copy()
    print("Iteration:", t + 1)
    print("Best (HBA):", Curve[0,t])
    t += 1                                # the ability of HB to get the food  Eq.(4)
    vec_flag=[1,-1]
    vec_flag=np.array(vec_flag)
    
    for t in range(Max_iter):
        #print("iteration: ",t)
        alpha=C*math.exp(-t/Max_iter);             # density factor in Eq. (3)
        I=Intensity(N,GbestPositon,X);           # intensity in Eq. (2)
        Vs=random.random()
        for i in range(N):
          Vs=random.random()
          F=vec_flag[math.floor((2*random.random()))]
          for j in range(dim):
            di=GbestPositon[0,j]-X[i,j]
            if (Vs <0.5):                           # Digging phase Eq. (4)
              r3=np.random.random()
              r4=np.random.randn()
              r5=np.random.randn()
              Xnew[i,j]=GbestPositon[0,j] +F*beta*I[i]* GbestPositon[0,j]+F*r3*alpha*(di)*np.abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5)));
            else:
              r7=random.random()
              Xnew[i,j]=GbestPositon[0,j]+F*r7*alpha*di;    # Honey phase Eq. (6)
          #print(di)
          Xnew[i,:] = BorderCheck1(Xnew[i,:], lb, ub, dim)
          tempFitness = CaculateFitness1(Xnew[i,:], Fun)
          if (tempFitness <= fitness[i]):
            fitness[i] = tempFitness               
            X[i,:] = Xnew[i,:] 
        for i in range(N):                         
          X[i,:] = BorderCheck1(X[i,:], lb, ub ,dim)
        Ybest,index = SortFitness(fitness)               # Sort fitness values.
        if (Ybest[0] <= GbestScore):                          
          GbestScore = Ybest[0]     # Update the global optimal solution.
          GbestPositon[0, :] = X[index[0], :]           # Sort fitness values 
        Curve[t] = GbestScore
    
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    pso_data = {'sf': sel_index, 'c': Curve, 'nf': num_feat}
    
    return pso_data  
    