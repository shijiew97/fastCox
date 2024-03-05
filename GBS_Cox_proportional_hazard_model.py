# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:57:17 2021

@author: Shijie Wang
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import warnings
import scipy.stats
import scipy
from scipy.stats import binom
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
warnings.filterwarnings("ignore")

#############################################
#############################################

def dirichlet(s, n, m):
    # generate wi ~ Dirichlet(1/n,...,1/n): m rows and n samples
    w = np.random.exponential(scale = 1, size = s*m).reshape(m,s)
    w_mean = np.repeat(np.mean(w, axis=1),s).reshape(m,s)
    w_s = w / w_mean
    if s == n:
        w_m = w_s
    else:
        w_m = np.repeat(w_s, n/s).reshape(m,n)
    return w_m
  
if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

def trans_cuda(obj):
    # transform data type
    obj = np.asarray(obj)
    obj = torch.from_numpy(obj)
    obj = obj.to(device, dtype = torch.float)
    return obj

#############################################
#############################################
#df = pd.read_csv("C:\\Users\\Shijie Wang\\Desktop\\Research Notes\\GBS and NPMLE simulation\\lung.csv")
#covariates = ["age", "sex",  "ph.karno", "ph.ecog", "wt.loss"]

df = pd.DataFrame(r.df)
X = pd.DataFrame(r.X)

t = df["time"]
status = df["status"]

t_K = t[status == 2]
X_K = X[status == 2]

p = X.shape[1]
N = int(r.N)
K = np.sum(status == 2)
n = K
s = K
size = int(r.size) # size for bootstrap samples


#############################################
#############################################

class Net(nn.Module):
  def __init__(self, hidden_size): # hidden_size = # of nodes in each layer
    super(Net, self).__init__()
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(K, hidden_size) # input dimension is 2: (wi,zi)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc_out = nn.Linear(hidden_size, p) # output is the boostrap distribution of theta
  
  def forward(self, X1):
    out = self.relu( self.fc1(X1) )
    out = self.relu( self.fc2(out) )
    out = self.relu( self.fc3(out) )
    out = self.fc_out(out)
    return out

#############################################
#############################################
    
x = trans_cuda(X) 
x_K = trans_cuda(X_K) 
iteration = 1200# iteration requires to converge
hidden_size = 500 
lr0 = 0.0005 # learning rate
NN = Net(hidden_size).to(device)
optimizer = torch.optim.Adam(NN.parameters(), lr=lr0)
LOSS = torch.zeros(iteration)

#############################################
#############################################

def Loss(x_K, w, x, t_K, t):
    # Loss function 
    w = trans_cuda(w)
    B = NN(w)
    X_B = torch.matmul(B, x_K.transpose(1,0))
    exp_X_B = torch.zeros(X_B.shape).to(device)
    for i in range(len(x_K)):
        X_i = x[t_K.iloc[i]<t]
        X_Bi = torch.matmul(B, X_i.transpose(1,0))
        exp_X_B[:,i] = torch.log(torch.sum(torch.exp(X_Bi),1))
    loss = - torch.mean(torch.sum(w * (X_B - exp_X_B), 1))
    return loss

#############################################
#############################################
    
time_start = time.clock()
for it in range(iteration):
    lr = lr0/(float(it) ** 0.5 + 1) # change of learning rate
    w = dirichlet(s=n, n=K, m=N)
    loss = Loss(x_K = x_K, x = x, w = w, t_K = t_K, t = t)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    LOSS[it] = loss.item()
    if (it)%100==0: print('iteration: {}/{}, loss: {:.4f}'.format(it, iteration, loss.item()))
time_elapsed = time.clock() - time_start
print('time for each iteration is: {:.4f}'.format(time_elapsed/iteration))

#############################################
#############################################

w = dirichlet(s=n, n=K, m=size)
B = NN(trans_cuda(w))
B = pd.DataFrame(B.cpu().detach().numpy())
