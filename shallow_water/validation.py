#!/usr/bin/python

import json
from matplotlib.pyplot import *
from numpy import *

fname = 'bonnecaze_1layer_h.json'
file = open(fname,'r')

data = []
for line in file.readlines():
    data.append(array(json.loads(line)))

fname = 'results/default_T.json'
file = open(fname,'r')
T = []
for line in file.readlines():
        T.append(array(json.loads(line)))

fname = 'results/default_h.json'
file = open(fname,'r')
H = []
for line in file.readlines():
            H.append(array(json.loads(line)))

fname = 'results/default_x_N.json'
file = open(fname,'r')
X = []
for line in file.readlines():
            X.append(array(json.loads(line)))

indices = [1,2,3,4,6,8,10,15,30,50,100,150]
sim_data = []
for i in indices:
    dx = X[i][0]/(len(H[i])-1)
    x = [j*dx for j in range(len(H[i]))]
    sim_data.append(array(zip(x, H[i])))

for i in range(0,len(sim_data)):
    plot(data[i][:,0],data[i][:,1],'r-')
    plot(sim_data[i][:,0],sim_data[i][:,1],'b-')
show()
