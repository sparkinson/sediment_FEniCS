#!/usr/bin/python

import json
from numpy import *
import matplotlib.pyplot as plt

f = open('phi_ic_adj1.json','r')
data = array([json.loads(line) for line in f])
f.close()

f = open('phi_ic.json','r')
target = array(json.loads(f.readline()))
f.close()

print data.shape
print target.shape

dx = 2.5e-2
L = 1.0
x = [i*dx for i in range(int(L/dx) + 1)]

data_x = [x for set in data]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(target, 'k--', linewidth=3)
try:
    ax.plot(data[:-1].T, linewidth=0.5)
    ax.plot(data[-1].T, linewidth=1.5)
except:
    pass

plt.show()

f = open('h_ic_adj1.json','r')
data = array([json.loads(line) for line in f])
f.close()

f = open('h_ic.json','r')
target = array(json.loads(f.readline()))
f.close()

print data.shape
print target.shape

dx = 2.5e-2
L = 1.0
x = [i*dx for i in range(int(L/dx) + 1)]

data_x = [x for set in data]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(target, 'k--', linewidth=3)
try:
    ax.plot(data[:-1].T, linewidth=0.5)
    ax.plot(data[-1].T, linewidth=1.5)
except:
    pass

plt.show()
