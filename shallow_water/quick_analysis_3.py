#!/usr/bin/python

import json
from numpy import *
import matplotlib.pyplot as plt

f = open('phi_ic_2.json','r')
data = array([json.loads(line) for line in f])#[-1]
f.close()

print data.shape

dx = 2.5e-2
L = 1.0
x = [i*dx for i in range(int(L/dx) + 1)]

data_x = [x for set in data]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.T)

plt.show()

f = open('deposit_data_2.json','r')
data = array([json.loads(line) for line in f])#[-1]
f.close()

print data.shape

dx = 2.5e-2
L = 1.0
x = [i*dx for i in range(int(L/dx) + 1)]

data_x = [x for set in data]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.T)

plt.show()
