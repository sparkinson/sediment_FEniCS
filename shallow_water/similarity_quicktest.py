import json
from matplotlib.pyplot import *
from numpy import *
fname = 'similarity_convergence.json'
file = open(fname,'r')
line = file.readline()
data = array(json.loads(line))
dX = [1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64]
for d in data:
    fig = figure(figsize = (5, 5))
    fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax = fig.add_subplot(111, axisbg='w')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot(dX, d)
    fig.show()
