#!/usr/bin/python

from __future__ import unicode_literals
import json
from matplotlib.pyplot import *
from numpy import *
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
rc('font',**{'family':'serif','serif':['cm']})

class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of 
    magnitude"""
    def __init__(self, order_of_mag=0, format=None, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag

        self.set_useOffset(useOffset)
        self._usetex = rcParams['text.usetex']
        self._useMathText = useMathText
        self.orderOfMagnitude = 0
        self.format = ''
        self._scientific = True
        self._powerlimits = rcParams['axes.formatter.limits']
        if format is None:
            self._useLocale = rcParams['axes.formatter.use_locale']
        else:
            self._useLocale = True
            self.format = format
            self.custom_format = True

    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

    def _set_format(self):
        # set the format string to format all the ticklabels
        # The floating point black magic (adding 1e-15 and formatting
        # to 8 digits) may warrant review and cleanup.
        if not self.custom_format:
            locs = (np.asarray(self.locs)-self.offset) / 10**self.orderOfMagnitude+1e-15
            sigfigs = [len(str('%1.8f'% loc).split('.')[1].rstrip('0')) \
                           for loc in locs]
            sigfigs.sort()
            self.format = '%1.' + str(sigfigs[-1]) + 'f'
            if self._usetex:
                self.format = '$%s$' % self.format
            elif self._useMathText:
                self.format = '$\mathdefault{%s}$' % self.format

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

fig = figure(figsize = (5, 5))
fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
ax = fig.add_subplot(111, axisbg='w')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$h$')
# ax.set_xlim([0.0,0.03])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()

for i in range(0,len(sim_data)):
    if i == 0:
        ax.plot(data[i][:,0],data[i][:,1],'r-',label=r'ref')
        ax.plot(sim_data[i][:,0],sim_data[i][:,1],'b-',label=r'model')
    else:
        ax.plot(data[i][:,0],data[i][:,1],'r-')
        ax.plot(sim_data[i][:,0],sim_data[i][:,1],'b-')
        
#show()

ax.legend(loc=1)

fig.savefig('h.png')
