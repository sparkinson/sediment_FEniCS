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

fname = 'j_log2.json'
file = open(fname,'r')

J = array(json.loads(file.readline()))

fig = figure(figsize = (6, 3))
fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
ax = fig.add_subplot(111, axisbg='w')

ax.set_xlabel(r'iterations')
ax.set_ylabel(r'$J$')
# ax.set_xlim([0.0,0.03])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()

ax.plot(J,'b-')

ax.set_yscale('log')
# ax.set_xscale('log')
        
#show()

# ax.legend(loc=1)

fig.savefig('J.png')
