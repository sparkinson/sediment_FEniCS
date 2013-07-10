from __future__ import unicode_literals
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import json

import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
plt.rc('font',**{'family':'serif','serif':['cm']})

class Plotter():

    def __init__(self, model, rescale = True, file = 'results/default'):

        self.rescale = rescale
        self.save_loc = file

        if model.show_plot:
            plt.ion()
        
        q, h, phi, phi_d = map_to_arrays(model.w[0], model.map_dict)     

        self.h_y_lim = np.array(h).max()*1.1
        self.u_y_lim = np.array(q).max()*1.1
        self.phi_y_lim = phi.max()*1.10
        
        x = np.linspace(0.0, model.L, 1001)
        self.fig = plt.figure(figsize=(12, 12), dpi=100)
        self.q_plot = self.fig.add_subplot(411)
        self.h_plot = self.fig.add_subplot(412)
        self.phi_plot = self.fig.add_subplot(413)
        self.phi_d_plot = self.fig.add_subplot(414)

        self.title = self.fig.text(0.05,0.935,r'variables at $t={}$'.format(model.t))
        
        self.update_plot(model) 

    def update_plot(self, model):

        q, h, phi, phi_d = map_to_arrays(model.w[0], model.map_dict)  

        x = np.linspace(0.0, model.L, 1001)

        # self.title.set_text(timestep_info_string(model, True))#('variables at t={}'.format(model.t))
        
        self.q_plot.clear()
        self.h_plot.clear()
        self.phi_plot.clear()
        self.phi_d_plot.clear()

        self.phi_d_plot.set_xlabel(r'$x$')
        self.q_plot.set_ylabel(r'$u$')
        self.h_plot.set_ylabel(r'$h$')
        self.phi_plot.set_ylabel(r'$\varphi$')
        self.phi_d_plot.set_ylabel(r'$\eta$')

        u_int = self.y_data(model, q, model.L, x)
        h_int = self.y_data(model, h, model.L, x)
        phi_int = self.y_data(model, phi, model.L, x)
        phi_d_int = self.y_data(model, phi_d, model.L, x)

        self.q_line, = self.q_plot.plot(x, u_int, 'r-')
        self.h_line, = self.h_plot.plot(x, h_int, 'r-')
        self.phi_line, = self.phi_plot.plot(x, phi_int, 'r-')
        self.phi_d_line, = self.phi_d_plot.plot(x, phi_d_int, 'r-')

        if self.rescale:
            self.h_y_lim = [h_int.min(), h_int.max()*1.1]
            self.u_y_lim = [u_int.min(), u_int.max()*1.1]
            self.phi_y_lim = [phi_int.min(), phi_int.max()*1.10]
            self.phi_d_y_lim = [phi_d_int.min(), max(phi_d_int.max()*1.10, 1e-10)]

            x_lim = model.L

            self.q_plot.set_xlim([0.0,x_lim])
            self.h_plot.set_xlim([0.0,x_lim])
            self.phi_plot.set_xlim([0.0,x_lim])
            self.phi_d_plot.set_xlim([0.0,x_lim])

            self.q_plot.set_autoscaley_on(False)
            self.h_plot.set_autoscaley_on(False)
            self.phi_plot.set_autoscaley_on(False)
            self.phi_d_plot.set_autoscaley_on(False)

            self.q_plot.set_ylim(self.u_y_lim)
            self.h_plot.set_ylim(self.h_y_lim)
            self.phi_plot.set_ylim(self.phi_y_lim)
            self.phi_d_plot.set_ylim(self.phi_d_y_lim)
        
        if model.show_plot:
            self.fig.canvas.draw()
        if model.save_plot:
            self.fig.savefig(self.save_loc + '_{:06.3f}.png'.format(model.t))  

    def y_data(self, model, u, x_n, x, norm = None):

        val_x = np.linspace(0.0, x_n, model.L/model.dX + 1)

        # PnDG
        if len(u) > len(val_x):
            v = val_x
            val_x = [v[0]]
            for val_x_ in v[1:-1]:
                val_x.append(val_x_)
                val_x.append(val_x_)
            val_x.append(v[-1])

        # P0DG
        if len(u) < len(val_x):
            v = val_x
            val_x = []
            for i in range(len(v[:-1])):
                val_x.append(v[i:i+1].mean())

        u_int = np.interp(x, val_x, u)
        
        if norm != None:
            norm_int = self.y_data(model, norm, x_n, x)
            u_int = u_int/norm_int

        return u_int

    def clean_up(self):
        plt.close()

def generate_dof_map(model):
    
    # get dof_maps
    model.map_dict = dict()

    for i in range(4):
        if model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()) + 1:   # P1CG 
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j)[0] 
                                 for j in range(len(model.mesh.cells()))]
            model.map_dict[i].append(model.W.sub(i).dofmap().cell_dofs(len(model.mesh.cells()) - 1)[-1])   
        elif model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()):   # P0DG
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j) 
                                 for j in range(len(model.mesh.cells()))]
            model.map_dict[i] = list(np.array(model.map_dict[i]).flatten())
        elif model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()) * 2:   # P1DG
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j) 
                                 for j in range(len(model.mesh.cells()))]
            model.map_dict[i] = list(np.array(model.map_dict[i]).flatten()) 
        else:   # R
            model.map_dict[i] = model.W.sub(i).dofmap().cell_dofs(0)  

def map_to_arrays(w, map):
    
    q = np.array([w.vector().array()[i] for i in map[0]])
    h = np.array([w.vector().array()[i] for i in map[1]])
    phi = np.array([w.vector().array()[i] for i in map[2]])
    phi_d = np.array([w.vector().array()[i] for i in map[3]])
    
    return q, h, phi, phi_d
