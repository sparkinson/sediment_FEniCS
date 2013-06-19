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

def similarity_u(model, y):
    K = (27*model.Fr_**2.0/(12-2*model.Fr_**2.0))**(1./3.)
    return (2./3.)*K*model.t**(-1./3.)*y

def similarity_h(model, y):
    K = (27*model.Fr_**2.0/(12-2*model.Fr_**2.0))**(1./3.)
    H0 = 1./model.Fr_**2.0 - 0.25 + 0.25*y**2.0
    return (4./9.)*K**2.0*model.t**(-2./3.)*H0

class Plotter():

    def __init__(self, model, rescale, file, similarity = False):

        self.rescale = rescale
        self.save_loc = file
        self.similarity = similarity

        if model.show_plot:
            plt.ion()
        
        q, h, phi, phi_d, x_N, u_N = map_to_arrays(model.w[0], model.map_dict)     

        self.h_y_lim = np.array(h).max()*1.1
        self.u_y_lim = np.array(q).max()*1.1
        self.phi_y_lim = phi.max()*1.10
        
        x = np.linspace(0.0, x_N[0], 10001)
        self.fig = plt.figure(figsize=(12, 12), dpi=100)
        self.q_plot = self.fig.add_subplot(411)
        self.h_plot = self.fig.add_subplot(412)
        self.phi_plot = self.fig.add_subplot(413)
        self.phi_d_plot = self.fig.add_subplot(414)

        self.title = self.fig.text(0.05,0.935,r'variables at $t={}$'.format(model.t))
        
        self.update_plot(model) 

    def update_plot(self, model):

        q, h, phi, phi_d, x_N, u_N = map_to_arrays(model.w[0], model.map_dict)  

        x = np.linspace(0.0, x_N[0], 10001)

        self.title.set_text(timestep_info_string(model, True))#('variables at t={}'.format(model.t))
        
        self.q_plot.clear()
        self.h_plot.clear()
        self.phi_plot.clear()
        self.phi_d_plot.clear()

        self.phi_d_plot.set_xlabel(r'$x$')
        self.q_plot.set_ylabel(r'$u$')
        self.h_plot.set_ylabel(r'$h$')
        self.phi_plot.set_ylabel(r'$\varphi$')
        self.phi_d_plot.set_ylabel(r'$\eta$')

        u_int = self.y_data(model, q, x_N[0], x, h)
        h_int = self.y_data(model, h, x_N[0], x)
        phi_int = self.y_data(model, phi, x_N[0], x, h)
        phi_d_int = self.y_data(model, phi_d, x_N[0], x)

        self.q_line, = self.q_plot.plot(x, u_int, 'r-')
        self.h_line, = self.h_plot.plot(x, h_int, 'r-')
        self.phi_line, = self.phi_plot.plot(x, phi_int, 'r-')
        self.phi_d_line, = self.phi_d_plot.plot(x, phi_d_int, 'r-')

        if self.similarity:
            similarity_x = np.linspace(0.0,(27*model.Fr_**2.0/(12-2*model.Fr_**2.0))**(1./3.)*model.t**(2./3.),1001)
            self.q_line_2, = self.q_plot.plot(similarity_x, [similarity_u(model,y) for y in np.linspace(0.0,1.0,1001)], 'k--')
            self.h_line_2, = self.h_plot.plot(similarity_x, [similarity_h(model,y) for y in np.linspace(0.0,1.0,1001)], 'k--')
            self.phi_line_2, = self.phi_plot.plot(similarity_x, np.ones([1001]), 'k--')
            self.phi_d_line_2, = self.phi_d_plot.plot(x, phi_d_int, 'k--')

        if self.rescale:
            self.h_y_lim = h_int.max()*1.1
            self.u_y_lim = u_int.max()*1.1
            self.phi_y_lim = phi_int.max()*1.10

        phi_d_y_lim = max(phi_d_int.max()*1.10, 1e-10)
        x_lim = x_N[0]
        self.q_plot.set_autoscaley_on(False)
        self.q_plot.set_xlim([0.0,x_lim])
        self.q_plot.set_ylim([0.0,self.u_y_lim])
        self.h_plot.set_autoscaley_on(False)
        self.h_plot.set_xlim([0.0,x_lim])
        self.h_plot.set_ylim([0.0,self.h_y_lim])
        self.phi_plot.set_autoscaley_on(False)
        self.phi_plot.set_xlim([0.0,x_lim])
        self.phi_plot.set_ylim([0.0,self.phi_y_lim])
        self.phi_d_plot.set_autoscaley_on(False)
        self.phi_d_plot.set_xlim([0.0,x_lim])
        self.phi_d_plot.set_ylim([0.0,phi_d_y_lim])
        
        if model.show_plot:
            self.fig.canvas.draw()
        if model.save_plot:
            self.fig.savefig(self.save_loc + '_{:06.3f}.png'.format(model.t))  

    def y_data(self, model, u, x_n, x, norm = None):

        val_x = np.linspace(0.0, x_n, model.L_/model.dX_ + 1)

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

class Adjoint_Plotter():

    def __init__(self, file, show, target):

        self.save_loc = file
        self.show = show
        self.target = target

        if self.show:
            plt.ion()

        if self.target:
            f = open('phi_ic.json','r')
            self.target_phi = np.array(json.loads(f.readline()))
            f = open('deposit_data.json','r')
            self.target_phi_d = np.array(json.loads(f.readline()))
        
        self.j = []
        
        self.fig = plt.figure(figsize=(6, 4), dpi=100)
        self.phi_plot = self.fig.add_subplot(131)
        self.phi_d_plot = self.fig.add_subplot(132)
        self.j_plot = self.fig.add_subplot(133)

    def update_plot(self, phi_ic, phi_d, j):  
        
        self.phi_plot.clear()
        self.phi_d_plot.clear()
        self.j_plot.clear()

        self.phi_plot.set_xlabel(r'$x$')
        self.phi_plot.set_ylabel(r'$\varphi$ (START)')
        self.phi_d_plot.set_xlabel(r'$x$')
        self.phi_d_plot.set_ylabel(r'$\eta$ (END)')
        self.j_plot.set_xlabel(r'iterations')
        self.j_plot.set_ylabel(r'$J$')

        if self.target:
            self.target_phi_line, = self.phi_plot.plot(np.linspace(0,1.0,len(self.target_phi)), self.target_phi, 'r-')
            self.target_phi_d_line, = self.phi_d_plot.plot(np.linspace(0,1.0,len(self.target_phi_d)), self.target_phi_d, 'r-')

        self.phi_line, = self.phi_plot.plot(np.linspace(0,1.0,len(phi_ic)), phi_ic, 'b-')
        self.phi_d_line, = self.phi_d_plot.plot(np.linspace(0,1.0,len(phi_d)), phi_d, 'b-')

        self.j.append(j)
        if all(e > 0.0 for e in self.j):
            self.j_plot.set_yscale('log')
        self.j_line, = self.j_plot.plot(self.j, 'r-')

        self.phi_plot.set_autoscaley_on(True)
        self.phi_d_plot.set_autoscaley_on(True)
        self.j_plot.set_autoscaley_on(True)
        
        if self.show:
            self.fig.canvas.draw()
        # if model.save_plot:
        #     self.fig.savefig(self.save_loc + '_{:06.3f}.png'.format(model.t)) 

    def clean_up(self):
        plt.close()

def generate_dof_map(model):
    
    # get dof_maps
    model.map_dict = dict()

    for i in range(6):
        if model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()) + 1:   # P1CG 
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j)[0] for j in range(len(model.mesh.cells()))]
            model.map_dict[i].append(model.W.sub(i).dofmap().cell_dofs(len(model.mesh.cells()) - 1)[-1])
        elif model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()) * 2 + 1:   # P2CG 
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j)[:-1] for j in range(len(model.mesh.cells()))]
            model.map_dict[i] = list(np.array(model.map_dict[i]).flatten())
            model.map_dict[i].append(model.W.sub(i).dofmap().cell_dofs(len(model.mesh.cells()) - 1)[-1])    
        elif model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()):   # P0DG
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j) for j in range(len(model.mesh.cells()))]
            model.map_dict[i] = list(np.array(model.map_dict[i]).flatten())
        elif model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()) * 2:   # P1DG
            model.map_dict[i] = [model.W.sub(i).dofmap().cell_dofs(j) for j in range(len(model.mesh.cells()))]
            model.map_dict[i] = list(np.array(model.map_dict[i]).flatten())   
        elif model.W.sub(i).dofmap().global_dimension() == len(model.mesh.cells()) * 3:   # P2DG
            dof = model.W.sub(i).dofmap()
            model.map_dict[i] = [[dof.cell_dofs(j)[0], dof.cell_dofs(j)[1]] for j in range(len(model.mesh.cells()))]
            model.map_dict[i] = list(np.array(model.map_dict[i]).flatten())
        else:   # R
            model.map_dict[i] = model.W.sub(i).dofmap().cell_dofs(0)  

def clear_model_files(file):

    files = [file + '_q.json',
             file + '_h.json',
             file + '_phi.json',
             file + '_phi_d.json',
             file + '_x_N.json',
             file + '_u_N.json',
             file + '_T.json']

    for file in files:
        f = open(file, 'w')
        f.write('')
        f.close()

def write_model_to_files(model, method, file):

    q, h, phi, phi_d, x_N, u_N = map_to_arrays(model.w[0], model.map_dict)  

    write_array_to_file(file + '_q.json', q, method)
    write_array_to_file(file + '_h.json', h, method)
    write_array_to_file(file + '_phi.json', phi, method)
    write_array_to_file(file + '_phi_d.json', phi_d, method)
    write_array_to_file(file + '_x_N.json', x_N, method)
    write_array_to_file(file + '_u_N.json', u_N, method)
    write_array_to_file(file + '_T.json', [model.t], method)

def print_timestep_info(model, delta):
    
    info_green("\nEND OF TIMESTEP " + timestep_info_string(model) + "dw = {:.2e}\n".format(delta))

def timestep_info_string(model, tex=False):
    
    q, h, phi, phi_d, x_N, u_N = map_to_arrays(model.w[0], model.map_dict) 
        
    mass = (h[:model.L_/model.dX_ + 1]*(x_N[0]*model.dX_)).sum()

    if tex:
        return ("$t$ = {0:.2e}, $dt$ = {1:.2e}:\n".format(model.t, model.timestep) +
                "$x_N$ = {0:.2e}, $\dot{{x}}_N$ = {1:.2e}, $h_N$ = {3:.2e}, mass = {4:.2e}"
                .format(x_N[0], u_N[0], q[-1]/h[-1], h[-1], mass))
    else:
        return ("t = {0:.2e}, dt = {1:.2e}:\n".format(model.t, model.timestep) +
                "x_N = {0:.2e}, u_N = {1:.2e}, u_N_2 = {2:.2e}, h_N = {3:.2e}, mass = {4:.2e}"
                .format(x_N[0], u_N[0], q[-1]/h[-1], h[-1], mass))

def map_to_arrays(w, map):
    
    q = np.array([w.vector().array()[i] for i in map[0]])
    h = np.array([w.vector().array()[i] for i in map[1]])
    phi = np.array([w.vector().array()[i] for i in map[2]])
    phi_d = np.array([w.vector().array()[i] for i in map[3]])
    x_N = np.array([w.vector().array()[i] for i in map[4]])
    u_N = np.array([w.vector().array()[i] for i in map[5]])

    if len(q) > len(h):
        indices = np.array([[i, i+2] for i in range(0, len(q), 2)]).flatten()
        q_p1 = [q[i] for i in range(0, len(q), 2)]
        q = np.array(q_p1)

    if len(phi) > len(h):
        indices = np.array([[i, i+2] for i in range(0, len(phi), 2)]).flatten()
        phi_p1 = [phi[i] for i in range(0, len(phi), 2)]
        phi = np.array(phi_p1)
    
    return q, h, phi, phi_d, x_N, u_N

def set_model_ic_from_file():
    print 'Not implemented'

def create_function_from_file(fname, fs):
    f = open(fname, 'r')
    data = np.array(json.loads(f.readline()))
    fn = Function(fs)
    fn.vector()[:] = data
    f.close()
    return fn

def write_array_to_file(fname, arr, method):
    f = open(fname, method)
    f.write(json.dumps(list(arr)))
    if method == 'a':
        f.write('\n')
    f.close()

def read_q_vals_from_file(fname):
    f = open(fname, 'r')
    q_a, q_pa, q_pb = json.loads(f.readline())
    f.close()
    return q_a, q_pa, q_pb

def write_q_vals_to_file(fname, q_a, q_pa, q_pb, method):
    f = open(fname, method)
    f.write(json.dumps([q_a, q_pa, q_pb]))
    if method == 'a':
        f.write('\n')
    f.close()

def clear_file(fname):
    f = open(fname, 'w')
    f.close()
