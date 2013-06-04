import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import json

class Plotter():

    def __init__(self, model, rescale, file):

        self.rescale = rescale
        self.save_loc = file

        if model.show_plot:
            plt.ion()
        
        q, h, phi, phi_d, x_N, u_N = map_to_arrays(model)     

        self.h_y_lim = np.array(h).max()*1.1
        self.u_y_lim = np.array(q).max()*1.1
        self.phi_y_lim = phi.max()*1.10
        
        x = np.linspace(0.0, x_N[0], 10001)
        self.fig = plt.figure(figsize=(16, 12), dpi=50)
        self.q_plot = self.fig.add_subplot(411)
        self.h_plot = self.fig.add_subplot(412)
        self.phi_plot = self.fig.add_subplot(413)
        self.phi_d_plot = self.fig.add_subplot(414)
        
        self.update_plot(model) 

    def update_plot(self, model):

        q, h, phi, phi_d, x_N, u_N = map_to_arrays(model)   

        x = np.linspace(0.0, x_N[0], 10001)
        
        self.q_plot.clear()
        self.h_plot.clear()
        self.phi_plot.clear()
        self.phi_d_plot.clear()

        u_int = self.y_data(model, q, x_N[0], x, h)
        h_int = self.y_data(model, h, x_N[0], x)
        phi_int = self.y_data(model, phi, x_N[0], x, h)
        phi_d_int = self.y_data(model, phi_d, x_N[0], x)

        self.q_line, = self.q_plot.plot(x, u_int, 'r-')
        self.h_line, = self.h_plot.plot(x, h_int, 'r-')
        self.phi_line, = self.phi_plot.plot(x, phi_int, 'r-')
        self.phi_d_line, = self.phi_d_plot.plot(x, phi_d_int, 'r-')

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

        if len(u) > len(val_x):
            v = val_x
            val_x = [v[0]]
            for val_x_ in v[1:-1]:
                val_x.append(val_x_)
                val_x.append(val_x_)
            val_x.append(v[-1])

        u_int = np.interp(x, val_x, u)
        
        if norm != None:
            norm_int = self.y_data(model, norm, x_n, x)
            u_int = u_int/norm_int

        return u_int

    def clean_up(self):
        plt.close()

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

    q, h, phi, phi_d, x_N, u_N = map_to_arrays(model) 

    write_array_to_file(file + '_q.json', q, method)
    write_array_to_file(file + '_h.json', h, method)
    write_array_to_file(file + '_phi.json', phi, method)
    write_array_to_file(file + '_phi_d.json', phi_d, method)
    write_array_to_file(file + '_x_N.json', x_N, method)
    write_array_to_file(file + '_u_N.json', u_N, method)
    write_array_to_file(file + '_T.json', [model.t], method)

def print_timestep_info(model, delta):
    
    q, h, phi, phi_d, x_N, u_N = map_to_arrays(model)
        
    mass = (h[:model.L_/model.dX_ + 1]*(x_N[0]*model.dX_)).sum()

    info_green("\nEND OF TIMESTEP t = {0:.2e}, dt = {1:.2e}:\n".format(model.t, model.timestep) +
               "x_N = {0:.2e}, u_N = {1:.2e}, u_N_2 = {2:.2e}, h_N = {3:.2e}, mass = {4:.2e}, dw = {5:.2e}\n"
               .format(x_N[0], u_N[0], q[-1]/h[-1], h[-1], mass, delta))

def map_to_arrays(model):
    
    q = np.array([model.w[0].vector().array()[i] for i in model.map_dict[0]])
    h = np.array([model.w[0].vector().array()[i] for i in model.map_dict[1]])
    phi = np.array([model.w[0].vector().array()[i] for i in model.map_dict[2]])
    phi_d = np.array([model.w[0].vector().array()[i] for i in model.map_dict[3]])
    x_N = np.array([model.w[0].vector().array()[i] for i in model.map_dict[4]])
    u_N = np.array([model.w[0].vector().array()[i] for i in model.map_dict[5]])

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
