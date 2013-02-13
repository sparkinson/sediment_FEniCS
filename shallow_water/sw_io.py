import matplotlib.pyplot as plt
# plt.ion()
from dolfin import *
import numpy as np
import json

class Plotter():

    def __init__(self, model):
        
        q, h, phi, c_d, x_N, u_N = map_to_arrays(model)        

        if len(q) > len(h):
            indices = np.array([[i, i+2] for i in range(0, len(q), 2)]).flatten()
            q_p1 = [q[i] for i in range(0, len(q), 2)]
            q = q_p1

        self.h_y_lim = h.max()*1.1
        self.u_y_lim = (q/h).max()*1.1
        self.phi_y_lim = (phi/(model.rho_R_*model.g_*h)).max()*1.10
        
        x = np.linspace(0.0, x_N[0], 10001)
        self.fig = plt.figure(figsize=(16, 12), dpi=50)
        self.vel_plot = self.fig.add_subplot(411)
        self.h_plot = self.fig.add_subplot(412)
        self.c_plot = self.fig.add_subplot(413)
        self.c_d_plot = self.fig.add_subplot(414)
        
        self.update_plot(model) 

    def update_plot(self, model):

        q, h, phi, c_d, x_N, u_N = map_to_arrays(model)   

        if len(q) > len(h):
            indices = np.array([[i, i+2] for i in range(0, len(q), 2)]).flatten()
            q_p1 = [q[i] for i in range(0, len(q), 2)]
            q = q_p1

        x = np.linspace(0.0, x_N[0], 10001)
        
        self.vel_plot.clear()
        self.h_plot.clear()
        self.c_plot.clear()
        self.c_d_plot.clear()

        self.vel_line, = self.vel_plot.plot(x, self.y_data(model, q/h, x_N[0], x), 'r-')
        self.h_line, = self.h_plot.plot(x, self.y_data(model, h, x_N[0], x), 'r-')
        self.c_line, = self.c_plot.plot(x, self.y_data(model, phi/(model.rho_R_*model.g_*h), x_N[0], x), 'r-')
        self.c_d_line, = self.c_d_plot.plot(x, self.y_data(model, c_d, x_N[0], x), 'r-')

        c_d_y_lim = max(c_d.max()*1.10, 1e-10)
        x_lim = x_N[0]
        self.vel_plot.set_autoscaley_on(False)
        self.vel_plot.set_xlim([0.0,x_lim])
        self.vel_plot.set_ylim([-self.u_y_lim/5.0,self.u_y_lim])
        self.h_plot.set_autoscaley_on(False)
        self.h_plot.set_xlim([0.0,x_lim])
        self.h_plot.set_ylim([0.0,self.h_y_lim])
        self.c_plot.set_autoscaley_on(False)
        self.c_plot.set_xlim([0.0,x_lim])
        self.c_plot.set_ylim([0.0,self.phi_y_lim])
        self.c_d_plot.set_autoscaley_on(False)
        self.c_d_plot.set_xlim([0.0,x_lim])
        self.c_d_plot.set_ylim([0.0,c_d_y_lim])

        # self.fig.canvas.draw()
        self.fig.savefig('results/{:06.3f}.png'.format(model.t))  

    def y_data(self, model, u, x_n, x):

        val_x = np.linspace(0.0, x_n, model.L_/model.dX_ + 1)

        if len(u) > len(val_x):
            v = val_x
            val_x = [v[0]]
            for x in v[1:-1]:
                val_x.append(x)
                val_x.append(x)
            val_x.append(v[-1])
        
        return np.interp(x, val_x, u) 

def print_timestep_info(model, delta):
    
    q, h, phi, c_d, x_N, u_N = map_to_arrays(model)
        
    mass = (h[:model.L_/model.dX_ + 1]*(x_N[0]*model.dX_)).sum()

    info_green("\nEND OF TIMESTEP t = {0:.2e}, dt = {1:.2e}:\n".format(model.t, model.timestep) +
               "x_N = {0:.2e}, u_N = {1:.2e}, u_N_2 = {2:.2e}, h_N = {3:.2e}, mass = {4:.2e}, dw = {5:.2e}\n"
               .format(x_N[0], u_N[0], q[-1]/h[-1], h[-1], mass, delta))

def map_to_arrays(model):
    
    q = np.array([model.w[0].vector().array()[i] for i in model.map_dict[0]])
    h = np.array([model.w[0].vector().array()[i] for i in model.map_dict[1]])
    phi = np.array([model.w[0].vector().array()[i] for i in model.map_dict[2]])
    c_d = np.array([model.w[0].vector().array()[i] for i in model.map_dict[3]])
    x_N = np.array([model.w[0].vector().array()[i] for i in model.map_dict[4]])
    u_N = np.array([model.w[0].vector().array()[i] for i in model.map_dict[5]])
    
    return q, h, phi, c_d, x_N, u_N

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
