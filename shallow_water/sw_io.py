import matplotlib.pyplot as plt
plt.ion()
from dolfin import *
import numpy as np
import json

class Plotter():
    
    def __init__(self, model):

        if model.mms:
            x_max = np.pi
            h_y_lim = 20.0
            u_y_lim = 1.5
            phi_y_lim = 0.2
            c_d_y_lim = 5.0
        else:
            x_max = 1.0
            h_y_lim = 0.5
            u_y_lim = 0.5
            phi_y_lim = 0.01
            c_d_y_lim = 1.0e-5
        
        self.x = np.linspace(0.0, x_max, 10001)
        self.fig = plt.figure(figsize=(16, 12), dpi=50)
        self.vel_plot = self.fig.add_subplot(411)
        self.h_plot = self.fig.add_subplot(412)
        self.c_plot = self.fig.add_subplot(413)
        self.c_d_plot = self.fig.add_subplot(414)
        self.plot_freq = 100000.0

         # axis settings
        self.h_plot.set_autoscaley_on(False)
        self.h_plot.set_ylim([0.0,h_y_lim])
        self.vel_plot.set_autoscaley_on(False)
        self.vel_plot.set_ylim([-u_y_lim/5.0,u_y_lim])
        self.c_plot.set_autoscaley_on(False)
        self.c_plot.set_ylim([0.0,phi_y_lim])
        self.c_d_plot.set_autoscaley_on(False)
        self.c_d_plot.set_ylim([0.0,c_d_y_lim])
        
        q, h, phi, c_d, x_N, u_N = map_to_arrays(model)
        
        self.vel_line, = self.vel_plot.plot(self.x, self.y_data(model, q/h), 'r-')
        self.h_line, = self.h_plot.plot(self.x, self.y_data(model, h), 'r-')
        self.c_line, = self.c_plot.plot(self.x, self.y_data(model, phi/(model.rho_R_*model.g_*h)), 'r-')
        self.c_d_line, = self.c_d_plot.plot(self.x, self.y_data(model, c_d), 'r-')

        self.fig.canvas.draw()
        self.fig.savefig('results/%06.2f.png' % (0.0))  

    def update_plot(self, model):
        
        q, h, phi, c_d, x_N, u_N = map_to_arrays(model)
        
        self.vel_line.set_ydata(self.y_data(model, q/h))
        self.h_line.set_ydata(self.y_data(model, h))
        self.c_line.set_ydata(self.y_data(model, phi/(model.rho_R_*model.g_*h)))
        self.c_d_line.set_ydata(self.y_data(model, c_d))

        self.fig.canvas.draw()
        self.fig.savefig('results/{:06.2f}.png'.format(model.t))  

    def y_data(self, model, u):

        x_N = model.w[0].vector().array()[model.W.sub(5).dofmap().cell_dofs(0)]

        val_x = np.linspace(0.0, x_N, model.L/model.dX_ + 1)
        val = u[:model.L/model.dX_ + 1]
        
        return np.interp(self.x, val_x, val, right=0.0) 

def print_timestep_info(model, delta):
    
    q, h, phi, c_d, x_N, u_N = map_to_arrays(model)
        
    mass = (h[:model.L/model.dX_ + 1]*(x_N[0]*model.dX_)).sum()

    info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, u_N_2 = %.2e, h_N = %.2e, mass = %.2e, dw = %.2e" % 
               (model.t, model.timestep, 
                x_N[0], 
                u_N[0], 
                q[-1]/h[-1], 
                h[-1],
                mass,
                delta))

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

def clear_file(fname):
    f = open(fname, 'w')
    f.close()
