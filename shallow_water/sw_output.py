import matplotlib.pyplot as plt
plt.ion()
from dolfin import *
import numpy as np

class Plotter():
    
    def __init__(self, model):

        if model.mms:
            x_max = np.pi
            h_y_lim = 20.0
            u_y_lim = 1.5
            phi_y_lim = 0.2
            c_d_y_lim = 5.0
        else:
            x_max = 2.0
            h_y_lim = 0.5
            u_y_lim = 0.4
            phi_y_lim = 0.01
            c_d_y_lim = 1.2e-5
        
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

        q_0_project = project(model.q[0], model.P1CG)
        q_1_project = project(model.q[1], model.P1CG)
        self.vel_line, = self.vel_plot.plot(self.x, 
                                            self.y_data(model, q_0_project.vector().array()/model.h[0].vector().array()), 'r-')
        self.h_line, = self.h_plot.plot(self.x, 
                                        self.y_data(model, model.h[0].vector().array()), 'r-')
        self.c_line, = self.c_plot.plot(self.x, 
                                        self.y_data(model, model.phi[0].vector().array()/
                                                             (model.rho_R_*model.g_*model.h[0].vector().array())
                                                             ), 'r-')
        self.c_d_line, = self.c_d_plot.plot(self.x, 
                                            self.y_data(model, model.c_d[0].vector().array()), 'r-')

        if model.mms:
            self.c_d_line_2, = self.c_d_plot.plot(self.x, 
                                                  self.y_data(model, model.c_d[1].vector().array()), 'b-')
            self.h_line_2, = self.h_plot.plot(self.x, 
                                              self.y_data(model, model.h[1].vector().array()), 'b-')
            self.c_line_2, = self.c_plot.plot(self.x, 
                                              self.y_data(model, model.phi[1].vector().array()/
                                                          (model.rho_R_*model.g_*model.h[0].vector().array())
                                                          ), 'b-')
            self.vel_line_2, = self.vel_plot.plot(self.x, 
                                                  self.y_data(model, q_1_project.vector().array()/model.h[0].vector().array()), 'b-')


        self.fig.canvas.draw()
        self.fig.savefig('results/%06.2f.png' % (0.0))  

    def update_plot(self, model):
        q_0_project = project(model.q[0], model.P1CG)
        self.vel_line.set_ydata(self.y_data(model, q_0_project.vector().array()/model.h[0].vector().array()))
        self.h_line.set_ydata(self.y_data(model, model.h[0].vector().array()))
        self.c_line.set_ydata(self.y_data(model, model.phi[0].vector().array()/(model.rho_R_*model.g_*model.h[0].vector().array())))
        self.c_d_line.set_ydata(self.y_data(model, model.c_d[0].vector().array()))
        self.fig.canvas.draw()
        self.fig.savefig('results/{:06.2f}.png'.format(model.t))  

    def y_data(self, model, u):
        val_x = np.linspace(0.0, model.x_N[0].vector().array()[-1], model.L/model.dX_ + 1)
        val = u[:model.L/model.dX_ + 1]
        return np.interp(self.x, val_x, val, right=0.0) 

def print_timestep_info(model, nl_its, du):
    mass = (model.h[0].vector().array()[:model.L/model.dX_ + 1]*(model.x_N[0].vector().array()[-1]*model.dX_)).sum()
    info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, u_N_2 = %.2e, h_N = %.2e, nl_its = %1d, mass = %.2e, du = %.2e" % 
               (model.t, model.timestep, 
                model.x_N[0].vector().array()[-1], 
                model.u_N[0].vector().array()[0], 
                model.q[0].vector().array()[-1]/model.h[0].vector().array()[-1], 
                model.h[0].vector().array()[-1],
                nl_its, 
                mass,
                du))
