#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
import mms_strings as mms
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

############################################################
# DOLFIN SETTINGS

info(parameters, False)
set_log_active(False)

############################################################
# TIME DISCRETISATION FUNCTIONS

def crank_nicholson(self, u):
    return 0.5*u[0] + 0.5*u[1]
def backward_euler(self, u): #implicit
    return u[0]
def forward_euler(self, u): #explicit
    return u[1]

class Model():
    
    # SIMULATION USER DEFINED PARAMETERS
    # function spaces
    shape = 1

    # mesh
    dX_ = 5e-2
    L = 1.0

    # stabilisation
    b_ = 0.3

    # current properties
    # g_prime_ = 0.81 # 9.81*0.0077
    c_0 = 0.00349
    rho_R_ = 1.717
    h_0 = 0.4
    x_N_ = 0.2
    Fr_ = 1.19
    g_ = 9.81
    u_sink_ = 1e-3

    # time step
    timestep = 5.0e-2 #1./1000.

    # define time discretisation
    td = crank_nicholson

    # mms test (default False)
    mms = False

    def setup(self):
        # define constants
        self.dX = Constant(self.dX_)
        self.g = Constant(self.g_)
        self.rho_R = Constant(self.rho_R_)
        self.b = Constant(self.b_)
        self.Fr = Constant(self.Fr_)
        self.u_sink = Constant(self.u_sink_)
        
        Q = self.initialise_functions_spaces(mesh = mesh, 
                                             h_exp = str(self.h_0), 
                                             phi_exp = str(self.c_0*self.rho_R_*self.g_*self.h_0), 
                                             c_d_exp = '0.0', 
                                             q_exp = '0.0', 
                                             x_N_exp = str(self.x_N_), 
                                             u_N_exp = '0.0')

        # define bc's
        self.bcq = [DirichletBC(Q, '0.0', "near(x[0], 0.0) && on_boundary")]
        self.bcc_d = [DirichletBC(Q, '0.0', "near(x[0], 1.0) && on_boundary")]

        # initialise plot
        self.initialise_plot()   

    def mms_setup(self, dX_):
        mms = True

        # define constants
        self.dX = Constant(dX_)
        self.L = Constant(np.pi)
        self.g = Constant(1.0)
        self.rho_R = Constant(1.0)
        self.b = Constant(1.0)
        self.Fr = Constant(1.0)
        self.u_sink = Constant(1.0)

        Q = self.initialise_functions_spaces(mesh = mesh, 
                                             h_exp = str(self.h_0), 
                                             phi_exp = str(self.c_0*self.rho_R_*self.g_*self.h_0), 
                                             c_d_exp = '0.0', 
                                             q_exp = '0.0', 
                                             x_N_exp = str(self.x_N_), 
                                             u_N_exp = '0.0')

        # define bc's
        self.bcq = [DirichletBC(Q, '0.0', "near(x[0], 0.0) && on_boundary")]
        self.bcc_d = [DirichletBC(Q, '0.0', "near(x[0], 1.0) && on_boundary")]

        # initialise plot
        self.initialise_plot() 

    def initialise_functions_spaces(self, mesh, h_exp, phi_exp, c_d_exp, q_exp, x_N_exp, u_N_exp):
        # define geometry
        mesh = Interval(int(self.L/self.dX_), 0.0, self.L)
        self.n = FacetNormal(mesh)

        # define function spaces
        Q = FunctionSpace(mesh, "CG", self.shape)
        R = FunctionSpace(mesh, "R", 0)
        
        # define test functions
        self.v = TestFunction(Q)
        self.r = TestFunction(R)

        # define function dictionaries for prognostic variables
        self.h = dict([[i, interpolate(Expression(h_exp), Q)] for i in range(2)])
        self.phi = dict([[i, interpolate(Expression(phi_exp), Q)] for i in range(2)])
        self.c_d = dict([[i, interpolate(Expression(c_d_exp), Q)] for i in range(2)])
        self.q = dict([[i, interpolate(Expression(q_exp), Q)] for i in range(2)])
        self.x_N = dict([[i, interpolate(Expression(x_N_exp), R)] for i in range(2)])
        self.u_N = dict([[i, interpolate(Expression(u_N_exp), R)] for i in range(2)])
        self.X = interpolate(Expression('x[0]'), Q) 

        # left boundary marked as 0, right as 1
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        left_boundary = LeftBoundary()
        exterior_facet_domains = FacetFunction("uint", mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains]  

        return Q       

    def y_data(self, u):
        val_x = np.linspace(0.0, self.x_N[0].vector().array()[-1], self.L/self.dX_ + 1)
        val = u[:self.L/self.dX_ + 1]
        return np.interp(self.plot_x, val_x, val, right=0.0)

    def solve(self, T):
        self.t = 0.0
        while (self.t < T):
            self.t += self.timestep
            k = Constant(self.timestep)

            ss = 1.0
            nl_its = 0
            while (nl_its < 2 or ss > 1e-4):

                # VALUES FOR CONVERGENCE TEST
                h_nl = self.h[0].copy(deepcopy=True)
                phi_nl = self.phi[0].copy(deepcopy=True)
                q_nl = self.q[0].copy(deepcopy=True)
                x_N_nl = self.x_N[0].copy(deepcopy=True)

                # DEFINE EQUATIONS
                # time discretisation of values
                x_N_td = self.td(self.x_N)
                inv_x_N = 1./x_N_td
                u_N_td = self.td(self.u_N)
                h_td = self.td(self.h)
                phi_td = self.td(self.phi)
                c_d_td = self.td(self.c_d)
                q_td = self.td(self.q)

                # momentum
                F_q = self.v*(self.q[0] - self.q[1])*dx + \
                    inv_x_N*grad(self.v*self.X)*u_N_td*q_td*k*dx + \
                    inv_x_N*self.v*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)*k*dx - \
                    inv_x_N*self.v*self.X*u_N_td**2.0*h_td*self.n*k*self.ds(1) + \
                    self.v*self.s_q*k*dx
                # momentum stabilisation
                u = q_td/h_td
                alpha = self.b*self.dX*(abs(u)+u+h_td**0.5)*h_td
                F_q = F_q + inv_x_N*grad(self.v)*alpha*grad(u)*k*dx

                # conservation
                F_h = self.v*(self.h[0] - self.h[1])*dx + \
                    inv_x_N*self.v*grad(q_td)*k*dx - \
                    inv_x_N*self.v*self.X*u_N_td*grad(h_td)*k*dx + \
                    self.v*self.s_h*k*dx

                # deposition
                F_phi = self.v*(self.phi[0] - self.phi[1])*dx + \
                    inv_x_N*self.v*grad(q_td*phi_td/h_td)*k*dx - \
                    inv_x_N*self.v*self.X*u_N_td*grad(phi_td)*k*dx + \
                    self.v*self.u_sink*phi_td/h_td*k*dx + \
                    self.v*self.s_phi*k*dx

                # deposit
                F_c_d = self.v*(self.c_d[0] - self.c_d[1])*dx - \
                    inv_x_N*self.v*self.X*u_N_td*grad(c_d_td)*k*dx - \
                    self.v*self.u_sink*phi_td/(self.rho_R*self.g*h_td)*k*dx

                # nose location/speed
                F_u_N = self.r*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
                    self.r*self.u_N[0]*self.ds(1) + self.r*self.s_u_N*self.ds(1)
                F_x_N = self.r*(self.x_N[0] - self.x_N[1])*dx - self.r*u_N_td*k*dx 

                # SOLVE EQUATIONS
                solve(F_q == 0, self.q[0], self.bcq)
                solve(F_h == 0, self.h[0])
                solve(F_phi == 0, self.phi[0])
                solve(F_c_d == 0, self.c_d[0], self.bcc_d)
                solve(F_u_N == 0, self.u_N[0])
                solve(F_x_N == 0, self.x_N[0])

                dh = errornorm(h_nl, self.h[0], norm_type="L2", degree=self.shape + 1)
                dphi = errornorm(phi_nl, self.phi[0], norm_type="L2", degree=self.shape + 1)
                dq = errornorm(q_nl, self.q[0], norm_type="L2", degree=self.shape + 1)
                dx_N = errornorm(x_N_nl, self.x_N[0], norm_type="L2", degree=self.shape + 1)
                ss = max(dh, dphi, dq, dx_N)

                nl_its += 1

            self.h[1].assign(self.h[0])
            self.phi[1].assign(self.phi[0])
            self.c_d[1].assign(self.c_d[0])
            self.q[1].assign(self.q[0])
            self.x_N[1].assign(self.x_N[0])
            self.u_N[1].assign(self.u_N[0])

            # display results
            self.update_plot()
            self.print_timestep_info(nl_its)

    def initialise_plot(self):
        # fig settings
        self.plot_x = np.linspace(0.0, 2.0, 10001)
        self.fig = plt.figure(figsize=(16, 12), dpi=50)
        self.vel_plot = self.fig.add_subplot(311)
        self. h_plot = self.fig.add_subplot(312)
        # self.c_plot = fig.add_subplot(413)
        self.c_d_plot = self.fig.add_subplot(313)
        self.plot_freq = 100000.0

        # axis settings
        self.h_plot.set_autoscaley_on(False)
        self.h_plot.set_ylim([0.0,0.5])
        self.vel_plot.set_autoscaley_on(False)
        self.vel_plot.set_ylim([0.0,0.2])
        # self.c_plot.set_autoscaley_on(False)
        # self.c_plot.set_ylim([0.0,0.005])
        self.c_d_plot.set_autoscaley_on(False)
        self.c_d_plot.set_ylim([0.0,1.2e-4])

        # set initial plot values
        self.vel_line, = self.vel_plot.plot(self.plot_x, 
                                            self.y_data(self.q[0].vector().array()/self.h[0].vector().array()), 'r-')
        self.h_line, = self.h_plot.plot(self.plot_x, 
                                        self.y_data(self.h[0].vector().array()), 'r-')
        # self.c_line, = self.c_plot.plot(self.plot_x, 
        #                                 self.y_data(self.phi[0].vector().array()/
        #                                                      (self.rho_R_*self.g_*self.h[0].vector().array())
        #                                                      ), 'r-')
        self.c_d_line, = self.c_d_plot.plot(self.plot_x, 
                                            self.y_data(self.c_d[0].vector().array()), 'r-')
        self.fig.canvas.draw()
        self.fig.savefig('results/%06.2f.png' % (0.0))       

    def update_plot(self):
        self.vel_line.set_ydata(self.y_data(self.q[0].vector().array()/self.h[0].vector().array()))
        self.h_line.set_ydata(self.y_data(self.h[0].vector().array()))
        # self.c_line.set_ydata(self.y_data(self.phi[0].vector().array()/(self.rho_R_*self.g_*self.h[0].vector().array())))
        self.c_d_line.set_ydata(self.y_data(self.c_d[0].vector().array()))
        self.fig.canvas.draw()
        self.fig.savefig('results/{:06.2f}.png'.format(self.t))   
        
    def print_timestep_info(self, nl_its):
        mass = (self.h[0].vector().array()[:self.L/self.dX_ + 1]*(self.x_N[0].vector().array()[-1]*self.dX_)).sum()
        info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, u_N_2 = %.2e, h_N = %.2e, nl_its = %1d, mass = %.2e" % 
                   (self.t, self.timestep, 
                    self.x_N[0].vector().array()[-1], 
                    self.u_N[0].vector().array()[0], 
                    self.q[0].vector().array()[-1]/self.h[0].vector().array()[-1], 
                    self.h[0].vector().array()[-1],
                    nl_its, 
                    mass))
        # info_red(self.c_d[0].vector().array().max())

if __name__ == '__main__':
    
    model = Model()
    model.setup()
    model.solve(5.0)

    # model = Model(mms = True)
