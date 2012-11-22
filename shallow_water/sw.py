#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
from dolfin_adjoint import *
import sw_mms_exp as mms
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from optparse import OptionParser

############################################################
# DOLFIN SETTINGS

parameters['krylov_solver']['relative_tolerance'] = 1e-15
info(parameters, True)
set_log_active(False)

dolfin.parameters["optimization"]["test_gradient"] = True
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.0001

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

    def setup(self, c_0 = None):

        # define constants
        if c_0:
            self.c_0 = c_0
        self.dX = Constant(self.dX_)
        self.g = Constant(self.g_)
        self.rho_R = Constant(self.rho_R_)
        self.b = Constant(self.b_)
        self.Fr = Constant(self.Fr_)
        self.u_sink = Constant(self.u_sink_)
        
        Q = self.initialise_functions_spaces(h_exp = str(self.h_0), 
                                             phi_exp = str(self.c_0*self.rho_R_*self.g_*self.h_0), 
                                             c_d_exp = '0.0', 
                                             q_exp = '0.0', 
                                             x_N_exp = str(self.x_N_), 
                                             u_N_exp = '0.0')

        # define bc's
        self.bch = []
        self.bcphi = []
        self.bcc_d = [DirichletBC(Q, '0.0', "near(x[0], 1.0) && on_boundary")]
        self.bcq = [DirichletBC(Q, '0.0', "near(x[0], 0.0) && on_boundary")]

        # initialise plot
        self.initialise_plot(2.0)   

    def mms_setup(self, dX_, dT):
        self.mms = True

        # define constants
        self.dX_ = dX_
        self.dX = Constant(self.dX_)
        self.L = np.pi
        self.g_ = 1.0
        self.g = Constant(1.0)
        self.rho_R_ = 1.0
        self.rho_R = Constant(1.0)
        self.b_ = 1.0 / dX_
        self.b = Constant(1.0 / dX_)
        self.Fr_ = 1.0
        self.Fr = Constant(1.0)
        self.u_sink_ = 1.0
        self.u_sink = Constant(1.0)

        Q = self.initialise_functions_spaces(h_exp = mms.h(), 
                                             phi_exp = mms.phi(),
                                             c_d_exp = mms.c_d(), 
                                             q_exp = mms.q(), 
                                             x_N_exp = 'pi', 
                                             u_N_exp = mms.u_N())

        # define bc's
        self.bch = [DirichletBC(Q, Expression(mms.h()), 
                                "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcphi = [DirichletBC(Q, Expression(mms.phi()), 
                                  "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcc_d = [DirichletBC(Q, Expression(mms.c_d()), 
                                  "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcq = [DirichletBC(Q, Expression(mms.q()), "(near(x[0], 0.0)) && on_boundary")]
                                # "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]

        # define source terms
        self.s_q = Expression(mms.s_q())
        self.s_h = Expression(mms.s_h())
        self.s_phi = Expression(mms.s_phi())
        self.s_c_d = Expression(mms.s_c_d())

        # initialise plot
        self.initialise_plot(np.pi, h_y_lim = 20.0, u_y_lim = 1.5, phi_y_lim = 0.2, c_d_y_lim = 5.0) 

        self.timestep = dT

    def initialise_functions_spaces(self, h_exp, phi_exp, c_d_exp, q_exp, x_N_exp, u_N_exp):
        # define geometry
        self.mesh = Interval(int(self.L/self.dX_), 0.0, self.L)
        self.n = FacetNormal(self.mesh)

        # define function spaces
        Q = FunctionSpace(self.mesh, "CG", self.shape)
        G = FunctionSpace(self.mesh, "DG", self.shape - 1)
        R = FunctionSpace(self.mesh, "R", 0)
        
        # define test functions
        self.v = TestFunction(Q)
        self.z = TestFunction(G)
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
        exterior_facet_domains = FacetFunction("uint", self.mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains] 
        
        return Q       

    def solve(self, T = None, tol = None, nl_tol = 1e-5):

        def time_finish(t):
            if T:
                if t >= T:
                    return True
            return False

        def converged(du):
            if tol:
                if du < tol:
                    return True
            return False

        self.t = 0.0
        du = 1e10
        while not (time_finish(self.t) or converged(du)):
            self.t += self.timestep
            k = Constant(self.timestep)

            ss = 1.0
            nl_its = 0
            while (nl_its < 2 or du_nl > nl_tol):

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
                q_N = u_N_td*h_td
                u = q_td/h_td
                alpha = self.b*self.dX*(abs(u)+u+(phi_td*h_td)**0.5)*h_td
                F_q = self.v*(self.q[0] - self.q[1])*dx + \
                    inv_x_N*self.v*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)*k*dx + \
                    inv_x_N*u_N_td*grad(self.v*self.X)*q_td*k*dx - \
                    inv_x_N*u_N_td*self.v*self.X*q_N*self.n*k*self.ds(1) + \
                    inv_x_N*grad(self.v)*alpha*grad(u)*k*dx - \
                    inv_x_N*self.v*alpha*grad(u)*self.n*k*self.ds(1) 
                    # inv_x_N*self.v*alpha*Constant(-0.22602295050021465)*self.n*k*self.ds(1) 
                if self.mms:
                    F_q = F_q + self.v*self.s_q*k*dx

                # conservation
                F_h = self.v*(self.h[0] - self.h[1])*dx + \
                    inv_x_N*self.v*grad(q_td)*k*dx - \
                    inv_x_N*self.v*self.X*u_N_td*grad(h_td)*k*dx 
                if self.mms:
                    F_h = F_h + self.v*self.s_h*k*dx

                # concentration
                F_phi = self.v*(self.phi[0] - self.phi[1])*dx + \
                    inv_x_N*self.v*grad(q_td*phi_td/h_td)*k*dx - \
                    inv_x_N*self.v*self.X*u_N_td*grad(phi_td)*k*dx + \
                    self.v*self.u_sink*phi_td/h_td*k*dx 
                if self.mms:
                    F_phi = F_phi + self.v*self.s_phi*k*dx

                # deposit
                F_c_d = self.v*(self.c_d[0] - self.c_d[1])*dx - \
                    inv_x_N*self.v*self.X*u_N_td*grad(c_d_td)*k*dx - \
                    self.v*self.u_sink*phi_td/(self.rho_R*self.g*h_td)*k*dx
                if self.mms:
                    F_c_d = F_c_d + self.v*self.s_c_d*k*dx

                # nose location/speed
                F_u_N = self.r*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
                    self.r*self.u_N[0]*self.ds(1)
                F_x_N = self.r*(self.x_N[0] - self.x_N[1])*dx - self.r*u_N_td*k*dx 

                # SOLVE EQUATIONS
                solve(F_q == 0, self.q[0], self.bcq)
                solve(F_h == 0, self.h[0], self.bch)
                solve(F_phi == 0, self.phi[0], self.bcphi)
                solve(F_c_d == 0, self.c_d[0], self.bcc_d)
                solve(F_u_N == 0, self.u_N[0])
                if not self.mms:
                    solve(F_x_N == 0, self.x_N[0])

                dh = errornorm(h_nl, self.h[0], norm_type="L2", degree=self.shape + 1)
                dphi = errornorm(phi_nl, self.phi[0], norm_type="L2", degree=self.shape + 1)
                dq = errornorm(q_nl, self.q[0], norm_type="L2", degree=self.shape + 1)
                dx_N = errornorm(x_N_nl, self.x_N[0], norm_type="L2", degree=self.shape + 1)
                du_nl = max(dh, dphi, dq, dx_N)/self.timestep

                nl_its += 1

            dh = errornorm(self.h[0], self.h[1], norm_type="L2", degree=self.shape + 1)
            dphi = errornorm(self.phi[0], self.phi[1], norm_type="L2", degree=self.shape + 1)
            dq = errornorm(self.q[0], self.q[1], norm_type="L2", degree=self.shape + 1)
            dx_N = errornorm(self.x_N[0], self.x_N[1], norm_type="L2", degree=self.shape + 1)
            du = max(dh, dphi, dq, dx_N)/self.timestep

            self.h[1].assign(self.h[0])
            self.phi[1].assign(self.phi[0])
            self.c_d[1].assign(self.c_d[0])
            self.q[1].assign(self.q[0])
            self.x_N[1].assign(self.x_N[0])
            self.u_N[1].assign(self.u_N[0])

            # display results
            self.update_plot()
            self.print_timestep_info(nl_its, du)

    def initialise_plot(self, x_max, h_y_lim = 0.5, u_y_lim = 0.3, phi_y_lim = 0.01, c_d_y_lim = 1.2e-4):
        # fig settings
        self.plot_x = np.linspace(0.0, x_max, 10001)
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
        self.vel_plot.set_ylim([0.0,u_y_lim])
        self.c_plot.set_autoscaley_on(False)
        self.c_plot.set_ylim([0.0,phi_y_lim])
        self.c_d_plot.set_autoscaley_on(False)
        self.c_d_plot.set_ylim([0.0,c_d_y_lim])

        # set initial plot values
        self.vel_line, = self.vel_plot.plot(self.plot_x, 
                                            self.y_data(self.q[0].vector().array()/self.h[0].vector().array()), 'r-')
        if self.mms:
            self.vel_line_2, = self.vel_plot.plot(self.plot_x, 
                                                  self.y_data(self.q[0].vector().array()/self.h[0].vector().array()), 'b-')
        self.h_line, = self.h_plot.plot(self.plot_x, 
                                        self.y_data(self.h[0].vector().array()), 'r-')
        if self.mms:
            self.h_line_2, = self.h_plot.plot(self.plot_x, 
                                        self.y_data(self.h[0].vector().array()), 'b-')
        self.c_line, = self.c_plot.plot(self.plot_x, 
                                        self.y_data(self.phi[0].vector().array()/
                                                             (self.rho_R_*self.g_*self.h[0].vector().array())
                                                             ), 'r-')
        if self.mms:
            self.c_line_2, = self.c_plot.plot(self.plot_x, 
                                        self.y_data(self.phi[0].vector().array()/
                                                             (self.rho_R_*self.g_*self.h[0].vector().array())
                                                             ), 'b-')
        self.c_d_line, = self.c_d_plot.plot(self.plot_x, 
                                            self.y_data(self.c_d[0].vector().array()), 'r-')
        if self.mms:
            self.c_d_line_2, = self.c_d_plot.plot(self.plot_x, 
                                            self.y_data(self.c_d[0].vector().array()), 'b-')

        self.fig.canvas.draw()
        self.fig.savefig('results/%06.2f.png' % (0.0))       

    def update_plot(self):
        self.vel_line.set_ydata(self.y_data(self.q[0].vector().array()/self.h[0].vector().array()))
        self.h_line.set_ydata(self.y_data(self.h[0].vector().array()))
        self.c_line.set_ydata(self.y_data(self.phi[0].vector().array()/(self.rho_R_*self.g_*self.h[0].vector().array())))
        self.c_d_line.set_ydata(self.y_data(self.c_d[0].vector().array()))
        self.fig.canvas.draw()
        self.fig.savefig('results/{:06.2f}.png'.format(self.t)) 

    def y_data(self, u):
        val_x = np.linspace(0.0, self.x_N[0].vector().array()[-1], self.L/self.dX_ + 1)
        val = u[:self.L/self.dX_ + 1]
        return np.interp(self.plot_x, val_x, val, right=0.0)  
        
    def print_timestep_info(self, nl_its, du):
        mass = (self.h[0].vector().array()[:self.L/self.dX_ + 1]*(self.x_N[0].vector().array()[-1]*self.dX_)).sum()
        info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, u_N_2 = %.2e, h_N = %.2e, nl_its = %1d, mass = %.2e, du = %.2e" % 
                   (self.t, self.timestep, 
                    self.x_N[0].vector().array()[-1], 
                    self.u_N[0].vector().array()[0], 
                    self.q[0].vector().array()[-1]/self.h[0].vector().array()[-1], 
                    self.h[0].vector().array()[-1],
                    nl_its, 
                    mass,
                    du))
        # info_red(self.c_d[0].vector().array().max())

    def getError(self):
        S_h = Expression(mms.h(), degree=self.shape + 1)
        S_phi = Expression(mms.phi(), degree=self.shape + 1)
        S_q = Expression(mms.q(), degree=self.shape + 1)
        S_u_N = Expression(mms.u_N(), degree=self.shape + 1)

        Eh = errornorm(self.h[0], S_h, norm_type="L2", degree=self.shape + 1)
        Ephi = errornorm(self.phi[0], S_phi, norm_type="L2", degree=self.shape + 1)
        Eq = errornorm(self.q[0], S_q, norm_type="L2", degree=self.shape + 1)
        Eu_N = errornorm(self.u_N[0], S_u_N, norm_type="L2", degree=self.shape + 1)

        return Eh, Ephi, Eq, Eu_N

if __name__ == '__main__':

    parser = OptionParser()
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)
    parser.add_option('-m', '--mms',
                      action='store_true', dest='mms', default=False,
                      help='mms test')
    (options, args) = parser.parse_args()
    
    model = Model()

    if options.mms == True:
        h = [] # element sizes
        E = [] # errors
        for i, nx in enumerate([32, 64, 128, 256]):
            dT = (pi/nx) * 0.5
            h.append(pi/nx)
            print 'dt is: ', dT, '; h is: ', h[-1]
            model.mms_setup(h[-1], dT)
            model.solve(tol = 5e-2)
            E.append(model.getError())

        # Convergence rates
        from math import log as ln # (log is a dolfin name too)

        for i in range(1, len(E)):
            rh = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
            rphi = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
            rq = ln(E[i][2]/E[i-1][2])/ln(h[i]/h[i-1])
            ru_N = ln(E[i][3]/E[i-1][3])/ln(h[i]/h[i-1])
            print "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f ru_N=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Eu_N=%.2e" % (h[i], rh, rphi, rq, ru_N, E[i][0], E[i][1], E[i][2], E[i][3]) 

    else:
        model.setup()
        model.solve(T = 1.0)

        # deposit = model.c_d[0].vector().array()
        # deposit_x = np.linspace(0.0, model.x_N[0].vector().array()[-1], model.L/model.dX_ + 1)

        # class DepositExpression(Expression):
        #     current_x_N = 1.0
        #     L = 1.0
        #     dX = 0.1

        #     def __init__(target_array, target_x):
        #         self.target = target
        #         self.target_x = target_x

        #     def eval(self, value, x):
        #         X = x[0]*current_x_N/L
        #         value[0] = np.interp(X, self.target_x, self.target, right=0.0)

        c_d_desired = model.c_d[0].copy(deepcopy=True)

        model.setup(c_0 = 0.0001)
        model.solve(T = 1.0)
        
        J = Functional((model.c_d[0]-c_d_desired)*dx*dt[FINISH_TIME])
        reduced_functional = ReducedFunctional(J, InitialConditionParameter(model.phi[0]))
        
        m_opt = minimize(reduced_functional, method = "L-BFGS-B")

        # deposit_desired = model.c_d[0].copy(deepcopy=True)

        # # left boundary marked as 0, right as 1
        # class MeasureLocation1(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return True if ( x[0] > 0.1 and x[0] < 0.11
        # left_boundary = LeftBoundary()
        # exterior_facet_domains = FacetFunction("uint", self.mesh)
        # exterior_facet_domains.set_all(1)
        # left_boundary.mark(exterior_facet_domains, 0)
        # self.ds = Measure("ds")[exterior_facet_domains] 
