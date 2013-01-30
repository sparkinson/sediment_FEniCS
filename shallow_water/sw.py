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
import json

############################################################
# DOLFIN SETTINGS

parameters['krylov_solver']['relative_tolerance'] = 1e-15
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
# info(parameters, True)
#set_log_active(False)
set_log_level(ERROR)

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

    # mesh
    dX_ = 2.5e-2
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
    timestep = 2.0e-2

    # define time discretisation
    td = crank_nicholson

    # mms test (default False)
    mms = False

    # display plot
    plot = True

    def initialise_function_spaces(self):

        # define geometry
        self.mesh = Interval(int(self.L/self.dX_), 0.0, self.L)
        self.n = FacetNormal(self.mesh)

        # define function spaces
        self.P2CG = FunctionSpace(self.mesh, "CG", 2)
        self.P1CG = FunctionSpace(self.mesh, "CG", 1)
        self.R = FunctionSpace(self.mesh, "R", 0)
        
        # define test functions
        self.v = TestFunction(self.P1CG)
        self.r = TestFunction(self.R)        

    def setup(self, phi_ic_override = None):

        # define constants
        self.dX = Constant(self.dX_, name="dX")
        self.g = Constant(self.g_, name="g")
        self.rho_R = Constant(self.rho_R_, name="rho_R")
        self.b = Constant(self.b_, name="b")
        self.Fr = Constant(self.Fr_, name="Fr")
        self.u_sink = Constant(self.u_sink_, name="u_sink")

        h_ic = project(Expression(str(self.h_0)), self.P1CG)
        if phi_ic_override:
            self.phi_ic = phi_ic_override
        else:
            self.phi_ic = project(Expression(str(self.c_0*self.rho_R_*self.g_*self.h_0)), self.P1CG)
        c_d_ic = project(Expression('0.0'), self.P1CG)
        q_ic = project(Expression('0.0'), self.P1CG)
        x_N_ic = project(Expression(str(self.x_N_)), self.R)
        u_N_ic = project(Expression('0.0'), self.R)

        self.initialise_functions(h_ic, c_d_ic, q_ic, x_N_ic, u_N_ic)

        # define bc's
        self.bch = []
        self.bcphi = []
        self.bcc_d = [DirichletBC(self.P1CG, '0.0', "near(x[0], 1.0) && on_boundary")]
        self.bcq = [DirichletBC(self.P1CG, '0.0', "near(x[0], 0.0) && on_boundary")]

        # initialise plot
        if self.plot:
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
        
        self.initialise_function_spaces()

        self.initialise_functions(h_ic = project(Expression(mms.h()), self.P1CG), 
                                  phi_ic = project(Expression(mms.phi()), self.P1CG),
                                  c_d_ic = project(Expression(mms.c_d()), self.P1CG), 
                                  q_ic = project(Expression(mms.q()), self.P1CG), 
                                  x_N_ic = project(Expression('pi'), self.R), 
                                  u_N_ic = project(Expression(mms.u_N()), self.R))

        # define bc's
        self.bch = [DirichletBC(self.P1CG, Expression(mms.h()), 
                                "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcphi = [DirichletBC(self.P1CG, Expression(mms.phi()), 
                                  "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcc_d = [DirichletBC(self.P1CG, Expression(mms.c_d()), 
                                  "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcq = [DirichletBC(self.P1CG, Expression(mms.q()), "(near(x[0], 0.0)) && on_boundary")]
                                # "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]

        # define source terms
        self.s_q = Expression(mms.s_q())
        self.s_h = Expression(mms.s_h())
        self.s_phi = Expression(mms.s_phi())
        self.s_c_d = Expression(mms.s_c_d())

        # initialise plot
        if self.plot:
            self.initialise_plot(np.pi, h_y_lim = 20.0, u_y_lim = 1.5, phi_y_lim = 0.2, c_d_y_lim = 5.0) 

        self.timestep = dT

    def initialise_functions(self, h_ic, c_d_ic, q_ic, x_N_ic, u_N_ic):
        # define function dictionaries for prognostic variables
        self.h = dict([[i, Function(h_ic, name='h_{}'.format(i))] for i in range(2)])
        # self.phi = dict([[i, Function(phi_ic, name='phi_{}'.format(i))] for i in range(2)])
        self.c_d = dict([[i, Function(c_d_ic, name='c_d_{}'.format(i))] for i in range(2)])
        self.q = dict([[i, Function(q_ic, name='q_{}'.format(i))] for i in range(2)])
        self.x_N = dict([[i, Function(x_N_ic, name='x_N_{}'.format(i))] for i in range(2)])
        self.u_N = dict([[i, Function(u_N_ic, name='u_N_{}'.format(i))] for i in range(2)])
        self.phi = dict()
        # self.phi[0] = project(self.phi_ic, self.P1CG)
        self.phi[0] = Function(self.phi_ic, name="phi_0")
        self.phi[1] = project(self.phi[0], self.P1CG)
        X_ = project(Expression('x[0]'), self.P1CG)
        self.X = Function(X_, name='X')

        # left boundary marked as 0, right as 1
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        left_boundary = LeftBoundary()
        exterior_facet_domains = FacetFunction("uint", self.mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains] 

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

        def smoothed_min(val, min, eps):
            return 0.5*(((val - min - eps)**2.0)**0.5 + min + val)
        
        if not self.mms:
            adj_start_timestep()

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
                smoothed_min(h_td, 1e-8, 1e-10)
                phi_td = self.td(self.phi)
                c_d_td = self.td(self.c_d)
                q_td = self.td(self.q)

                # momentum
                q_N = u_N_td*h_td
                u = q_td/h_td
                # alpha = 0.0 
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

                # nose location/speed
                F_u_N = self.r*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
                    self.r*self.u_N[0]*self.ds(1)
                F_x_N = self.r*(self.x_N[0] - self.x_N[1])*dx - self.r*u_N_td*k*dx 

                # SOLVE COUPLED EQUATIONS
                solve(F_q == 0, self.q[0], self.bcq)
                solve(F_h == 0, self.h[0], self.bch)
                solve(F_phi == 0, self.phi[0], self.bcphi)
                solve(F_u_N == 0, self.u_N[0])
                if not self.mms:
                    solve(F_x_N == 0, self.x_N[0])

                dh = errornorm(h_nl, self.h[0], norm_type="L2", degree=3)
                dphi = errornorm(phi_nl, self.phi[0], norm_type="L2", degree=3)
                dq = errornorm(q_nl, self.q[0], norm_type="L2", degree=3)
                dx_N = errornorm(x_N_nl, self.x_N[0], norm_type="L2", degree=3)
                du_nl = max(dh, dphi, dq, dx_N)/self.timestep

                nl_its += 1

            # deposit
            F_c_d = self.v*(self.c_d[0] - self.c_d[1])*dx - \
                inv_x_N*self.v*self.X*u_N_td*grad(c_d_td)*k*dx - \
                self.v*self.u_sink*phi_td/(self.rho_R*self.g*h_td)*k*dx
            if self.mms:
                F_c_d = F_c_d + self.v*self.s_c_d*k*dx
            
            # SOLVE NON-COUPLED EQUATIONS
            solve(F_c_d == 0, self.c_d[0], self.bcc_d)

            dh = errornorm(self.h[0], self.h[1], norm_type="L2", degree=3)
            dphi = errornorm(self.phi[0], self.phi[1], norm_type="L2", degree=3)
            dq = errornorm(self.q[0], self.q[1], norm_type="L2", degree=3)
            dx_N = errornorm(self.x_N[0], self.x_N[1], norm_type="L2", degree=3)
            du = max(dh, dphi, dq, dx_N)/self.timestep

            self.h[1].assign(self.h[0])
            self.phi[1].assign(self.phi[0])
            self.c_d[1].assign(self.c_d[0])
            self.q[1].assign(self.q[0])
            self.x_N[1].assign(self.x_N[0])
            self.u_N[1].assign(self.u_N[0])

            # display results
            if self.plot:
                self.update_plot()
            self.print_timestep_info(nl_its, du)

            if not self.mms:
                adj_inc_timestep()

    def initialise_plot(self, x_max, h_y_lim = 0.5, u_y_lim = 0.3, phi_y_lim = 0.03, c_d_y_lim = 1.2e-5):
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
                                                  self.y_data(self.q[1].vector().array()/self.h[0].vector().array()), 'b-')
        self.h_line, = self.h_plot.plot(self.plot_x, 
                                        self.y_data(self.h[0].vector().array()), 'r-')
        if self.mms:
            self.h_line_2, = self.h_plot.plot(self.plot_x, 
                                              self.y_data(self.h[1].vector().array()), 'b-')
        self.c_line, = self.c_plot.plot(self.plot_x, 
                                        self.y_data(self.phi[0].vector().array()/
                                                             (self.rho_R_*self.g_*self.h[0].vector().array())
                                                             ), 'r-')
        if self.mms:
            self.c_line_2, = self.c_plot.plot(self.plot_x, 
                                              self.y_data(self.phi[1].vector().array()/
                                                          (self.rho_R_*self.g_*self.h[0].vector().array())
                                                          ), 'b-')
        self.c_d_line, = self.c_d_plot.plot(self.plot_x, 
                                            self.y_data(self.c_d[0].vector().array()), 'r-')
        if self.mms:
            self.c_d_line_2, = self.c_d_plot.plot(self.plot_x, 
                                                  self.y_data(self.c_d[1].vector().array()), 'b-')

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
    parser.add_option('-t', '--taylor_test_u_sink',
                      action='store_true', dest='taylor_test_u_sink', default=False,
                      help='adjoint taylor test when varying the u_sink parameter')
    parser.add_option('-i', '--taylor_test_phi_1',
                      action='store_true', dest='taylor_test_phi_1', default=False,
                      help='adjoint taylor test when varying initial condition parameter phi_1')
    parser.add_option('-a', '--adjoint',
                      action='store_true', dest='adjoint', default=False,
                      help='adjoint run')
    parser.add_option('-p', '--phi_ic_test',
                      action='store_true', dest='phi_ic_test', default=False,
                      help='test phi initial conditions')
    parser.add_option('-T', '--end_time',
                      dest='T', type=float, default=0.2,
                      help='simulation end time')
    (options, args) = parser.parse_args()
    
    model = Model()

    # MMS test
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
            print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f ru_N=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Eu_N=%.2e" 
                    % (h[i], rh, rphi, rq, ru_N, E[i][0], E[i][1], E[i][2], E[i][3]) )

    # Adjoint taylor test
    elif options.taylor_test_u_sink == True:

        model.plot = False
        model.initialise_function_spaces()

        info_blue('Taylor test for u_sink')
        
        model.setup()
        model.solve(T = 0.05)

        c_d = model.c_d[0]
        J = Functional(c_d*dx*dt[FINISH_TIME])

        parameters["adjoint"]["stop_annotating"] = True # stop registering equations

        dJdu_sink = compute_gradient(J, ScalarParameter("u_sink"))
        Ju_sink = assemble(c_d*dx)

        def Jhat(u_sink): # the functional as a pure function of u_sink
            model.u_sink_ = u_sink
            model.setup()
            model.solve(T = 0.05)

            c_d = model.c_d[0]
            return assemble(c_d*dx)

        # set_log_active(True)

        conv_rate = taylor_test(Jhat, ScalarParameter("u_sink"), Ju_sink, dJdu_sink)

        info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
        if conv_rate > 1.9:
            info_blue('*** test passed ***')
        else:
            info_red('*** ERROR: test failed ***')

    # Adjoint taylor test
    elif options.taylor_test_phi_1 == True:

        model.plot = False
        model.initialise_function_spaces()

        info_blue('Taylor test for phi')

        # parameters["adjoint"]["stop_annotating"] = False # start registering equations

        model.setup()
        model.solve(T = 0.05)

        c_d = model.c_d[0]

        J = Functional(c_d*dx*dt[FINISH_TIME])
        # reduced_functional = ReducedFunctional(J, InitialConditionParameter("PhiInitialCondition"))
        # m_opt = minimize(reduced_functional, method = "L-BFGS-B", options = {'disp': True})

        parameters["adjoint"]["stop_annotating"] = True # stop registering equations

        # for some reason this only works with forget=False
        dJdphi = compute_gradient(J, InitialConditionParameter("phi_0"), forget=False)
        Jphi = assemble(c_d*dx)
        print dJdphi.vector().array(), Jphi

        def Jhat(phi_ic): # the functional as a pure function of phi_ic
            model.setup(phi_ic)
            model.solve(T = 0.05)

            c_d = model.c_d[0]
            print 'Jhat: ', assemble(c_d*dx)
            return assemble(c_d*dx)

        # set_log_active(True)

        conv_rate = taylor_test(Jhat, InitialConditionParameter("phi_0"), Jphi, dJdphi, value = model.phi_ic, seed=1e-1)

        info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
        if conv_rate > 1.9:
            info_blue('*** test passed ***')
        else:
            info_red('*** ERROR: test failed ***')

    # Adjoint 
    elif options.adjoint == True:

        model.plot = False
        model.initialise_function_spaces()

        f = open('deposit_data.json','r')
        json_string = f.readline()
        data = np.array(json.loads(json_string))
        c_d_aim = Function(model.P1CG)
        c_d_aim.vector()[:] = data
        print c_d_aim.vector().array()

        f = open('length_data.json','r')
        json_string = f.readline()
        data = np.array(json.loads(json_string))
        x_N_aim = Function(model.R)
        x_N_aim.vector()[:] = data
        print x_N_aim.vector().array()

        phi_ic = project(Expression('0.01'), model.P1CG)
        model.setup(phi_ic)
        model.solve(T = options.T)

        adj_html("forward.html", "forward")
        adj_html("adjoint.html", "adjoint")

        # functional components
        int_0_scale = Constant(1)
        int_0_scale_2 = Constant(1)
        int_1_scale = Constant(1)

        int_0 = inner(model.c_d[0]-c_d_aim, model.c_d[0]-c_d_aim)*int_0_scale*dx
        int_0_2_d = inner(grad(model.c_d[0])-grad(c_d_aim), grad(model.c_d[0])-grad(c_d_aim))*int_0_scale_2
        int_0_2 = int_0_2_d*ds(0) + int_0_2_d*ds(1)
        int_1 = inner(model.x_N[0]-x_N_aim, model.x_N[0]-x_N_aim)*int_1_scale*dx

        int_0_scale.assign(1e-2/assemble(int_0))
        int_0_scale_2.assign(0) # 1e-4/assemble(int_0_2))
        int_1_scale.assign(1e-4/assemble(int_1))
        print assemble(int_0)
        print assemble(int_0_2)
        print assemble(int_1)
        ### int_0 1e-2, int_1 1e-4 - worked well
        
        ## functional regularisation
        reg_scale = Constant(1)
        int_reg = inner(grad(model.phi[0]), grad(model.phi[0]))*reg_scale*dx
        reg_scale_base = 1e-2
        reg_scale.assign(reg_scale_base)

        ## functional
        J = Functional((int_0 + int_0_2 + int_1)*dt[FINISH_TIME] + int_reg*dt[START_TIME])

        ## display gradient information
        dJdphi = compute_gradient(J, InitialConditionParameter("phi_0"))
        
        import IPython
        IPython.embed()

        # clear old data
        f = open('phi_ic_adj.json','w')
        f.close()

        j_log = []

        def eval_cb(j, m):
            print "* * * Completed forward model"
            print "j = {}".format(j)
            j_log.append(j)
            f = open('j_log.json','w')
            f.write(json.dumps(j_log))
            f.close()

        ##############################
        #### REDUCED FUNCTIONAL HACK
        ##############################

        class MyReducedFunctional(ReducedFunctional):

            def __call__(self, value):

                #### initial condition dump hack ####
                ic = list(value[0].vector().array())
                f = open('phi_ic_adj_latest.json','w')
                f.write(json.dumps(ic))
                f.write('\n')
                f.close()
                f = open('phi_ic_adj.json','a')
                f.write(json.dumps(ic))
                f.write('\n')
                f.close()

                print "\n* * * Computing forward model"

                return (super(MyReducedFunctional, self)).__call__(value)

        #######################################
        #### END OF REDUCED FUNCTIONAL HACK
        #######################################

        reduced_functional = MyReducedFunctional(J, InitialConditionParameter("phi_0"),
                                                 eval_cb = eval_cb,
                                                 scale = 1e-0)
        
        for i in range(10):
            reg_scale.assign(reg_scale_base*2**(0-i))

            m_opt = minimize(reduced_functional, method = "L-BFGS-B", options = {'maxiter': 4, 'disp': True, 'gtol': 1e-9, 'ftol': 1e-9}, 
                             bounds = (1e-7, 1e-0))

        print m_opt.vector().array()

    # Adjoint 
    elif options.phi_ic_test == True:

        model.initialise_function_spaces()
        f = open('phi_ic_adj_latest.json','r')
        json_string = f.readline()
        data = np.array(json.loads(json_string))
        phi_ic = Function(model.P1CG)
        phi_ic.vector()[:] = data
        
        model.setup(phi_ic)

        model.solve(T = options.T)    

    else:        

        model.initialise_function_spaces()
        phi_ic = project(Expression('0.03+0.005*sin(2*pi*x[0])'), model.P1CG)
        
        model.setup(phi_ic)
        data = list(model.phi[0].vector().array())
        f = open('phi_ic.json','w')
        f.write(json.dumps(data))
        f.close()

        model.solve(T = options.T)

        data = list(model.c_d[0].vector().array())
        f = open('deposit_data.json','w')
        f.write(json.dumps(data))
        f.close()

        data = list(model.x_N[0].vector().array())
        f = open('length_data.json','w')
        f.write(json.dumps(data))
        f.close()
