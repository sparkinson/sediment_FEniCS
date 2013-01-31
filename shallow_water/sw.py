#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
from dolfin_adjoint import *
import numpy as np
from optparse import OptionParser
import json
import sw_output

############################################################
# DOLFIN SETTINGS

parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters['krylov_solver']['relative_tolerance'] = 1e-15
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
# info(parameters, True)
#set_log_active(False)
set_log_level(PROGRESS)

# parameters["adjoint"]["stop_annotating"] = True

############################################################
# TIME DISCRETISATION FUNCTIONS

class Model():
    
    # SIMULAITON USER DEFINED PARAMETERS

    # mesh
    dX_ = 5.0e-2
    L = 1.0

    # stabilisation
    b_ = 0.0

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
    timestep = 1e-3

    # mms test (default False)
    mms = False

    # display plot
    plot = True

    def initialise_function_spaces(self):

        # define geometry
        self.mesh = IntervalMesh(int(self.L/self.dX_), 0.0, self.L)
        self.n = FacetNormal(self.mesh)

        # define function spaces
        self.h_degree = 1
        self.h_CG = FunctionSpace(self.mesh, "CG", self.h_degree)
        self.q_degree = 1
        self.q_CG = FunctionSpace(self.mesh, "CG", self.q_degree)
        self.R = FunctionSpace(self.mesh, "R", 0)
        
        # define test functions
        self.v1 = TestFunction(self.h_CG)
        self.v2 = TestFunction(self.q_CG)
        self.r = TestFunction(self.R)        

    def setup(self, h_ic = None, phi_ic = None, q_ic = None):

        # define constants
        self.dX = Constant(self.dX_, name="dX")
        self.g = Constant(self.g_, name="g")
        self.rho_R = Constant(self.rho_R_, name="rho_R")
        self.b = Constant(self.b_, name="b")
        self.Fr = Constant(self.Fr_, name="Fr")
        self.u_sink = Constant(self.u_sink_, name="u_sink")

        # define initial conditions
        if h_ic:
            self.h_ic = h_ic
        else:
            self.h_ic = project(Expression(str(self.h_0)), self.h_CG)

        if phi_ic:
            self.phi_ic = phi_ic
        else:
            self.phi_ic = project(Expression(str(self.c_0*self.rho_R_*self.g_*self.h_0)), self.h_CG)

        if q_ic:
            self.q_ic = q_ic
        else:
            x = (1.0)
            self.q_ic = project(Expression('(1.0 - cos((x[0]/{0})*pi))*{1}/2.0'
                                           .format(self.L, self.Fr_*self.phi_ic(x)**0.5*self.h_ic(x)))
                                , self.q_CG)
            # self.q_ic = project(Expression('0.0'), self.q_CG)

        self.c_d_ic = project(Expression('0.0'), self.h_CG)
        self.x_N_ic = project(Expression(str(self.x_N_)), self.R)
        self.u_N_ic = project(Expression('0.0'), self.R)

        # initialise functions
        self.initialise_functions()

        # define bc's
        self.bch = []
        self.bcphi = []
        self.bcc_d = [DirichletBC(self.h_CG, '0.0', "near(x[0], 1.0) && on_boundary")]
        self.bcq = [DirichletBC(self.q_CG, '0.0', "near(x[0], 0.0) && on_boundary")]

        # initialise plotting
        if self.plot:
            self.plotter = sw_output.Plotter(self)

        # define form of equations
        self.form()

    def initialise_functions(self):
        # define function dictionaries for prognostic variables
        self.h = dict([[i, Function(self.h_ic, name='h_{}'.format(i))] for i in range(2)])
        # self.phi = dict([[i, Function(phi_ic, name='phi_{}'.format(i))] for i in range(2)])
        self.c_d = dict([[i, Function(self.c_d_ic, name='c_d_{}'.format(i))] for i in range(2)])
        self.q = dict([[i, Function(self.q_ic, name='q_{}'.format(i))] for i in range(2)])
        self.x_N = dict([[i, Function(self.x_N_ic, name='x_N_{}'.format(i))] for i in range(2)])
        self.u_N = dict([[i, Function(self.u_N_ic, name='u_N_{}'.format(i))] for i in range(2)])
        self.phi = dict()
        # self.phi[0] = project(self.phi_ic, self.h_CG)
        self.phi[0] = Function(self.phi_ic, name="phi_0")
        self.phi[1] = project(self.phi[0], self.h_CG)
        X_ = project(Expression('x[0]'), self.h_CG)
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

    def form(self):

        def smoothed_min(val, min, eps):
            return 0.5*(((val - min - eps)**2.0)**0.5 + min + val)

        self.k = Constant(self.timestep)

        # time discretisation of values
        def time_discretise(u):
            return 0.5*u[0] + 0.5*u[1]

        x_N_td = time_discretise(self.x_N)
        inv_x_N = 1./x_N_td
        u_N_td = time_discretise(self.u_N)
        h_td = time_discretise(self.h)
        smoothed_min(h_td, 1e-8, 1e-10)
        phi_td = time_discretise(self.phi)
        c_d_td = time_discretise(self.c_d)
        q_td = time_discretise(self.q)

        # momentum
        q_N = u_N_td*h_td
        u = q_td/h_td
        # alpha = 0.0 
        alpha = self.b*self.dX*(abs(u)+u+(phi_td*h_td)**0.5)*h_td
        self.F_q = self.v2*(self.q[0] - self.q[1])*dx + \
            inv_x_N*self.v2*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)*self.k*dx + \
            inv_x_N*u_N_td*grad(self.v2*self.X)*q_td*self.k*dx - \
            inv_x_N*u_N_td*self.v2*self.X*q_N*self.n*self.k*self.ds(1) + \
            inv_x_N*grad(self.v2)*alpha*grad(u)*self.k*dx - \
            inv_x_N*self.v2*alpha*grad(u)*self.n*self.k*self.ds(1) 
            # inv_x_N*self.v2*alpha*Constant(-0.22602295050021465)*self.n*self.k*self.ds(1) 
        if self.mms:
            self.F_q = self.F_q + self.v2*self.s_q*self.k*dx

        # conservation
        self.F_h = self.v1*(self.h[0] - self.h[1])*dx + \
            inv_x_N*self.v1*grad(q_td)*self.k*dx - \
            inv_x_N*self.v1*self.X*u_N_td*grad(h_td)*self.k*dx 
        if self.mms:
            self.F_h = self.F_h + self.v1*self.s_h*self.k*dx

        # concentration
        self.F_phi = self.v1*(self.phi[0] - self.phi[1])*dx + \
            inv_x_N*self.v1*grad(q_td*phi_td/h_td)*self.k*dx - \
            inv_x_N*self.v1*self.X*u_N_td*grad(phi_td)*self.k*dx + \
            self.v1*self.u_sink*phi_td/h_td*self.k*dx 
        if self.mms:
            self.F_phi = self.F_phi + self.v1*self.s_phi*self.k*dx

        # nose location/speed
        self.F_u_N = self.r*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            self.r*self.u_N[0]*self.ds(1)
        self.F_x_N = self.r*(self.x_N[0] - self.x_N[1])*dx - self.r*u_N_td*self.k*dx 

        # deposit
        self.F_c_d = self.v1*(self.c_d[0] - self.c_d[1])*dx - \
            inv_x_N*self.v1*self.X*u_N_td*grad(c_d_td)*self.k*dx - \
            self.v1*self.u_sink*phi_td/(self.rho_R*self.g*h_td)*self.k*dx
        if self.mms:
            self.F_c_d = self.F_c_d + self.v1*self.s_c_d*self.k*dx

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
        
        # if not self.mms:
        #     adj_start_timestep()

        self.t = 0.0
        du = 1e10
        while not (time_finish(self.t) or converged(du)):
            
            # THIS IS WHERE ADAPTIVE TIMESTEP WILL GO
            self.k.assign(self.timestep)

            # if self.t > self.timestep*6 and not self.mms:
            #     parameters["adjoint"]["stop_annotating"] = False

            ss = 1.0
            nl_its = 0
            while (nl_its < 2 or du_nl > nl_tol):

                # VALUES FOR CONVERGENCE TEST
                h_nl = self.h[0].copy(deepcopy=True)
                phi_nl = self.phi[0].copy(deepcopy=True)
                q_nl = self.q[0].copy(deepcopy=True)
                x_N_nl = self.x_N[0].copy(deepcopy=True)

                # SOLVE COUPLED EQUATIONS
                solve(self.F_q == 0, self.q[0], self.bcq)
                solve(self.F_h == 0, self.h[0], self.bch)
                solve(self.F_phi == 0, self.phi[0], self.bcphi)
                solve(self.F_u_N == 0, self.u_N[0])
                if not self.mms:
                    solve(self.F_x_N == 0, self.x_N[0])

                dh = errornorm(h_nl, self.h[0], norm_type="L2", degree_rise=1)
                dphi = errornorm(phi_nl, self.phi[0], norm_type="L2", degree_rise=1)
                dq = errornorm(q_nl, self.q[0], norm_type="L2", degree_rise=1)
                dx_N = errornorm(x_N_nl, self.x_N[0], norm_type="L2", degree_rise=1)
                du_nl = max(dh, dphi, dq, dx_N)/self.timestep

                nl_its += 1
            
            # SOLVE NON-COUPLED EQUATIONS
            solve(self.F_c_d == 0, self.c_d[0], self.bcc_d)

            dh = errornorm(self.h[0], self.h[1], norm_type="L2", degree_rise=1)
            dphi = errornorm(self.phi[0], self.phi[1], norm_type="L2", degree_rise=1)
            dq = errornorm(self.q[0], self.q[1], norm_type="L2", degree_rise=1)
            dx_N = errornorm(self.x_N[0], self.x_N[1], norm_type="L2", degree_rise=1)
            du = max(dh, dphi, dq, dx_N)/self.timestep

            self.h[1].assign(self.h[0])
            self.phi[1].assign(self.phi[0])
            self.c_d[1].assign(self.c_d[0])
            self.q[1].assign(self.q[0])
            self.x_N[1].assign(self.x_N[0])
            self.u_N[1].assign(self.u_N[0])

            self.t += self.timestep
            # display results
            if self.plot:
                self.plotter.update_plot(self)
            sw_output.print_timestep_info(self, nl_its, du)

            # if not self.mms:
            #     adj_inc_timestep() 

if __name__ == '__main__':

    parser = OptionParser()
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)
    parser.add_option('-a', '--adjoint',
                      action='store_true', dest='adjoint', default=False,
                      help='adjoint run')
    parser.add_option('-t', '--phi_ic_test',
                      action='store_true', dest='phi_ic_test', default=False,
                      help='test phi initial conditions')
    parser.add_option('-T', '--end_time',
                      dest='T', type=float, default=0.2,
                      help='simulation end time')
    (options, args) = parser.parse_args()
    
    model = Model()

    # Adjoint taylor test
    if options.taylor_test_u_sink == True:

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

    # Adjoint 
    elif options.adjoint == True:

        model.plot = False
        model.initialise_function_spaces()

        f = open('deposit_data.json','r')
        json_string = f.readline()
        data = np.array(json.loads(json_string))
        c_d_aim = Function(model.h_CG)
        c_d_aim.vector()[:] = data
        print c_d_aim.vector().array()

        f = open('length_data.json','r')
        json_string = f.readline()
        data = np.array(json.loads(json_string))
        x_N_aim = Function(model.R)
        x_N_aim.vector()[:] = data
        print x_N_aim.vector().array()

        phi_ic = project(Expression('0.01'), model.h_CG)
        model.setup(phi_ic = phi_ic)
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

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dJdphi.vector().array())
        plt.show()

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
        phi_ic = Function(model.h_CG)
        phi_ic.vector()[:] = data
        
        model.setup(phi_ic = phi_ic)

        model.solve(T = options.T)    

    else:        

        model.initialise_function_spaces()
        phi_ic = project(Expression('0.03+0.005*sin(2*pi*x[0])'), model.h_CG)
        model.setup(phi_ic=phi_ic)
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
