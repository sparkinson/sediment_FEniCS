#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
from dolfin_adjoint import *
import numpy as np
from optparse import OptionParser
import json
import sw_io

############################################################
# DOLFIN SETTINGS

parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters['krylov_solver']['relative_tolerance'] = 1e-15
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
solver_parameters = {}
solver_parameters["linear_solver"] = "gmres"
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["maximum_iterations"] = 15
solver_parameters["newton_solver"]["relaxation_parameter"] = 1.0
# info(parameters, True)
# set_log_active(False)
set_log_level(PROGRESS)

############################################################
# TIME DISCRETISATION FUNCTIONS

class Model():
    
    # SIMULAITON USER DEFINED PARAMETERS

    # mesh
    dX_ = 5.0e-2
    L = 1.0

    # current properties
    c_0 = 0.00349
    rho_R_ = 1.717
    h_0 = 0.4
    x_N_ = 0.2
    Fr_ = 1.19
    g_ = 9.81
    u_sink_ = 1e-3

    # time step
    timestep = 5e-3

    # mms test (default False)
    mms = False

    # display plot
    plot = True

    # smoothing eps value
    eps = 1e-18

    # define stabilisation parameters
    q_su = False
    q_bs = True
    q_b = 0.1
    h_b = 0.0
    phi_b = 0.0
    c_d_b = 0.0

    def initialise_function_spaces(self):

        # define geometry
        self.mesh = IntervalMesh(int(self.L/self.dX_), 0.0, self.L)
        self.n = FacetNormal(self.mesh)

        # left boundary marked as 0, right as 1
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        left_boundary = LeftBoundary()
        exterior_facet_domains = FacetFunction("uint", self.mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains] 

        # define function spaces
        self.q_degree = 1
        self.q_disc = "CG"
        self.q_FS = FunctionSpace(self.mesh, self.q_disc, self.q_degree)
        self.h_degree = 1
        self.h_disc = "CG"
        self.h_FS = FunctionSpace(self.mesh, self.h_disc, self.h_degree)
        self.phi_degree = 1
        self.phi_disc = "CG"
        self.phi_FS = FunctionSpace(self.mesh, self.phi_disc, self.phi_degree)
        self.c_d_degree = 1
        self.c_d_disc = "CG"
        self.c_d_FS = FunctionSpace(self.mesh, self.c_d_disc, self.c_d_degree)
        self.var_N_FS = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.q_FS, self.h_FS, self.phi_FS, self.c_d_FS, self.var_N_FS, self.var_N_FS])
        self.X_FS = FunctionSpace(self.mesh, "CG", 1)

        # get dof_maps for plots
        self.map_dict = dict()
        for i in range(6):
            if len(self.W.sub(i).dofmap().dofs()) == len(self.mesh.cells()) + 1:   # CG 
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j)[0] for j in range(len(self.mesh.cells()))]
                self.map_dict[i].append(self.W.sub(i).dofmap().cell_dofs(len(self.mesh.cells()) - 1)[1])
            elif len(self.W.sub(i).dofmap().dofs()) == len(self.mesh.cells()) * 2:   # DG
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j) for j in range(len(self.mesh.cells()))]
                self.map_dict[i] = list(np.array(self.map_dict[i]).flatten())
            else:   # R
                self.map_dict[i] = self.W.sub(i).dofmap().cell_dofs(0)              

        # define test functions
        (self.q_tf, self.h_tf, self.phi_tf, self.c_d_tf, self.x_N_tf, self.u_N_tf) = TestFunctions(self.W)

        # initialise functions
        X_ = project(Expression('x[0]'), self.X_FS)
        self.X = Function(X_, name='X')

        self.w = dict()
        self.w[0] = Function(self.W, name='U')

    def setup(self, h_ic = None, phi_ic = None, q_ic = None, w_ic = None):

        # define constants
        self.dX = Constant(self.dX_, name="dX")
        self.g = Constant(self.g_, name="g")
        self.rho_R = Constant(self.rho_R_, name="rho_R")
        self.Fr = Constant(self.Fr_, name="Fr")
        self.u_sink = Constant(self.u_sink_, name="u_sink")

        if type(w_ic) == type(None):
            # define initial conditions
            if type(h_ic) == type(None):
                h_ic = Constant(self.h_0)
                h_N = self.h_0
            else:
                h_N = h_ic.vector().array()[-1]
            if type(phi_ic) == type(None): 
                phi_ic = Constant(self.c_0*self.rho_R_*self.g_*self.h_0)
                phi_N = self.c_0*self.rho_R_*self.g_*self.h_0
            else:
                phi_N = phi_ic.vector().array()[-1]

            # set u_N component
            trial = TrialFunction(self.var_N_FS)
            test = TestFunction(self.var_N_FS)
            u_N_ic = Function(self.var_N_FS)
            a = inner(test, trial)*self.ds(1)
            L = inner(test, self.Fr*phi_ic**0.5)*self.ds(1)             
            solve(a == L, u_N_ic)

            if type(q_ic) == type(None):
                trial = TrialFunction(self.q_FS)
                test = TestFunction(self.q_FS)
                q_ic = Function(self.q_FS)
                a = inner(test, trial)*dx
                L = inner(test, (1.0 - cos(self.X*np.pi))/2.0*u_N_ic*h_ic)*dx             
                solve(a == L, q_ic)

            self.w_ic = [
                q_ic, 
                h_ic, 
                phi_ic, 
                Function(self.c_d_FS), 
                Constant(self.x_N_), 
                u_N_ic
                ]
            
        else:
            self.w_ic = w_ic

        # define bc's
        bcc_d = DirichletBC(self.W.sub(3), '0.0', "near(x[0], 1.0) && on_boundary")
        bcq = DirichletBC(self.W.sub(0), '0.0', "near(x[0], 0.0) && on_boundary")
        self.bc = [bcq]#, bcc_d]

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self)

    def generate_form(self):

        # galerkin projection of initial conditions on to w
        test = TestFunction(self.W)
        trial = TrialFunction(self.W)
        
        # set q, h, phi, c_d and x_N components
        L = 0; a = 0
        for i in range(len(self.w_ic)):
            a += inner(test[i], trial[i])*dx
            L += inner(test[i], self.w_ic[i])*dx
        solve(a == L, self.w[0])

        # copy to w[1]
        self.w[1] = project(self.w[0], self.W)

        # smooth functions (also never hit zero)
        def smooth_pos(val):
            return (val + smooth_abs(val))/2.0
        def smooth_neg(val):
            return (val - smooth_abs(val))/2.0
        def smooth_abs(val):
            return (val**2.0 + self.eps)**0.5

        # define 1/dt
        self.k = Constant(self.timestep)

        # time discretisation of values
        def time_discretise(u):
            return 0.5*u[0] + 0.5*u[1]

        q = dict()
        h = dict()
        phi = dict()
        c_d = dict()
        x_N = dict()
        u_N = dict()

        q[0], h[0], phi[0], c_d[0], x_N[0], u_N[0] = split(self.w[0])
        q[1], h[1], phi[1], c_d[1], x_N[1], u_N[1] = split(self.w[1])

        q_td = time_discretise(q)
        # h_td = smooth_pos(time_discretise(h))
        h_td = time_discretise(h)
        h_td_p = smooth_pos(time_discretise(h))
        phi_td = time_discretise(phi)
        c_d_td = time_discretise(c_d)
        x_N_td = time_discretise(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = time_discretise(u_N)
        ux = Constant(-1.0)*u_N_td*self.X

        uxn_up = smooth_pos(ux*self.n)
        uxn_down = smooth_neg(ux*self.n)

        def transForm(u, v, index, disc, stab, weak_b):
            F = self.k*grad(v)*ux*u*dx - \
                self.k*v*grad(ux)*u*dx 
            if disc == "CG":
                F += self.k*v*self.n*ux*u*weak_b
                if stab > 0.0:
                    tau = Constant(stab)*self.dX/smooth_pos(ux)
                    F += tau*inner(ux, grad(v))*inner(ux, grad(u))*self.k*dx - \
                        tau*inner(ux, self.n*v)*inner(ux, grad(u))*self.k*weak_b
            elif disc == "DG":
                F += avg(self.k)*jump(v)*(uxn_up('+')*u('+') - uxn_up('-')*u('-'))*dS 
                if self.mms:
                    F += self.k*v*uxn_down*self.w_ic[index]*(self.ds(0) + self.ds(1))
                else:
                    F += self.k*v*uxn_down*u*weak_b
            else:
                sys.exit("unknown element type for function index {}".format(index))

            return F

        # momentum
        q_N = u_N_td*h_td
        F_q = x_N_td*self.q_tf*(q[0] - q[1])*dx + \
            self.k*self.q_tf*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)*dx + \
            self.k*u_N_td*grad(self.q_tf*self.X)*q_td*dx - \
            self.k*u_N_td*self.q_tf*self.X*q_N*self.n*self.ds(1) 
        if self.q_bs:
            u = q_td/h_td
            alpha = self.q_b*self.dX*(smooth_abs(u)+u+(phi_td*h_td)**0.5)*h_td
            F_q += self.k*grad(self.q_tf)*alpha*grad(u)*dx - \
                self.k*self.q_tf*alpha*grad(u)*self.n*self.ds(1)  
        if self.mms:
            F_q += x_N_td*self.q_tf*self.s_q*self.k*dx

        # conservation
        F_h = x_N_td*self.h_tf*(h[0] - h[1])*dx + \
            self.k*self.h_tf*grad(q_td)*dx
        F_h += transForm(h_td, self.h_tf, 1, self.h_disc, self.h_b, self.ds(0) + self.ds(1))
        if self.mms:
            F_h += x_N_td*self.h_tf*self.s_h*self.k*dx

        # concentration
        F_phi = x_N_td*self.phi_tf*(phi[0] - phi[1])*dx + \
            self.phi_tf*grad(q_td*phi_td/h_td)*self.k*dx + \
            x_N_td*self.phi_tf*self.u_sink*phi_td/h_td*self.k*dx - \
            self.phi_tf*self.X*u_N_td*grad(phi_td)*self.k*dx
        if self.mms:
            F_phi += x_N_td*self.phi_tf*self.s_phi*self.k*dx

        # deposit
        F_c_d = x_N_td*self.c_d_tf*(c_d[0] - c_d[1])*dx - \
            x_N_td*self.c_d_tf*self.u_sink*phi_td/(self.rho_R*self.g*h_td)*self.k*dx 
        F_c_d += transForm(c_d_td, self.c_d_tf, 3, self.c_d_disc, self.c_d_b, self.ds(0))
        if self.mms:
            F_c_d = F_c_d + x_N_td*self.c_d_tf*self.s_c_d*self.k*dx

        # nose location/speed
        if self.mms:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx - self.x_N_tf*Constant(0.0)*self.k*dx
        else:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx - self.x_N_tf*u_N_td*self.k*dx 
        F_u_N = self.u_N_tf*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            self.u_N_tf*u_N[0]*self.ds(1)

        # combine PDE's
        self.F = F_q + F_h + F_phi + F_c_d + F_x_N + F_u_N

        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0], trial)

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
        delta = 1e10

        while not (time_finish(self.t) or converged(delta)):
            
            # SOLVE COUPLED EQUATIONS
            solve(self.F == 0, self.w[0], bcs=self.bc, J=self.J, solver_parameters=solver_parameters)
            
            delta = 0.0
            f_list = [[self.w[0].split()[i], self.w[1].split()[i]] for i in range(len(self.w[0].split()))]
            for f_0, f_1 in f_list:
                delta = max(errornorm(f_0, f_1, norm_type="L2", degree_rise=1)/self.timestep, delta)

            self.w[1].assign(self.w[0])

            self.t += self.timestep
            
            # ADAPTIVE TIMESTEP
            # timestep = ...
            # self.k.assign(self.timestep)

            # plot(self.w[0][1], interactive = False, rescale = True)

            # display results
            if self.plot:
                self.plotter.update_plot(self)
            sw_io.print_timestep_info(self, delta)

            # q, h, phi, c_d, x_N, u_N = sw_io.map_to_arrays(self)
            # import IPython
            # IPython.embed()

        list_timings(True)

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
    parser.add_option('-p', '--plot',
                      action='store_true', dest='plot', default=False,
                      help='plot results in real-time')
    (options, args) = parser.parse_args()
    
    model = Model()
    model.plot = options.plot
    model.initialise_function_spaces()

    # Adjoint 
    if options.adjoint == True:

        phi_ic = project(Expression('0.01'), model.phi_FS)
        model.setup(phi_ic = phi_ic)
        model.solve(T = options.T)

        # get model data
        c_d_aim = sw_io.create_function_from_file('deposit_data.json', model.c_d_FS)
        x_N_aim = sw_io.create_function_from_file('runout_data.json', model.var_N_FS)
        (q, h, phi, c_d, x_N, u_N) = split(model.w[0])

        # form Functional integrals
        int_0_scale = Constant(1)
        int_1_scale = Constant(1)
        int_0 = inner(c_d-c_d_aim, c_d-c_d_aim)*int_0_scale*dx
        int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*int_1_scale*dx

        # determine scalaing
        int_0_scale.assign(1e-2/assemble(int_0))
        int_1_scale.assign(1e-4/assemble(int_1))
        print assemble(int_0)
        print assemble(int_1)
        ### int_0 1e-2, int_1 1e-4 - worked well
        
        ## functional regularisation
        reg_scale = Constant(1)
        int_reg = inner(grad(phi), grad(phi))*reg_scale*dx
        reg_scale_base = 1e-3
        reg_scale.assign(reg_scale_base)

        ## functional
        J = Functional((int_0 + int_1)*dt[FINISH_TIME] + int_reg*dt[START_TIME])

        tic()

        ## compute gradient information
        dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic))

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(dJdphi.vector().array())
        # plt.show()

        # import IPython
        # IPython.embed()

        # clear old data
        sw_io.clear_file('phi_ic_adj.json')
        j_log = []

        ##############################
        #### REDUCED FUNCTIONAL HACK
        ##############################

        class MyReducedFunctional(ReducedFunctional):

            def __call__(self, value):

                print "\n* * * Adjoint and optimiser time taken = {}".format(toc())
                list_timings(True)

                #### initial condition dump hack ####
                ic = value[0].vector().array()
                sw_io.write_array_to_file('phi_ic_adj_latest.json',ic,'w')
                sw_io.write_array_to_file('phi_ic_adj.json',ic,'a')

                tic()

                print "\n* * * Computing forward model"

                func_value = (super(MyReducedFunctional, self)).__call__(value)

                print "* * * Completed forward model"
                print "* * * Time Taken = {}".format(toc())
                
                list_timings(True)

                j = self.scale * func_value
                j_log.append(j)
                sw_io.write_array_to_file('j_log.json', j_log, 'w')

                tic()

                return func_value                

        #######################################
        #### END OF REDUCED FUNCTIONAL HACK
        #######################################

        reduced_functional = MyReducedFunctional(J, InitialConditionParameter(phi_ic),
                                                 scale = 1e-0)
        
        for i in range(15):
            reg_scale.assign(reg_scale_base*2**(0-i))

            m_opt = minimize(reduced_functional, method = "L-BFGS-B", options = {'maxiter': 4, 'disp': True, 'gtol': 1e-9, 'ftol': 1e-9}, 
                             bounds = (1e-7, 1e-0))

        print m_opt.vector().array()

    # Adjoint 
    elif options.phi_ic_test == True:

        f = open('phi_ic_adj_latest.json','r')
        json_string = f.readline()
        data = np.array(json.loads(json_string))
        phi_ic = Function(model.phi_FS)
        phi_ic.vector()[:] = data
        
        model.setup(phi_ic = phi_ic)

        model.solve(T = options.T)    

    else:        

        phi_ic = project(Expression('0.03+0.005*sin(2*pi*x[0])'), model.phi_FS)
        model.setup(phi_ic=phi_ic)

        q, h, phi, c_d, x_N, u_N = sw_io.map_to_arrays(model)
        sw_io.write_array_to_file('phi_ic.json', phi, 'w')

        model.solve(T = options.T)

        q, h, phi, c_d, x_N, u_N = sw_io.map_to_arrays(model)
        sw_io.write_array_to_file('deposit_data.json', c_d, 'w')
        sw_io.write_array_to_file('runout_data.json', x_N, 'w')
