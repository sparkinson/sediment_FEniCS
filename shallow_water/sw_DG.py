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
parameters['krylov_solver']['absolute_tolerance'] = 1e-20
parameters['krylov_solver']['relative_tolerance'] = 1e-20
# parameters['krylov_solver']['maximum_iterations'] = 1000000
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
solver_parameters = {}
# solver_parameters["linear_solver"] = "gmres"
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["maximum_iterations"] = 15
solver_parameters["newton_solver"]["relaxation_parameter"] = 1.0
# solver_parameters["newton_solver"]["absolute_tolerance"] = 1E-13
# solver_parameters["newton_solver"]["relative_tolerance"] = 1E-13
# print solver_parameters["newton_solver"].keys()
# info(parameters, True)
# set_log_active(False)
set_log_level(ERROR)

############################################################
# TIME DISCRETISATION FUNCTIONS

class Model():
    
    # SIMULAITON USER DEFINED PARAMETERS

    # mesh
    dX_ = 5.0e-2
    L_ = 1.0

    # current properties
    x_N_ = 0.5
    Fr_ = 1.19
    beta_ = 5e-3

    # time stepping
    t = 0.0
    timestep = dX_/10.0
    adapt_timestep = True
    adapt_initial_timestep = True
    cfl = Constant(1.0)

    # mms test (default False)
    mms = False

    # display plot
    plot = None
    show_plot = True
    save_plot = False

    # output data
    write = None

    # save location
    save_loc = 'results/'

    # smoothing eps value
    eps = 1e-12

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    q_b = Constant(0.2)
    h_b = Constant(0.0)
    phi_b = Constant(0.0)
    phi_d_b = Constant(0.0)

    # discretisation
    q_degree = 2
    h_degree = 1
    phi_degree = 1
    phi_d_degree = 1
    q_disc = "CG"
    h_disc = "CG"
    phi_disc = "CG"
    phi_d_disc = "CG"

    def initialise_function_spaces(self):

        # define geometry
        self.mesh = IntervalMesh(int(self.L_/self.dX_), 0.0, self.L_)
        self.n = FacetNormal(self.mesh)[0]

        self.dX = Constant(self.dX_)
        self.L = Constant(self.L_)

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
        self.q_FS = FunctionSpace(self.mesh, self.q_disc, self.q_degree)
        self.h_FS = FunctionSpace(self.mesh, self.h_disc, self.h_degree)
        self.phi_FS = FunctionSpace(self.mesh, self.phi_disc, self.phi_degree)
        self.phi_d_FS = FunctionSpace(self.mesh, self.phi_d_disc, self.phi_d_degree)
        self.cg_FS = FunctionSpace(self.mesh, "CG", 1)
        self.var_N_FS = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.q_FS, self.cg_FS, 
                                     self.h_FS, self.cg_FS, 
                                     self.phi_FS, self.cg_FS, 
                                     self.phi_d_FS, 
                                     self.var_N_FS, 
                                     self.var_N_FS])
        self.X_FS = FunctionSpace(self.mesh, "CG", 1)

        # generate dof_map
        sw_io.generate_dof_map(self)

        # define test functions
        (self.q_tf, self.q_cg_tf, 
         self.h_tf, self.h_cg_tf, 
         self.phi_tf, self.phi_cg_tf, 
         self.phi_d_tf, 
         self.x_N_tf, self.u_N_tf) = TestFunctions(self.W)

        # initialise functions
        X_ = project(Expression('x[0]'), self.X_FS)
        self.X = Function(X_, name='X')
        self.w = dict()
        self.w[0] = Function(self.W, name='U')

    def generate_form(self):

        # galerkin projection of initial conditions on to w
        test = TestFunction(self.W)
        trial = TrialFunction(self.W)
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

        # time discretisation of values
        def time_discretise(u):
            if not self.mms:
                return 0.5*u[0] + 0.5*u[1]
            else:
                return u[0]

        q = dict()
        q_cg = dict()
        h = dict()
        h_cg = dict()
        phi = dict()
        phi_cg = dict()
        phi_d = dict()
        x_N = dict()
        u_N = dict()

        q[0], q_cg[0], h[0], h_cg[0], phi[0], phi_cg[0], phi_d[0], x_N[0], u_N[0] = split(self.w[0])
        q[1], q_cg[1], h[1], h_cg[1], phi[1], phi_cg[1], phi_d[1], x_N[1], u_N[1] = split(self.w[1])

        # define adaptive timestep form
        if self.adapt_timestep:
            self.k = project(Expression(str(self.timestep)), self.var_N_FS)
            self.k_tf = TestFunction(self.var_N_FS)
            self.k_trf = TrialFunction(self.var_N_FS)
            self.a_k = self.k_tf*self.k_trf*dx 
            self.L_k = self.k_tf*(x_N[0]*self.dX)/(self.L*u_N[0])*self.cfl*dx
        else:
            self.k = Constant(self.timestep)

        q_td = time_discretise(q)
        q_cg_td = time_discretise(q_cg)
        # h_td = smooth_pos(time_discretise(h))
        h_td = time_discretise(h)
        h_cg_td = time_discretise(h_cg)
        h_td_p = smooth_pos(time_discretise(h))
        h_cg_td_p = smooth_pos(time_discretise(h_cg))
        phi_td = time_discretise(phi)
        phi_cg_td = time_discretise(phi_cg)
        phi_td_p = smooth_pos(phi_td)
        phi_cg_td_p = smooth_pos(time_discretise(phi_cg))
        phi_d_td = time_discretise(phi_d)
        x_N_td = time_discretise(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = time_discretise(u_N)
        ux = Constant(-1.0)*u_N_td*self.X

        uxn_up = smooth_pos(ux*self.n)
        uxn_down = smooth_neg(ux*self.n)

        # mass term, coordinate transformation, source, su and weak bc's
        def transForm(u, u_td, v, index, disc, stab, weak_b = None):

            # coordinate transforming advection term
            # F = self.k*v*ux*grad(u_td)[0]*dx
            # integrated by parts
            # F = - self.k*grad(v*ux)[0]*u_td*dx
            F = - self.k*grad(v)[0]*ux*u_td*dx - self.k*v*grad(ux)[0]*u_td*dx

            # mass term
            if not self.mms:
                F += x_N_td*v*(u[0] - u[1])*dx
            # source term
            if self.mms:
                F += x_N_td*v*self.S[index]*self.k*dx
            
            if disc == "CG":

                # boundary integral for coordinate transforming advection term
                F += self.k*v*ux*u_td*self.n*(self.ds(0) + self.ds(1))

                # bc
                if self.mms:
                    F -= self.k*v*u_td*self.n*(self.ds(0) + self.ds(1))
                    F += self.k*v*self.w_ic[index]*self.n*(self.ds(0) + self.ds(1)) 
                elif b_val:
                    F -= self.k*v*u_td*self.n*weak_b[1]
                    F += self.k*v*weak_b[0]*self.n*weak_b[1]

                # stabilisation
                if stab((0,0)) > 0.0:
                    tau = Constant(stab)*self.dX/smooth_abs(ux)
                    F += tau*ux*grad(v)[0]*ux*grad(u_td)[0]*self.k*dx - \
                        tau*ux*self.n*v*ux*grad(weak_b[0])[0]*self.k*weak_b[1]

            elif disc == "DG":

                # interior interface jump terms for coordinate transforming advection term
                F += avg(self.k)*jump(v)*(uxn_up('+')*u_td('+') - uxn_up('-')*u_td('-'))*dS 

                # bc
                if self.mms:
                    F += self.k*v*uxn_down*self.w_ic[index]*(self.ds(0) + self.ds(1))
                else:
                    F += self.k*v*uxn_down*b_val*weak_b

            else:
                sys.exit("unknown element type for function index {}".format(index))

            return F

        # MOMENTUM 
        F_q = self.k*self.q_tf*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)[0]*dx
        # # integrated by parts
        # F_q = - self.k*grad(self.q_tf)[0]*(q_td**2.0/h_td + 0.5*phi_td*h_td)*dx
        # if self.h_disc == "CG":
        #     F_q += self.k*self.q_tf*(self.w_ic[0]**2.0/self.w_ic[1] + 0.5*self.w_ic[2]*self.w_ic[1])*(self.ds(0) + self.ds(1))
        #     # F_q += self.k*self.q_tf*(q_td**2.0/h_td + 0.5*phi_td*h_td)*(self.ds(0) + self.ds(1))
        # # BONNECAZE STABILISATION
        # u = q_td/h_td
        # alpha = self.q_b*self.dX*(smooth_abs(u)+u+(phi_td_p*h_td_p)**0.5)*h_td
        # F_q += self.k*grad(self.q_tf)[0]*alpha*grad(u)[0]*dx - \
        #     self.k*self.q_tf*alpha*grad(u)[0]*self.n*(self.ds(0) + self.ds(1))  
        # F_q += self.k*grad(self.q_tf)[0]*alpha*grad(u)[0]*dx - \
        #     self.k*self.q_tf*alpha*grad(self.w_ic[0]/self.w_ic[1])[0]*self.n*(self.ds(0) + self.ds(1)) 
        F_q = transForm(q, q_td, self.q_tf, 0, self.q_disc, Constant(0.0), weak_b = [u_N_td*h_td, self.ds(1)])

        # MOMENTUM PROJECTION
        F_cg_q = self.q_cg_tf*q_cg[0]*dx - self.q_cg_tf*q[0]*dx

        # F_q = self.q_tf*(q[0] - q[1])*dx 

        # CONSERVATION 
        F_h = self.k*self.h_tf*grad(q_td)[0]*dx
        F_h += transForm(h, h_td, self.h_tf, 2, self.h_disc, self.h_b)

        # CONSERVATION PROJECTION
        F_cg_h = self.h_cg_tf*h_cg[0]*dx - self.h_cg_tf*h[0]*dx

        # F_h = self.q_tf*(q[0] - q[1])*dx 

        # VOLUME FRACTION
        F_phi = self.phi_tf*grad(q_td*phi_td/h_cg_td)[0]*self.k*dx + \
            x_N_td*self.phi_tf*self.beta*phi_td/h_td*self.k*dx
        F_phi += transForm(phi, phi_td, self.phi_tf, 4, self.phi_disc, self.phi_b)

        # VOLUME FRACTION PROJECTION
        F_cg_phi = self.phi_cg_tf*phi_cg[0]*dx - self.phi_cg_tf*phi[0]*dx

        F_phi = self.phi_tf*(phi[0] - phi[1])*dx 

        # DEPOSIT
        F_phi_d = -1.0*x_N_td*self.phi_d_tf*self.beta*phi_td/h_td*self.k*dx 
        F_phi_d += transForm(phi_d, phi_d_td, self.phi_d_tf, 6, self.phi_d_disc, self.phi_d_b)

        # F_phi_d = self.phi_d_tf*(phi_d[0] - phi_d[1])*dx 

        # NOSE LOCATION AND SPEED
        if self.mms:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx 
        else:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx - self.x_N_tf*u_N_td*self.k*dx 
        F_u_N = self.u_N_tf*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            self.u_N_tf*u_N[0]*self.ds(1)
        # F_u_N = self.u_N_tf*(0.5*h_td**-0.5*(phi_td)**0.5)*self.ds(1) - \
        #     self.u_N_tf*u_N[0]*self.ds(1)

        # combine PDE's
        self.F = F_q + F_cg_q + F_h + F_cg_h + F_phi + F_cg_phi + F_phi_d + F_x_N + F_u_N
        # self.F = F_h

        # try:
        #     print self.A - self.F
        #     print 'done'
        # except:
        #     self.A = self.F

        # print self.F

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

        tic()

        delta = 1e10
        while not (time_finish(self.t) or converged(delta)):
            
            # ADAPTIVE TIMESTEP
            if self.adapt_timestep and (self.t > 0.0 or self.adapt_initial_timestep):
                solve(self.a_k == self.L_k, self.k)
                self.timestep = self.k.vector().array()[0]
            
            # SOLVE COUPLED EQUATIONS
            # solve(self.F == 0, self.w[0], bcs=self.bc, solver_parameters=solver_parameters)

            solve(self.F == 0, self.w[0], bcs=self.bc, J=self.J, solver_parameters=solver_parameters)
            
            if tol:
                delta = 0.0
                f_list = [[self.w[0].split()[i], self.w[1].split()[i]] for i in range(len(self.w[0].split()))]
                for f_0, f_1 in f_list:
                    delta = max(errornorm(f_0, f_1, norm_type="L2", degree_rise=1)/self.timestep, delta)

            self.w[1].assign(self.w[0])

            self.t += self.timestep

            # display results
            if self.plot:
                if self.t > self.plot_t:
                    self.plotter.update_plot(self)
                    self.plot_t += self.plot

            # save data
            if self.write:
                if self.t > self.write_t:
                    sw_io.write_model_to_files(self, 'a', file=self.save_loc)
                    self.write_t += self.write

            # write timestep info
            sw_io.print_timestep_info(self, delta)

        print "\n* * * Initial forward run finished: time taken = {}".format(toc())
        list_timings(True)

        if self.plot:
            self.plotter.clean_up()

if __name__ == '__main__':

    model = Model()    
    model.plot = 0.05
    model.initialise_function_spaces()
    model.setup(zero_q = False)     
    model.solve(60.0) 
