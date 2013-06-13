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
    h_disc = "CG"
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
        self.q_FS = FunctionSpace(self.mesh, "CG", self.q_degree)
        self.h_FS = FunctionSpace(self.mesh, self.h_disc, self.h_degree)
        self.phi_FS = FunctionSpace(self.mesh, "CG", self.phi_degree)
        self.phi_d_FS = FunctionSpace(self.mesh, self.phi_d_disc, self.phi_d_degree)
        self.var_N_FS = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.q_FS, self.h_FS, self.phi_FS, self.phi_d_FS, self.var_N_FS, self.var_N_FS])
        self.X_FS = FunctionSpace(self.mesh, "CG", 1)

        # get dof_maps for plots
        self.map_dict = dict()
        for i in range(6):
            if self.W.sub(i).dofmap().global_dimension() == len(self.mesh.cells()) + 1:   # P1CG 
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j)[0] for j in range(len(self.mesh.cells()))]
                self.map_dict[i].append(self.W.sub(i).dofmap().cell_dofs(len(self.mesh.cells()) - 1)[-1])
            elif self.W.sub(i).dofmap().global_dimension() == len(self.mesh.cells()) * 2 + 1:   # P2CG 
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j)[:-1] for j in range(len(self.mesh.cells()))]
                self.map_dict[i] = list(np.array(self.map_dict[i]).flatten())
                self.map_dict[i].append(self.W.sub(i).dofmap().cell_dofs(len(self.mesh.cells()) - 1)[-1])    
            elif self.W.sub(i).dofmap().global_dimension() == len(self.mesh.cells()) * 2:   # P1DG
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j) for j in range(len(self.mesh.cells()))]
                self.map_dict[i] = list(np.array(self.map_dict[i]).flatten())
            else:   # R
                self.map_dict[i] = self.W.sub(i).dofmap().cell_dofs(0)    

        # define test functions
        (self.q_tf, self.h_tf, self.phi_tf, self.phi_d_tf, self.x_N_tf, self.u_N_tf) = TestFunctions(self.W)

        # initialise functions
        X_ = project(Expression('x[0]'), self.X_FS)
        self.X = Function(X_, name='X')

        self.w = dict()
        self.w[0] = Function(self.W, name='U')

    def setup(self, h_ic = None, phi_ic = None, 
              q_a = Constant(0.0), q_pa = Constant(0.0), q_pb = Constant(1.0), 
              w_ic = None, zero_q = False):
        # q_a between 0.0 and 1.0 
        # q_pa between 0.2 and 0.99 
        # q_pb between 1.0 and 

        # set current time to 0.0
        self.t = 0.0

        # define constants
        self.Fr = Constant(self.Fr_, name="Fr")
        self.beta = Constant(self.beta_, name="beta")

        if type(w_ic) == type(None):
            # define initial conditions
            if type(h_ic) == type(None):
                h_ic = 1.0 
                h_N = 1.0 
            else:
                h_N = h_ic.vector().array()[-1]
            if type(phi_ic) == type(None): 
                phi_ic = 1.0 
                phi_N = 1.0 
            else:
                phi_N = phi_ic.vector().array()[-1]

            # calculate u_N component
            trial = TrialFunction(self.var_N_FS)
            test = TestFunction(self.var_N_FS)
            u_N_ic = Function(self.var_N_FS, name='u_N_ic')
            a = inner(test, trial)*self.ds(1)
            L = inner(test, self.Fr*phi_ic**0.5)*self.ds(1)             
            solve(a == L, u_N_ic)

            # define q
            q_N_ic = Function(self.var_N_FS, name='q_N_ic')
            q_ic = Function(self.q_FS, name='q_ic')

            # cosine initial condition for u
            if not zero_q:
                a = inner(test, trial)*self.ds(1)
                L = inner(test, u_N_ic*h_ic)*self.ds(1)             
                solve(a == L, q_N_ic)

                trial = TrialFunction(self.q_FS)
                test = TestFunction(self.q_FS)
                a = inner(test, trial)*dx
                q_b = Constant(1.0) - q_a  
                f = (1.0 - (q_a*cos(((self.X/self.L)**q_pa)*np.pi) + q_b*cos(((self.X/self.L)**q_pb)*np.pi)))/2.0
                L = inner(test, f*q_N_ic)*dx             
                solve(a == L, q_ic)

            # create ic array
            self.w_ic = [
                q_ic, 
                h_ic, 
                phi_ic, 
                Function(self.phi_d_FS, name='phi_d_ic'), 
                self.x_N_, 
                u_N_ic
                ]
            
        else:
            # whole of w_ic defined externally
            self.w_ic = w_ic

        # define bc's
        bcphi_d = DirichletBC(self.W.sub(3), '0.0', "near(x[0], 1.0) && on_boundary")
        bcq = DirichletBC(self.W.sub(0), '0.0', "near(x[0], 0.0) && on_boundary")
        self.bc = [bcq] #, bcphi_d]

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self, rescale=True, file=self.save_loc)
            self.plot_t = self.plot

        # write ic's
        if self.write:
            sw_io.clear_model_files(file=self.save_loc)
            sw_io.write_model_to_files(self, 'a', file=self.save_loc)
            self.write_t = self.write

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
            return 0.5*u[0] + 0.5*u[1]

        q = dict()
        h = dict()
        phi = dict()
        phi_d = dict()
        x_N = dict()
        u_N = dict()

        q[0], h[0], phi[0], phi_d[0], x_N[0], u_N[0] = split(self.w[0])
        q[1], h[1], phi[1], phi_d[1], x_N[1], u_N[1] = split(self.w[1])

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
        # h_td = smooth_pos(time_discretise(h))
        h_td = time_discretise(h)
        h_td_p = smooth_pos(time_discretise(h))
        phi_td = time_discretise(phi)
        phi_td_p = smooth_pos(phi_td)
        phi_d_td = time_discretise(phi_d)
        x_N_td = time_discretise(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = time_discretise(u_N)
        ux = Constant(-1.0)*u_N_td*self.X

        uxn_up = smooth_pos(ux*self.n)
        uxn_down = smooth_neg(ux*self.n)

        def transForm(u, v, index, disc, stab, weak_b, b_val = None):
            if type(b_val) == type(None):
                b_val = u

            F = - self.k*grad(v)[0]*ux*u*dx - self.k*v*grad(ux)[0]*u*dx 
            
            if disc == "CG":
                F += self.k*v*self.n*ux*b_val*weak_b
                if stab((0,0)) > 0.0:
                    tau = Constant(stab)*self.dX/smooth_abs(ux)
                    F += tau*ux*grad(v)[0]*ux*grad(u)[0]*self.k*dx - \
                        tau*ux*self.n*v*ux*grad(b_val)[0]*self.k*weak_b
            elif disc == "DG":
                F += avg(self.k)*jump(v)*(uxn_up('+')*u('+') - uxn_up('-')*u('-'))*dS 
                if self.mms:
                    F += self.k*v*uxn_down*self.w_ic[index]*(self.ds(0) + self.ds(1))
                else:
                    F += self.k*v*uxn_down*b_val*weak_b
            else:
                sys.exit("unknown element type for function index {}".format(index))

            return F

        # momentum 
        F_q = x_N_td*self.q_tf*(q[0] - q[1])*dx + \
            self.k*self.q_tf*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)[0]*dx
        F_q += transForm(q_td, self.q_tf, 0, "CG", Constant(0.0), self.ds(1), u_N_td*h_td)
        # stabilisation
        u = q_td/h_td
        alpha = self.q_b*self.dX*(smooth_abs(u)+u+(phi_td_p*h_td_p)**0.5)*h_td
        F_q += self.k*grad(self.q_tf)[0]*alpha*grad(u)[0]*dx - \
            self.k*self.q_tf*alpha*grad(u)[0]*self.n*self.ds(1)  
        if self.mms:
            F_q += x_N_td*self.q_tf*self.s_q*self.k*dx

        # conservation 
        F_h = x_N_td*self.h_tf*(h[0] - h[1])*dx + \
            self.k*self.h_tf*grad(q_td)[0]*dx
        F_h += transForm(h_td, self.h_tf, 1, self.h_disc, self.h_b, self.ds(0) + self.ds(1))
        if self.mms:
            F_h += x_N_td*self.h_tf*self.s_h*self.k*dx

        # mass volume per unit width of sediment = g'(phi)*h = R*g*c*h
        F_phi = x_N_td*self.phi_tf*(phi[0] - phi[1])*dx + \
            self.phi_tf*grad(q_td*phi_td/h_td)[0]*self.k*dx + \
            x_N_td*self.phi_tf*self.beta*phi_td/h_td*self.k*dx
        F_phi += transForm(phi_td, self.phi_tf, 2, "CG", self.phi_b, self.ds(0) + self.ds(1))
        if self.mms:
            F_phi += x_N_td*self.phi_tf*self.s_phi*self.k*dx

        # deposit = c
        F_phi_d = x_N_td*self.phi_d_tf*(phi_d[0] - phi_d[1])*dx - \
            x_N_td*self.phi_d_tf*self.beta*phi_td/h_td*self.k*dx 
        F_phi_d += transForm(phi_d_td, self.phi_d_tf, 3, self.phi_d_disc, self.phi_d_b, self.ds(0))
        if self.mms:
            F_phi_d += x_N_td*self.phi_d_tf*self.s_phi_d*self.k*dx

        # nose location/speed
        if self.mms:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx 
        else:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx - self.x_N_tf*u_N_td*self.k*dx 
        F_u_N = self.u_N_tf*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            self.u_N_tf*u_N[0]*self.ds(1)
        # F_u_N = self.u_N_tf*(0.5*h_td**-0.5*(phi_td)**0.5)*self.ds(1) - \
        #     self.u_N_tf*u_N[0]*self.ds(1)

        # combine PDE's
        self.F = F_q + F_h + F_phi + F_phi_d + F_x_N + F_u_N
        # self.F = F_h

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
