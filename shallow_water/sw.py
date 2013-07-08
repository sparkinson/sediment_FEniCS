#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
from dolfin_adjoint import *
import numpy as np
from optparse import OptionParser
import json
import sw_io
import scipy

############################################################
# DOLFIN SETTINGS

parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
# parameters['krylov_solver']['relative_tolerance'] = 1e-15
# parameters['krylov_solver']['maximum_iterations'] = 1000000
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
solver_parameters = {}
solver_parameters["linear_solver"] = "lu"
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["maximum_iterations"] = 15
solver_parameters["newton_solver"]["relaxation_parameter"] = 1.0
info(parameters, True)
# set_log_active(False)
set_log_level(ERROR)

# smooth functions (also never hit zero)
def smooth_pos(val):
    return (val + smooth_abs(val))/2.0
def smooth_neg(val):
    return (val - smooth_abs(val))/2.0
def smooth_abs(val):
    # smoothing eps value
    eps = 1e-12
    return (val**2.0 + eps)**0.5

# time discretisations
def explicit(object, u):
    return u[1]
def implicit(object, u):
    return u[0]
def runge_kutta(object, u):
    return u[1]
def crank_nicholson(object, u):
    return 0.5*u[0] + 0.5*u[1]

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
    timestep = dX_/100.0
    adapt_timestep = False
    adapt_initial_timestep = True
    cfl = Constant(0.2)

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

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    q_b = Constant(0.0)
    h_b = Constant(0.0)
    phi_b = Constant(0.0)
    phi_d_b = Constant(0.0)

    # discretisation
    q_degree = 1
    h_degree = 1
    phi_degree = 1
    phi_d_degree = 1
    q_disc = "DG"
    h_disc = "DG"
    phi_disc = "DG"
    phi_d_disc = "DG"
    disc = [q_disc, h_disc, phi_disc, phi_d_disc]
    time_discretise = crank_nicholson #implicit #crank_nicholson
    slope_limiter = 'vb'

    # error calculation
    error_callback = None

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
        self.var_N_FS = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.q_FS, self.h_FS, self.phi_FS, self.phi_d_FS, self.var_N_FS, self.var_N_FS])
        self.X_FS = FunctionSpace(self.mesh, "CG", 1)

        # generate dof_map
        sw_io.generate_dof_map(self)

        # define test functions
        (self.q_tf, self.h_tf, self.phi_tf, self.phi_d_tf, self.x_N_tf, self.u_N_tf) = TestFunctions(self.W)
        self.v = TestFunction(self.W)

        # initialise functions
        X_ = project(Expression('x[0]'), self.X_FS)
        self.X = Function(X_, name='X')
        self.w = dict()
        self.w[0] = Function(self.W, name='U')

    def setup(self, h_ic = None, phi_ic = None, 
              q_a = Constant(0.0), q_pa = Constant(0.0), q_pb = Constant(1.0), 
              w_ic = None, zero_q = False, similarity = False):
        # q_a between 0.0 and 1.0 
        # q_pa between 0.2 and 0.99 
        # q_pb between 1.0 and 

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
        self.bc = [bcq, bcphi_d]

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self, rescale=True, file=self.save_loc, similarity = similarity)
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
        # copy to w[2] and w[3] - for intermedaite values in RK scheme
        if self.time_discretise.im_func == runge_kutta:
            self.w[2] = project(self.w[0], self.W)

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

        q_td = self.time_discretise(q)
        # h_td = smooth_pos(self.time_discretise(h))
        h_td = self.time_discretise(h)
        h_td_p = smooth_pos(self.time_discretise(h))
        phi_td = self.time_discretise(phi)
        phi_td_p = smooth_pos(phi_td)
        phi_d_td = self.time_discretise(phi_d)
        x_N_td = self.time_discretise(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = self.time_discretise(u_N)
        ux = Constant(-1.0)*u_N_td*self.X

        uxn_up = smooth_pos(ux*self.n)
        uxn_down = smooth_neg(ux*self.n)

        # mass term, coordinate transformation, source, su and weak bc's
        def createForm(u, u_td, v, index, disc, stab, weak_b = 0, grad_term = None, settling = None, enable = True):

            if enable:
            # coordinate transforming advection term
                F = - self.k*grad(v)[0]*ux*u_td*dx - self.k*v*grad(ux)[0]*u_td*dx

                if disc == "DG":
                    ux_n = uxn_down
                else:
                    ux_n = ux*self.n

                # surface integrals for coordinate transforming advection term
                F += avg(self.k)*jump(v)*(uxn_up('+')*u_td('+') - uxn_up('-')*u_td('-'))*dS 
                if self.mms or not weak_b:
                    F += self.k*v*ux_n*u_td*(self.ds(0) + self.ds(1))
                else:
                    for weak_b_ in weak_b:
                        F += self.k*v*ux_n*weak_b_[0]*self.n*weak_b_[1]

                # mass term
                if not self.mms:
                    F += x_N_td*v*(u[0] - u[1])*dx
                # F += x_N_td*v*(u[0] - u[1])*dx
                # source term
                if self.mms:
                    F += x_N_td*v*self.S[index]*self.k*dx

                # mms bc
                if self.mms:
                    F -= self.k*v*u_td*self.n*(self.ds(0) + self.ds(1))
                    F += self.k*v*self.w_ic[index]*self.n*(self.ds(0) + self.ds(1)) 
                if index == 0 and not self.mms:
                    F -= self.k*v*u_td*self.n*self.ds(0)

                # stabilisation - untested
                if stab((0,0)) > 0.0:
                    if index > 0:
                        tau = stab*self.dX/smooth_abs(ux)
                        F += self.k*tau*ux*grad(v)[0]*ux*grad(u_td)[0]*dx - \
                            self.k*tau*ux*v*ux*grad(u_td)[0]*self.n*(self.ds(0) + self.ds(1))
                    else:
                        u = q_td/h_td
                        tau = stab*self.dX*(smooth_abs(u)+u+(phi_td_p*h_td_p)**0.5)*h_td
                        F += self.k*grad(v)[0]*tau*grad(u)[0]*dx
                        F -= self.k*v*tau*grad(u)[0]*self.n*(self.ds(0) + self.ds(1)) 

                if grad_term:
                    F -= self.k*grad(v)[0]*grad_term*dx
                    F += avg(self.k)*jump(v)*avg(grad_term)*self.n('+')*dS
                    F += self.k*v*grad_term*self.n*(self.ds(0) + self.ds(1))

                if settling:
                    F += self.k*x_N_td*v*settling*dx

            else:
                F = v*(u[0] - u[1])*dx

            return F

        # MOMENTUM 
        F_q =      createForm(q, q_td, self.q_tf, 0, 
                              self.q_disc, self.q_b, 
                              weak_b = ([0.0, self.ds(0)], [u_N_td*h_td, self.ds(1)]),
                              grad_term = q_td**2.0/h_td + 0.5*phi_td*h_td, enable=True)

        # CONSERVATION 
        F_h =      createForm(h, h_td, self.h_tf, 1, 
                              self.h_disc, self.h_b,  
                              grad_term = q_td, enable=True)

        # VOLUME FRACTION
        F_phi =    createForm(phi, phi_td, self.phi_tf, 2, 
                              self.phi_disc, self.phi_b, 
                              grad_term = q_td*phi_td/h_td,
                              settling = self.beta*phi_td/h_td, enable=True)

        # DEPOSIT
        F_phi_d =  createForm(phi_d, phi_d_td, self.phi_d_tf, 3, 
                              self.phi_d_disc, self.phi_d_b,
                              weak_b = [[phi_d_td, self.ds(0)], [0.0, self.ds(1)]],
                              settling = -self.beta*phi_td/h_td, enable=True)

        # NOSE LOCATION AND SPEED
        if self.mms:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx 
        else:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx - self.x_N_tf*u_N_td*self.k*dx 
        F_u_N = self.u_N_tf*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            self.u_N_tf*u_N[0]*self.ds(1)
        # F_u_N = self.u_N_tf*(0.5*h_td**-0.5*(phi_td)**0.5)*self.ds(1) - \
        #     self.u_N_tf*u_N[0]*self.ds(1)
        # F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx 
        # F_u_N = self.x_N_tf*(u_N[0] - u_N[1])*dx 

        # combine PDE's
        self.F = F_q + F_h + F_phi + F_phi_d + F_x_N + F_u_N

        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0], trial)
        
        if self.time_discretise.im_func == runge_kutta:
            # 2nd order
            self.F_RK  = (-(1./2.)*inner(self.v, self.w[2])*dx + 
                           (1./2.)*(self.F - x_N_td*inner(self.v, self.w[0])*dx)
                           + x_N_td*inner(self.v, self.w[0])*dx)
            self.J_RK  = derivative(self.F_RK, self.w[0], trial)

            # 3rd order
            self.F_RK1 = (-(3./4.)*inner(self.v, self.w[2])*dx + 
                           (1./4.)*(self.F - x_N_td*inner(self.v, self.w[0])*dx)
                           + x_N_td*inner(self.v, self.w[0])*dx)
            self.F_RK2 = (-(1./3.)*inner(self.v, self.w[2])*dx + 
                           (2./3.)*(self.F - x_N_td*inner(self.v, self.w[0])*dx)
                           + x_N_td*inner(self.v, self.w[0])*dx)
            self.J_RK1 = derivative(self.F_RK1, self.w[0], trial)
            self.J_RK2 = derivative(self.F_RK2, self.w[0], trial)

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

            M = assemble(self.J)
            U, s, Vh = scipy.linalg.svd(M.array())
            cond = s.max()/s.min()
            print cond, s.min(), s.max()
            
            # SOLVE COUPLED EQUATIONS
            if self.time_discretise.im_func == runge_kutta:
                # store previous solution
                self.w[2].assign(self.w[1])

                # runge kutta timestep
                solve(self.F == 0, self.w[0], bcs=self.bc, J=self.J, solver_parameters=solver_parameters)
                if self.slope_limiter:
                    self.slope_limit()

                # # 2nd order
                # self.w[1].assign(self.w[0])
                # solve(self.F_RK == 0, self.w[0], bcs=self.bc, J=self.J_RK, 
                #       solver_parameters=solver_parameters)

                # 3rd order
                self.w[1].assign(self.w[0])
                solve(self.F_RK1 == 0, self.w[0], bcs=self.bc, J=self.J_RK1, 
                      solver_parameters=solver_parameters)
                if self.slope_limiter:
                    self.slope_limit()
                
                self.w[1].assign(self.w[0])
                solve(self.F_RK2 == 0, self.w[0], bcs=self.bc, J=self.J_RK2, 
                      solver_parameters=solver_parameters)

                # replace previous solution
                self.w[1].assign(self.w[2])
            else:
                solve(self.F == 0, self.w[0], bcs=self.bc, J=self.J, solver_parameters=solver_parameters)
                
            if self.slope_limiter:
                self.slope_limit()
                            
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

        # error calc callback
        if self.error_callback:
            return self.error_callback(self)

    def slope_limit(self):

        # get array for variable
        arr = self.w[0].vector().array()

        q0, h0, phi0, phi_d0 = sw_io.map_to_arrays(self.w[0], self.map_dict)[:4] 

        for a in range(4):

            if self.disc[a] == 'DG':

                # create storage arrays for max, min and mean values
                ele_dof = 2 # only for P1 DG elements (self.W.sub(a).dofmap().cell_dofs(0).shape[0])
                n_dof = ele_dof * len(self.mesh.cells())
                u_i_max = np.ones([n_dof]) * 1e-200
                u_i_min = np.ones([n_dof]) * 1e200
                u_c = np.empty([len(self.mesh.cells())])

                # for each vertex in the mesh store the min and max and mean values
                for b in range(len(self.mesh.cells())):

                    # obtain u_c, u_min and u_max
                    u_i = np.array([arr[index] for index in self.W.sub(a).dofmap().cell_dofs(b)])
                    u_c[b] = u_i.mean()

                    n1 = b*ele_dof
                    u_i_max[n1:n1+1] = u_c[b]
                    u_i_min[n1:n1+1] = u_c[b]
                    if b > 0:
                        u_j = np.array([arr[index] for index in self.W.sub(a).dofmap().cell_dofs(b - 1)])
                        if self.slope_limiter == 'vb':
                            u_i_max[n1] = max(u_i_max[n1], u_j.max())
                            u_i_min[n1] = min(u_i_min[n1], u_j.min())
                        elif self.slope_limiter == 'bj':
                            u_i_max[n1] = max(u_i_max[n1], u_j.mean())
                            u_i_min[n1] = min(u_i_min[n1], u_j.mean())
                        else:
                            sys.exit('Can only do vertex-based (vb) or Barth-Jesperson (bj) slope limiting')
                    else:
                        u_i_max[n1] = max(u_i_max[n1], u_i[0])
                        u_i_min[n1] = min(u_i_min[n1], u_i[0])
                    if b + 1 < len(self.mesh.cells()):
                        u_j = np.array([arr[index] for index in self.W.sub(a).dofmap().cell_dofs(b + 1)])
                        if self.slope_limiter == 'vb':
                            u_i_max[n1+1] = max(u_i_max[n1+1], u_j.max())
                            u_i_min[n1+1] = min(u_i_min[n1+1], u_j.min())
                        elif self.slope_limiter == 'bj':
                            u_i_max[n1+1] = max(u_i_max[n1+1], u_j.mean())
                            u_i_min[n1+1] = min(u_i_min[n1+1], u_j.mean())
                        else:
                            sys.exit('Can only do vertex-based (vb) or Barth-Jesperson (bj) slope limiting')
                    else:
                        u_i_max[n1+1] = max(u_i_max[n1+1], u_i[-1])
                        u_i_min[n1+1] = min(u_i_min[n1+1], u_i[-1])

                for b in range(len(self.mesh.cells())):

                    # calculate alpha
                    u_i = np.array([arr[index] for index in self.W.sub(a).dofmap().cell_dofs(b)])
                    alpha = 1.0
                    n1 = b*ele_dof
                    for c in range(ele_dof):
                        if u_i[c] - u_c[b] > 0:
                            alpha = min(alpha, (u_i_max[n1+c] - u_c[b])/(u_i[c] - u_c[b]))
                        if u_i[c] - u_c[b] < 0:
                            alpha = min(alpha, (u_i_min[n1+c] - u_c[b])/(u_i[c] - u_c[b]))

                    # apply slope limiting
                    slope = u_i - u_c[b]
                    indices = self.W.sub(a).dofmap().cell_dofs(b)
                    for c in range(ele_dof):
                        arr[indices[c]] = u_c[b] + alpha*slope[c]

        # put array back into w[0]
        self.w[0].vector()[:] = arr

        q1, h1, phi1, phi_d1 = sw_io.map_to_arrays(self.w[0], self.map_dict)[:4]
        print np.abs((q1-q0)).max(), np.abs((h1-h0)).max(), np.abs((phi1-phi0)).max(), np.abs((phi_d1-phi_d0)).max()

if __name__ == '__main__':

    model = Model()    
    model.plot = 0.0005
    model.initialise_function_spaces()
    model.setup(zero_q = False)     
    model.solve(60.0) 
