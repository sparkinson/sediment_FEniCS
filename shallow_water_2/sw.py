#!/usr/bin/python

import sys
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from optparse import OptionParser
import json
import sw_io
from scipy.special import erf

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
info(parameters, False)
# set_log_active(False)
set_log_level(ERROR)

# smooth functions (also never hit zero)
def smooth_pos(val):
    return (val + smooth_abs(val))/2.0
def smooth_neg(val):
    return (val - smooth_abs(val))/2.0
def smooth_abs(val, eps = 1e-4):
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

class MyEquation():

    def __init__(self, model, index, dirichlet, neumann, 
                 grad_term = None, source = None, enable = True):

        # split w
        w = dict()
        w[0] = split(model.w[0])
        w[1] = split(model.w[1])

        # get functions for index
        u = dict()
        u[0] = w[0][index]
        u[1] = w[1][index]
        u_td = model.time_discretise(u)
        v = TestFunctions(model.W)[index]

        self.dirichlet = dirichlet

        if enable:

            # initialise F
            self.F = v*Constant(0.0)*dx

            # mass term
            if not model.mms:
                self.F += v*(u[0] - u[1])*dx

            # grad term - currently use average flux
            if grad_term:
                self.F -= model.k*grad(v)[0]*grad_term*dx
                self.F += avg(model.k)*jump(v)*avg(grad_term)*model.n('+')*dS
                self.F += model.k*v*grad_term*model.n*(model.ds(0) + model.ds(1))

            # source terms
            if model.mms:
                self.F += v*model.S[index]*model.k*dx
            if source:
                self.F += model.k*v*source*dx

            # bc
            if model.mms:
                self.F -= model.k*v*u_td*model.n*(model.ds(0) + model.ds(1))
                self.F += model.k*v*model.w_ic[index]*model.n*(model.ds(0) + model.ds(1)) 
            else:
                for i in range(2):
                    if dirichlet[i] != None:
                        self.F -= model.k*v*u_td*model.n*model.ds(i)
                        self.F += model.k*v*dirichlet[i]*model.n*model.ds(i)
                    if neumann[i] != None:
                        self.F -= model.k*v*grad(u_td)[0]*model.n*model.ds(i)
                        self.F += model.k*v*neumann[i]*model.n*model.ds(i)

        else:
            self.F = v*(u[0] - u[1])*dx

class Model():
    
    # mesh
    dX = 5.0e-1
    L = 2000.0
    
    # current properties
    l = 150                 # initial length over initial height
    beta = Constant(5e-3)
    theta = Constant(0.0)
    c_f = Constant(1e-3)

    # time stepping
    t = 0.0
    timestep = dX/10.0
    adapt_timestep = False
    adapt_initial_timestep = False
    cfl = Constant(5e-2)
    
    # mms test (default False)
    mms = False

    # discretisation
    disc = ["DG", "DG", "DG", "DG"] # q, h, phi, phi_d
    time_discretise = crank_nicholson
    slope_limiter = 'vb'

    # plotting
    show_plot = True
    save_plot = False

    def define_mesh(self):

        # define geometry
        self.mesh = IntervalMesh(int(self.L/self.dX), 0.0, self.L)
        self.n = FacetNormal(self.mesh)[0]

        # left boundary marked as 0, right as 1
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        left_boundary = LeftBoundary()
        exterior_facet_domains = FacetFunction("uint", self.mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains] 

    def initialise_function_spaces(self):

        # define function spaces
        self.Q     = FunctionSpace(self.mesh, self.disc[0], 1)
        self.H     = FunctionSpace(self.mesh, self.disc[1], 1)
        self.PHI   = FunctionSpace(self.mesh, self.disc[2], 1)
        self.PHI_D = FunctionSpace(self.mesh, self.disc[3], 1)
        self.R     = FunctionSpace(self.mesh, 'R', 0)
        self.W = MixedFunctionSpace([self.Q, 
                                     self.H, 
                                     self.PHI, 
                                     self.PHI_D])
        self.WB = MixedFunctionSpace([self.R, 
                                      self.R, 
                                      self.R, 
                                      self.R])

        # generate dof_map
        sw_io.generate_dof_map(self)

        # define test functions
        (self.q_tf, self.h_tf, self.phi_tf, self.phi_d_tf) = \
            TestFunctions(self.W)
        self.v = TestFunction(self.W)
        self.r_v = TestFunction(self.R)

        # initialise functions
        self.w = dict()
        self.w[0] = Function(self.W, name='U')
        self.wb_0 = Function(self.WB, name='WB0')
        self.wb_1 = Function(self.WB, name='WB1')

    def ic(self):

        class MyExpression(Expression):
            def __init__(self, l):
                self.l = l

            def eval(self, value, x):
                value[0] = 0.5*(1.0 + erf(10.0*(1.0 - (x[0]/self.l))))
        
        # define initial conditions
        self.w_ic = [
            project(Expression('0.0'), self.Q),
            project(MyExpression(self.l), self.H),
            project(MyExpression(self.l), self.PHI),
            project(Expression('0.0'), self.PHI_D)
            ]

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

    def setup(self):
        
        self.define_mesh()
        self.initialise_function_spaces()
        self.ic()  
        self.generate_form()   
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self)
            self.plot_t = self.plot

    def generate_form(self):

        # define adaptive timestep form
        if self.adapt_timestep:
            self.k = project(Expression(str(self.timestep)), self.R)
            self.k_tf = TestFunction(self.R)
            self.k_trf = TrialFunction(self.R)
            self.a_k = self.k_tf*self.k_trf*dx 
        else:
            self.k = Constant(self.timestep)
        
        # get time discretised functions
        q = dict()
        h = dict()
        phi = dict()
        phi_d = dict()
        x_N = dict()
        u_N = dict()
        q[0], h[0], phi[0], phi_d[0] = split(self.w[0])
        q[1], h[1], phi[1], phi_d[1] = split(self.w[1])
        q_td = self.time_discretise(q)
        h_td = self.time_discretise(h)
        phi_td = self.time_discretise(phi)
        u_td = q_td/smooth_pos(h_td)

        self.E = dict()

        # MOMENTUM 
        self.E[0] = MyEquation(model=self,
                               index=0, 
                               dirichlet = (0.0, None),
                               neumann = (None, 0.0),
                               grad_term = (q_td**2.0/smooth_pos(h_td) + 
                                            0.5*phi_td*h_td),
                               source = model.c_f*abs(u_td)*u_td,
                               enable=True)

        # CONSERVATION 
        self.E[1] = MyEquation(model=self,
                               index=1, 
                               dirichlet = (None, None),
                               neumann = (0.0, 0.0),
                               grad_term = q_td, 
                               enable=True)

        # VOLUME FRACTION
        self.E[2] = MyEquation(model=self,
                               index=2, 
                               dirichlet = (None, None),
                               neumann = (0.0, 0.0),
                               grad_term = q_td*phi_td/smooth_pos(h_td),
                               source = self.beta*phi_td/smooth_pos(h_td), 
                               enable=True)
    
        # DEPOSIT
        self.E[3] = MyEquation(model=self,
                               index=3, 
                               dirichlet = (None, None),
                               neumann = (0.0, 0.0),
                               source = -self.beta*phi_td/smooth_pos(h_td), 
                               enable=True)

        # combine PDE's
        self.F = self.E[0].F + self.E[1].F + self.E[2].F + self.E[3].F

        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0])

        # test = TestFunction(self.R)
        # trial = TrialFunction(self.R)
        # L = 0; a = 0
        # for i in range(4):
        #     if self.E[i].dirichlet[0] != None:
        #         a += self.r_v*self.wb_0[i]*self.ds(0)
        #         L += self.r_v*self.E[i].dirichlet[0]*self.ds(0)
        #     else:
        #         a += self.r_v*self.wb_0[i]*dx
        #         L += self.r_v*1.0*dx
        # self.F_wb_0 = a-L
        # L = 0; a = 0
        # for i in range(4):
        #     if self.E[i].dirichlet[1] != None:
        #         a += self.r_v*self.wb_1[i]*self.ds(1)
        #         L += self.r_v*self.E[i].dirichlet[1]*self.ds(1)
        #     else:
        #         a += self.r_v*self.wb_0[i]*dx
        #         L += self.r_v*1.0*dx
        # self.F_wb_1 = a-L

    def slope_limit(self):

        # get array for variable
        arr = self.w[0].vector().array()

        q0, h0, phi0, phi_d0 = sw_io.map_to_arrays(self.w[0], self.map_dict)
        # solve(self.F_wb_0==0, self.wb_0)
        # solve(self.F_wb_1==0, self.wb_1)

        for i_eq in range(4):

            if self.disc[i_eq] == 'DG':

                # create storage arrays for max, min and mean values
                ele_dof = 2 # only for P1 DG elements (self.W.sub(i_eq).dofmap().cell_dofs(0).shape[0])
                n_dof = ele_dof * len(self.mesh.cells())
                u_i_max = np.ones([n_dof]) * 1e-200
                u_i_min = np.ones([n_dof]) * 1e200
                u_c = np.empty([len(self.mesh.cells())])

                # for each vertex in the mesh store the min and max and mean values
                for b in range(len(self.mesh.cells())):

                    # obtain u_c, u_min and u_max
                    u_i = np.array([arr[index] for index in self.W.sub(i_eq).dofmap().cell_dofs(b)])
                    u_c[b] = u_i.mean()

                    n1 = b*ele_dof
                    u_i_max[n1:n1+ele_dof] = u_c[b]
                    u_i_min[n1:n1+ele_dof] = u_c[b]

                    if b > 0:
                        u_j = np.array([arr[index] for index in self.W.sub(i_eq).dofmap().cell_dofs(b - 1)])
                        if self.slope_limiter == 'vb':
                            u_i_max[n1] = max(u_i_max[n1], u_j.max())
                            u_i_min[n1] = min(u_i_min[n1], u_j.min())
                        elif self.slope_limiter == 'bj':
                            u_i_max[n1] = max(u_i_max[n1], u_j.mean())
                            u_i_min[n1] = min(u_i_min[n1], u_j.mean())
                        else:
                            sys.exit('Can only do vertex-based (vb) or Barth-Jesperson (bj) slope limiting')
                    # elif self.E[i_eq].weak_b[0] != None:
                    #     wb = sw_io.map_to_arrays(self.wb_0, self.map_dict)[i_eq]
                    #     u_i_max[n1] = max(u_i_max[n1], wb[0])
                    #     u_i_min[n1] = min(u_i_min[n1], wb[0])
                    if b + 1 < len(self.mesh.cells()):
                        u_j = np.array([arr[index] for index in self.W.sub(i_eq).dofmap().cell_dofs(b + 1)])
                        if self.slope_limiter == 'vb':
                            u_i_max[n1+1] = max(u_i_max[n1+1], u_j.max())
                            u_i_min[n1+1] = min(u_i_min[n1+1], u_j.min())
                        elif self.slope_limiter == 'bj':
                            u_i_max[n1+1] = max(u_i_max[n1+1], u_j.mean())
                            u_i_min[n1+1] = min(u_i_min[n1+1], u_j.mean())
                        else:
                            sys.exit('Can only do vertex-based (vb) or Barth-Jesperson (bj) slope limiting')
                    # elif self.E[i_eq].weak_b[1] != None:
                    #     wb = sw_io.map_to_arrays(self.wb_1, self.map_dict)[i_eq]
                    #     u_i_max[n1+1] = max(u_i_max[n1], wb[0])
                    #     u_i_min[n1+1] = min(u_i_min[n1], wb[0])

                # print u_i_max
                # print u_i_min
                # print wb[0]

                for b in range(len(self.mesh.cells())):

                    # calculate alpha
                    u_i = np.array([arr[index] for index in self.W.sub(i_eq).dofmap().cell_dofs(b)])
                    alpha = 1.0
                    n1 = b*ele_dof
                    for c in range(ele_dof):
                        if u_i[c] - u_c[b] > 0:
                            alpha = min(alpha, (u_i_max[n1+c] - u_c[b])/(u_i[c] - u_c[b]))
                        if u_i[c] - u_c[b] < 0:
                            alpha = min(alpha, (u_i_min[n1+c] - u_c[b])/(u_i[c] - u_c[b]))

                    # apply slope limiting
                    slope = u_i - u_c[b]
                    indices = self.W.sub(i_eq).dofmap().cell_dofs(b)
                    for c in range(ele_dof):
                        arr[indices[c]] = u_c[b] + alpha*slope[c]

        # put array back into w[0]
        self.w[0].vector()[:] = arr

        q1, h1, phi1, phi_d1 = sw_io.map_to_arrays(self.w[0], self.map_dict)[:4]
        print np.abs((q1-q0)).max(), np.abs((h1-h0)).max(), np.abs((phi1-phi0)).max(), np.abs((phi_d1-phi_d0)).max()
        
    def solve(self, T = None, tol = None):
        
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
        
        delta = 1e10
        while not (time_finish(self.t) or converged(delta)):
            
            # ADAPTIVE TIMESTEP
            if self.adapt_timestep and (self.t > 0.0 or self.adapt_initial_timestep):
                q = sw_io.map_to_arrays(model.w[0], model.map_dict)[0]
                self.L_k = self.k_tf*self.dX/(self.L*q.max())*self.cfl*dx
                solve(self.a_k == self.L_k, self.k)
                self.timestep = self.k.vector().array()[0]
                print self.timestep

            solve(self.F == 0, self.w[0], J=self.J, solver_parameters=solver_parameters)
            if self.slope_limiter:
                self.slope_limit()

            if tol:
                delta = 0.0
                delta = errornorm(w[0], w[1], norm_type="L2", degree_rise=1)/self.timestep

            self.w[1].assign(self.w[0])

            self.t += self.timestep

            # display results
            if self.plot:
                if self.t > self.plot_t:
                    self.plotter.update_plot(self)
                    self.plot_t += self.plot

            print model.t, T, self.w[0].vector().array().min(), self.w[0].vector().array().max()

        if self.plot:
            self.plotter.clean_up()
            
if __name__ == '__main__':

    model = Model()  
    model.plot = 0.0001
    model.setup()
    model.solve(10.0) 
