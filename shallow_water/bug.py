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

    # mesh
    dX_ = 2.0e-2
    L_ = 1.0

    # current properties
    x_N_ = 0.5
    Fr_ = 1.19
    beta_ = 5e-3

    # discretisation
    q_degree = 2
    h_degree = 1
    phi_degree = 1
    phi_d_degree = 1
    h_disc = "CG"
    phi_d_disc = "CG"

    # define geometry
    mesh = IntervalMesh(int(L_/dX_), 0.0, L_)
    n = FacetNormal(mesh)[0]

    # define function spaces
    q_FS = FunctionSpace(mesh, "CG", q_degree)
    h_FS = FunctionSpace(mesh, h_disc, h_degree)
    phi_FS = FunctionSpace(mesh, "CG", phi_degree)
    phi_d_FS = FunctionSpace(mesh, phi_d_disc, phi_d_degree)
    var_N_FS = FunctionSpace(mesh, "R", 0)
    W = MixedFunctionSpace([q_FS, h_FS, phi_FS, phi_d_FS, var_N_FS, var_N_FS])
    X_FS = FunctionSpace(mesh, "CG", 1)
    
    w = dict()
    w[0] = Function(W, name='U')
    
    phi_ic = project(Expression('1.0'), phi_FS)

    # create ic array
    w_ic = [
        0.0, 
        1.0, 
        phi_ic, 
        Function(phi_d_FS, name='phi_d_ic'), 
        0.5, 
        1.0
        ]

    def solve(self):
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

model = Model()
model.solve()

# generate functional
(q, h, phi, phi_d, x_N, u_N) = split(model.w[0])

# form Functional integrals
# get model data
phi_d_aim = sw_io.create_function_from_file('deposit_data.json', model.phi_d_FS)

# form Functional integrals
int_0_scale = Constant(1)
int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*int_0_scale*dx
int_0_scale.assign(1e-2/assemble(int_0))

J = Functional(int_0*dt[FINISH_TIME])

reduced_functional = ReducedFunctional(J, [InitialConditionParameter(model.phi_ic)])

bounds = [[0.5], 
          [1.5]]

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'maxiter': 5,
                            'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}) 
