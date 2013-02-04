#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *

parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters['krylov_solver']['relative_tolerance'] = 1e-15
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1

mesh = IntervalMesh(10, 0.0, 1.0)
n = FacetNormal(mesh)

V = FunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)

W = MixedFunctionSpace([V, P, R])
w = Function(W, name='U')

test = TestFunction(W)
trial = TrialFunction(W)

bcc_d = DirichletBC(W.sub(1), '1.0', "near(x[0], 1.0) && on_boundary")
bcq = DirichletBC(W.sub(0), '0.0', "near(x[0], 0.0) && on_boundary")
bc = [bcq, bcc_d]

def main(ic):

    w_ic = [
        ic,
        project(Constant(1.0), P),
        project(Constant(0.2), R)
        ]

    a = 0; L = 0
    for i in range(len(w_ic)):
        a += inner(test[i], trial[i])*dx 
        L += inner(test[i], w_ic[i])*dx
        
    solve(a == L, w, bc, solver_parameters={"linear_solver": "mumps"})

ic = project(Expression('x[0]'), V)
main(ic)

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

J = Functional(inner(w, w)*dx*dt[FINISH_TIME])
dJdv = compute_gradient(J, InitialConditionParameter(ic))
print dJdv
Jv = assemble(inner(w, w)*dx)

parameters["adjoint"]["stop_annotating"] = True # stop registering equations

def Jhat(ic):
    main(ic)
    print 'Jhat: ', assemble(inner(w, w)*dx)
    return assemble(inner(w, w)*dx)

conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jv, dJdv, value = ic, seed=1e-4)    
