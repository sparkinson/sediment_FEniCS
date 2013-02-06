#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *

mesh = IntervalMesh(10, 0.0, 1.0)
n = FacetNormal(mesh)

FS = FunctionSpace(mesh, "CG", 1)
w = Function(FS, name='U')

test = TestFunction(FS)
trial = TrialFunction(FS)

def main(ic):

    a = inner(test, trial)*dx 
    L = inner(test, ic)*dx
    
    solve(a == L, w, solver_parameters={"linear_solver": "mumps"})

ic = project(Constant(1.0), FS, annotate=False)     # this doesn't work
# ic = project(Expression('1.0'), FS)    # this works

main(ic)

J = Functional(inner(w, w)*dx*dt[FINISH_TIME])
dJdv = compute_gradient(J, InitialConditionParameter(ic))
print dJdv.vector().array()
Jv = assemble(inner(w, w)*dx)

adj_html('forward.html', 'forward')

parameters["adjoint"]["stop_annotating"] = True 

def Jhat(ic):
    main(ic)
    print 'Jhat: ', assemble(inner(w, w)*dx)
    return assemble(inner(w, w)*dx)

taylor_test(Jhat, InitialConditionParameter(ic), Jv, dJdv, value = ic, seed=1e-4)    
