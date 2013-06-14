#!/usr/bin/python

from dolfin import *
import mms_test_exp as mms
import numpy as np

set_log_level(ERROR)

def run(nx):

    mesh = IntervalMesh(nx, 0.0, pi)
    n = FacetNormal(mesh)
    n = n[0]
    
    ele_disc = "CG"
    ele_order = 1
    FS = FunctionSpace(mesh, ele_disc, ele_order)
    FST = FunctionSpace(mesh, ele_disc, ele_order + 1)

    v = TestFunction(FS)
    u = project(Expression('0.0', degree=5), FS)
    X = project(Expression('x[0]'), FS)
    S = project(Expression(mms.s_u(), degree=5), FST)
    U = project(Expression(mms.u(), degree=5), FST)

    bc = DirichletBC(FS, U, "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
    
    # F = grad(v)[0]*grad(u)[0]*dx  - v*grad(U)[0]*n*ds - v*S*dx + v*u*n*ds - v*U*n*ds
    F = v*X*grad(u)[0]*dx - v*S*dx + v*u*n*ds - v*U*n*ds
    # F = v*X*grad(u)[0]*dx + grad(v)[0]*grad(u)[0]*dx  - v*grad(U)[0]*n*ds - v*S*dx + v*u*n*ds - v*U*n*ds
        
    solve(F == 0, u)#, bcs=bc)

    print u.vector().array()

    return errornorm(u, U, norm_type="L2", degree_rise=2)    

if __name__=='__main__':
    
    h = []
    E = [] # errors
    
    for i, nx in enumerate([6, 12, 24, 48, 96, 192, 384]):
        dT = (pi/nx) * 1.0
        h.append(pi/nx)
        print 'dt is: ', dT, '; h is: ', h[-1]
        E.append(run(nx))

    for i in range(1, len(E)):
        ru = np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
        print ("h=%10.2E r=%.2f E=%.2e" % (h[i], ru, E[i]))    
