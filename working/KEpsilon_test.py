from dolfin import *
import numpy as np
import KEpsilon_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True
 
def main():

    # define space
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)

    ############################################################

    # generate function spaces and functions

    # momentum equation
    V = VectorFunctionSpace(mesh, "CG", shape_U)
    Q = FunctionSpace(mesh, "CG", shape_P)
    W = V*Q
    (v, q) = TestFunctions(W)
    w = project(w_0, W)
    w_1 = project(w_0, W)
    w_nl = project(w_0, W)
    (u, p) = split(w)
    (u_1, p_1) = w_1.split()
    (u_nl, p_nl) = w_nl.split()

    # k and epsilon equations
    D = FunctionSpace(mesh, "CG", shape_U)
    ke_ = TestFunction(D)
    eps_ = TestFunction(D)
    ke = project(ke_0, D)
    eps = project(eps_0, D)
    ke_1 = project(ke_0, D)
    eps_1 = project(eps_0, D)
    ke_nl = project(ke_0, D)
    eps_nl = project(eps_0, D)

    ############################################################

    # define equations to be solved

    # time-averaged values
    ke_ta = theta*ke+(1.0-theta)*ke_1
    eps_ta = theta*eps+(1.0-theta)*eps_1
    u_ta = u

    # define stress tensors - !!!partly explicit!!!
    tau = 2.0*nu*sym(grad(u_ta)) 
    nu_T = (ke_1**2)/eps_1
    # nu_T_e = Expression(mms.nu_T(), degree = shape_U + 1)
    # nu_T = project(nu_T_e, D)
    tau_R = 2.0*nu_T*sym(grad(u_ta)) #- (2./3.)*ke_1*Identity(2)

    # turbulent kinetic energy
    P = inner(grad(u_ta), tau_R - (2./3.)*ke_ta*Identity(2))
    F_KE = ((ke_*inner(u_ta, grad(ke_ta))
             + inner(grad(ke_), nu_T*grad(ke_ta))
             - ke_*f_ke_0
             - ke_*P
             + ke_*eps_ta
             )*dx
            # - 
            # (inner(ke_*n, nu_T*grad(ke_ta))
            #  )*ds 
            )

    ############################################################

    # define dirichlet boundary conditions

    # kinetic energy
    bcke = [DirichletBC(D, ke_0, "on_boundary")] 

    ############################################################

    solve(F_KE == 0.0, ke, bcke)
        
    # Compute error
    Eke = errornorm(ke, ke_0, norm_type="L2", degree=shape_U + 1)
    EP = errornorm(project(P, D), P_0, norm_type="L2", degree=shape_U + 1)

    # Save to file
    ke_file << ke
    P_ke_file << project(P, D)

    return Eke, EP

# mms test

# show parameters
info(parameters, False)
parameters["form_compiler"]["quadrature_degree"] = 8
set_log_active(True)
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_P = int(sys.argv[2]) 

# set hard-coded variables
theta = Constant(1.0)
nu = Constant(1.0)

# initialise save files
ke_file = File("results/ke.pvd")
P_ke_file = File("results/P_ke.pvd")

# describe initial conditions (also analytical solutions) and forcing - as strings
u0_s = mms.u0_s()
u1_s = mms.u1_s()
p_s = mms.p_s()
ke_s = mms.ke_s()
eps_s = mms.eps_s()
u0_fs = mms.u0_fs()
u1_fs = mms.u1_fs()
ke_fs = mms.ke_fs()
eps_fs = mms.eps_fs()
P0_s = mms.P0_s()

# generate expressions for initial conditions, boundary conditions and source terms
u_0 = Expression((u0_s, u1_s), degree = shape_U + 1)
p_0 = Expression((p_s), degree = shape_P + 1)
w_0 = Expression((u0_s, u1_s, p_s), degree = shape_U + 1)
f_0 = Expression((u0_fs, u1_fs), degree = shape_U + 1)
ke_0 = Expression(ke_s, degree = shape_U + 1)
f_ke_0 = Expression(ke_fs, degree = shape_U + 1)
eps_0 = Expression(eps_s, degree = shape_U + 1)
f_eps_0 = Expression(eps_fs, degree = shape_U + 1)
P_0 = Expression(P0_s, degree = shape_U + 1)

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32, 64, 128, 256]: 
    h.append(pi/nx)
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    rke = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rP = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E rke=%.2f rP=%.2f E_ke=%.2e E_P=%.2e" % (h[i], rke, rP, E[i][0], E[i][1])# , " Eke=", E[i][0], " Eke=", E[i][0]
