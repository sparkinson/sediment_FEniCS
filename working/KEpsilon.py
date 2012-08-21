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

    # calculate dt
    # u_ = w.split(deepcopy=True)[0]
    # u_max = (np.sqrt(u_.vector().array()[0]**2 + u_.vector().array()[1]**2)).max()
    dt = ((pi/nx))*CFL
    print 'dt is: ', dt
    k = Constant(dt)

    # time-averaged values
    u_ta = theta*u+(1.0-theta)*u_1
    p_ta = theta*p+(1.0-theta)*p_1
    ke_ta = theta*ke+(1.0-theta)*ke_1
    eps_ta = theta*eps+(1.0-theta)*eps_1

    # define stress tensors - !!!partly explicit!!!
    tau = 2.0*nu*sym(grad(u_ta)) 
    nu_T = (ke_1**2)/eps_1
    # nu_T_e = Expression(mms.nu_T(), degree = shape_U + 1)
    # nu_T = project(nu_T_e, D)
    tau_R = 2.0*nu_T*sym(grad(u_ta))# - (2./3.)*ke_1*Identity(2)

    # momentum equation
    F = (((1/k)*inner(u - u_1, v)
          + inner(grad(u_ta)*u_ta, v)
          + inner(tau + tau_R, grad(v))
          - inner(v, f_0)
          - div(v)*p_ta
          - div(u_ta)*q
          )*dx
         + p_ta*inner(v, n)*ds)

    # turbulent kinetic energy
    P = inner(grad(u_ta), tau_R - (2./3.)*ke_1*Identity(2))
    F_KE = (((1/k)*inner(ke - ke_1, ke_)
             + ke_*inner(u_ta, grad(ke_ta))
             + inner(grad(ke_), nu_T*grad(ke_ta))
             - ke_*f_ke_0
             - ke_*P
             + ke_*eps_ta
             )*dx
            - 
            (inner(ke_*n, nu_T*grad(ke_ta))
             )*ds 
            )
    
    # turbulent dissipation
    F_EPS = (((1/k)*inner(eps - eps_1, eps_)
              + eps_*inner(u_ta, grad(eps_ta))
              + inner(grad(eps_), nu_T*grad(eps_ta))
              - eps_*f_eps_0
              - eps_*(eps_1/ke_1)*P
              + eps_*eps_ta*(eps_1/ke_1)
              )*dx
             - 
             (inner(eps_*n, nu_T*grad(eps_ta))
              )*ds 
             )

    ############################################################

    # define dirichlet boundary conditions

    # momentum
    V_ = W.sub(0)
    Q_ = W.sub(1)
    bcu = [DirichletBC(V_, u_0, "on_boundary")]
    bcp = [DirichletBC(Q_, p_0, "on_boundary && near(x[0], 0.0)")]
    bcw = bcu + bcp

    # kinetic energy
    bcke = [DirichletBC(D, ke_0, "on_boundary")]

    # dissipation
    bceps = [DirichletBC(D, eps_0, "on_boundary")]    

    ############################################################

    t = dt
    dE = 1.0; Eu_1 = 1.0; Ep_1 = 1.0; Eke_1 = 1.0; Eeps_1 = 1.0
    while (t < T and dE > ss_tol * (dt*10)):
        # solve equations
    
        nl_it = 0
        dnl = 1.0
        while nl_it < nl_its and dnl > nl_tol:

            # kinetic energy
            # print 'solving for ke'
            solve(F_KE == 0.0, ke, bcke)
            
            # epsilon
            # print 'solving for eps'
            dF_EPS = derivative(F_EPS, eps)
            pde = NonlinearVariationalProblem(F_EPS, eps, bceps, dF_EPS)
            solver = NonlinearVariationalSolver(pde)
            solver.parameters["newton_solver"]["maximum_iterations"] = 50
            solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
            solver.solve() 
        
            # momentum
            # print 'solving for momentum'
            dF = derivative(F, w)
            pde = NonlinearVariationalProblem(F, w, bcw, dF)
            solver = NonlinearVariationalSolver(pde)
            solver.parameters["newton_solver"]["maximum_iterations"] = 50
            solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
            solver.solve()

            dnl = max(errornorm(w.split()[0], w_nl.split()[0], norm_type="L2", degree=shape_U + 1), 
                      errornorm(w.split()[1], w_nl.split()[1], norm_type="L2", degree=shape_U + 1), 
                      errornorm(ke, ke_nl, norm_type="L2", degree=shape_U + 1),
                      errornorm(eps, eps_nl, norm_type="L2", degree=shape_U + 1))
            #print 'dnl is:', dnl 

            ke_nl.assign(ke)
            eps_nl.assign(eps)
            w_nl.assign(w) 
                      
            nl_it += 1

        ke_1.assign(ke)
        eps_1.assign(eps)
        w_1.assign(w)  

        t += dt
        # print t

        ############################################################
        
        # Compute error
        Eu = errornorm(w.split()[0], u_0, norm_type="L2", degree=shape_U + 1)
        Ep = errornorm(w.split()[1], p_0, norm_type="L2", degree=shape_P + 1)
        Eke = errornorm(ke, ke_0, norm_type="L2", degree=shape_U + 1)
        Eeps = errornorm(eps, eps_0, norm_type="L2", degree=shape_U + 1)

        # Steady state check (change in error)
        dE = max(abs(Eu_1 - Eu), abs(Ep_1 - Ep), abs(Eke_1 - Eke), abs(Eeps_1 - Eeps))
        print 'dE is:', dE

        Eu_1 = Eu
        Ep_1 = Ep
        Eke_1 = Eke
        Eeps_1 = Eeps

        # print Eu, Ep, Eke, Eeps

    ############################################################

    # Save to file
    u_file << w.split()[0]
    p_file << w.split()[1]
    ke_file << ke
    eps_file << eps
    P_ke_file << project(P, D)

    return Eu, Ep, Eke, Eeps

############################################################

# mms test

# show parameters
info(parameters, False)
parameters["form_compiler"]["quadrature_degree"] = 8
set_log_active(False)
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_P = int(sys.argv[2]) 
CFL = float(sys.argv[3])
T = float(sys.argv[4])
nl_its = int(sys.argv[5])
ss_tol = float(sys.argv[6])
nl_tol = float(sys.argv[7])

# set hard-coded variables
theta = Constant(0.5)
nu = Constant(1.0)

# initialise save files
u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
ke_file = File("results/ke.pvd")
eps_file = File("results/eps.pvd")
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

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32]:#, 64]:#, 128]: 
    h.append(pi/nx)
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    rke = ln(E[i][2]/E[i-1][2])/ln(h[i]/h[i-1])
    reps = ln(E[i][3]/E[i-1][3])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f rke=%.2f reps=%.2f" % (h[i], ru, rp, rke, reps), " Eu=", E[i][0], " Ep=", E[i][1], " Eke=", E[i][2], " Eeps=", E[i][3] 
