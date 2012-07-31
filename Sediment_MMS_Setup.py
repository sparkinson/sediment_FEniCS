from dolfin import *
import Sediment_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True
 
def main():
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)
    h = CellSize(mesh)

    ############################################################
    # generate function spaces and functions
    
    # MOMENTUM & CONSERVATION
    V = VectorFunctionSpace(mesh, "CG", shape_U)
    Q = FunctionSpace(mesh, "CG", shape_P)
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)
    u_0 = project(u_s, V)
    u_star = project(u_s, V)
    u_1 = project(u_s, V)
    p_0 = project(p_s, Q)
    p_star = project(p_s, Q)
    p_1 = project(p_s, Q)

    # ADVECTION
    D = FunctionSpace(mesh, "CG", shape_C)
    c = TrialFunction(D)   
    d = TestFunction(D) 
    # c_0 = project(c_s, D)
    # c_1 = project(c_s, D)
    c_0 = Function(D)
    c_1 = Function(D)

    ############################################################
    # define equations to be solved

    k = Constant(dt)
    alpha = 12.0
    gamma = 5.0

    # time-averaged values
    u_ta = theta*u+(1.0-theta)*u_1
    c_ta = theta*c+(1.0-theta)*c_1
    u_0_ta = theta*u_0+(1.0-theta)*u_1
    c_0_ta = theta*c_0+(1.0-theta)*c_1
    # non-linear variables
    u_nl = theta*u_star+(1.0-theta)*u_1
    p_nl = p_star

    # MOMENTUM & CONSERVATION

    # momentum equation
    F = (((1./k)*inner(v, (u - u_1))
          + inner(grad(u_nl)*u_ta, v)
          + nu*inner(grad(u_ta), grad(v))
          - inner(v, Mf)
          - div(v)*p_nl
          )*dx
         + p_nl*inner(v, n)*ds)
    # pressure equation
    P = ((inner(grad(p - p_nl), grad(q)) - 
          inner(u_0, grad(q))
          )*dx 
         + q*inner(u_0, n)*ds)
    # pressure correction
    F_2 = (inner(u, v) - 
           inner(u_0, v) +
           inner(grad(p_0 - p_nl), v)
           )*dx
    # seperate bilinear and linear forms of equations and preassemble bilinear form
    a1 = lhs(F)
    L1 = rhs(F)
    a2 = lhs(P)
    L2 = rhs(P)
    a3 = lhs(F_2)
    L3 = rhs(F_2)
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3) 

    # ADVECTION-DIFFUSION

    # define equations to be solved   
    # F_c = (d*(1./k)*(c - c_1)*dx 
    #        + inner(grad(d), kappa*grad(c) -  u_0_ta*c_ta)*dx 
    #        + d*inner(n, u_0_ta*c_ta)*ds
    #        - inner(d*n, kappa*grad(c))*ds
    #        - d*Af*dx
    #        )
    F_c = (d*(1./k)*(c - c_1)*dx 
           + d*inner(u_0_ta, grad(c_ta))*dx
           + inner(grad(d), kappa*grad(c_ta))*dx 
           # - inner(d*n, kappa*grad(c))*ds  # zero-flux
           - d*Af*dx
           )
    # # SUPG stabilisation
    # # calculate residual
    # r = ((1./k)*(c - c_1) 
    #      + inner(u_0_ta, grad(c_ta))
    #      - div(kappa*grad(c_ta)) 
    #      - Af
    #      ) 
    # # calculate norm of velocity
    # vnorm = sqrt(inner(u_0_ta, u_0_ta))
    # # add stabilisation
    # F_c += (h/2.0*vnorm)*inner(u_0_ta, grad(d))*r*dx
    # seperate bilinear and linear forms of equations and preassemble bilinear form
    a4 = lhs(F_c)
    L4 = rhs(F_c)
    A4 = assemble(a4) 

    ############################################################
    # define dirichlet boundary conditions
    
    bcu  = [DirichletBC(V, u_s, "on_boundary")]
    bcp = [DirichletBC(Q, p_s, "on_boundary && near(x[0], 0.0)")]
    bcc  = [DirichletBC(D, c_s, "on_boundary")]
    bcw = bcu + bcp

    ############################################################
    # time-loop
    u_file << u_1
    p_file << p_1
    dE = 1.0
    Eu1 = 1e6; Ep1 = 1e6; Ec1 = 1e6
    while dE > ss_tol:
        
        nl_it = 0
        while nl_it < nl_its:
    
            ############################################################
            # solve equations

            # MOMENTUM & CONSERVATION

            # Iterate until solution is converged
            Eu = 1.0
            Ep = 1.0
            while (Eu > picard_tol or Ep > picard_tol):
                # Compute tentative velocity step
                b1 = assemble(L1)
                [bc.apply(A1, b1) for bc in bcu]
                solve(A1, u_0.vector(), b1, "gmres", "default")

                # Pressure correction
                b2 = assemble(L2)
                [bc.apply(A2, b2) for bc in bcp]
                solve(A2, p_0.vector(), b2, "gmres", "default")

                # Velocity correction
                b3 = assemble(L3)
                [bc.apply(A3, b3) for bc in bcu]
                solve(A3, u_0.vector(), b3, "gmres", "default")

                Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)
                Ep = errornorm(p_0, p_star, norm_type="L2", degree=shape_P + 1)

                u_star.assign(u_0)
                p_star.assign(p_0)

            # ADVECTION-DIFFUSION

            b4 = assemble(L4)
            [bc.apply(A4, b4) for bc in bcc]
            solve(A4, c_0.vector(), b4, "gmres", "default")

            nl_it += 1
                
        # Compute error
        Eu = errornorm(u_0, u_s, norm_type="L2", degree=shape_U + 1)
        Ep = errornorm(p_0, p_s, norm_type="L2", degree=shape_P + 1)
        Ec = errornorm(c_1, c_s, norm_type="L2", degree=shape_C + 1)

        # Compute change in error
        dE = max(abs(Eu - Eu1), abs(Ep - Ep1), abs(Ec - Ec1))
        print 'dE: ', dE
        Eu1 = Eu; Ep1 = Ep; Ec1 = Ec

        # Store values for next timestep
        u_1.assign(u_0)
        p_1.assign(p_0)
        c_1.assign(c_0)
        
        # Save to file
        u_file << u_0
        p_file << p_0
        c_file << c_0

    return Eu, Ep, Ec

# mms test
   
# read command line variables
import sys
shape_U = 2
shape_P = 1
shape_C = 1
picard_tol = 1e-3
ss_tol = 1e-3
CFL = 1.0
nl_its = 2

# show parameters
info(parameters, False)
set_log_active(True)

# describe initial conditions (also analytical solutions) - as strings
u_sS = mms.u_s()
v_sS = mms.v_s()
p_sS = mms.p_s()
c_sS = mms.c_s()

# describe source terms - as strings
u_fS = mms.u_fs()
v_fS = mms.v_fs()
c_fS = mms.c_fs()

# set constants
nu = Constant(1.0)
theta = Constant(0.5)
kappa = Constant(0.1)

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression((u_sS, v_sS), degree = shape_U + 1)
p_s = Expression(p_sS, degree = shape_P + 1)
c_s = Expression(c_sS, degree = shape_C + 1)
Mf = Expression((u_fS, v_fS), degree = shape_U + 1)
Af = Expression(c_fS, degree = shape_C + 1)

label = 'a','b','c','d','e','f','g'
h = [] # element sizes
E = [] # errors
for i, nx in enumerate([4, 8, 16]):#, 32]):#, 64, 128]):
    u_file = File("results/u_" + label[i] + ".pvd") 
    p_file = File("results/p_" + label[i] + ".pvd") 
    c_file = File("results/c_" + label[i] + ".pvd")
    dt = ((pi/nx))*CFL
    h.append(pi/nx)
    print 'dt is: ', dt, '; h is: ', h[-1]
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    rc = ln(E[i][2]/E[i-1][2])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f rc=%.2f Eu=%.2e Ep=%.2e Ec=%.2e" % (h[i], ru, rp, rc, E[i][0], E[i][1], E[i][2]) 
