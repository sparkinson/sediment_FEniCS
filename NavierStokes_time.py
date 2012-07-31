from dolfin import *
import NavierStokes_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True
 
def main():
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)

    ############################################################
    # generate function spaces and functions
    
    V = VectorFunctionSpace(mesh, "CG", shape_U)
    Q = FunctionSpace(mesh, "CG", shape_P)
    if coupled:
        W = V*Q
        (v, q) = TestFunctions(W)
        # w = project(w_s, W)
        w = Function(W)
        (u, p) = split(w)
        # w_star = project(w_s, W)
        w_star = Function(W)
        (u_star, p_star) = w_star.split()
        # w_1 = project(w_s, W)
        w_1 = Function(W)
        (u_1, p_1) = w_1.split()
    else:
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

    ############################################################
    # define equations to be solved

    k = Constant(dt)

    # time-averaged values
    u_ta = theta*u+(1.0-theta)*u_1
    # non-linear variables
    if coupled:
        p_nl = theta*p+(1.0-theta)*p_1
        if newton:
            u_nl = u_ta
        else:
            u_nl = theta*u_star+(1.0-theta)*u_1
    else:
        u_nl = theta*u_star+(1.0-theta)*u_1
        p_nl = p_star

    # momentum equation
    F = (((1./k)*inner(v, (u - u_1))
          + inner(grad(u_nl)*u_ta, v)
          + nu*inner(grad(u_ta), grad(v))
          - inner(v, f)
          - div(v)*p_nl
          )*dx
         + p_nl*inner(v, n)*ds)
    if coupled:
        # coupled momentum equation
        F = F - div(u_ta)*q*dx
    else:
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

    ############################################################
    # define dirichlet boundary conditions
    
    if coupled:
        V_ = W.sub(0)
        Q_ = W.sub(1)
    else:
        V_ = V
        Q_ = Q
    bcu  = [DirichletBC(V_, u_s, "on_boundary")]
    bcp = [DirichletBC(Q_, p_s, "on_boundary && near(x[0], 0.0)")]
    bcw = bcu + bcp

    ############################################################
    # time-loop
    u_file << u_1
    p_file << p_1
    dE = 1.0
    Eu1 = 1e6; Ep1 = 1e6
    while dE > ss_tol:
    
        ############################################################
        # solve equations
    
        if coupled and newton:
            dF = derivative(F, w)
            pde = NonlinearVariationalProblem(F, w, bcw, dF)
            solver = NonlinearVariationalSolver(pde)
            solver.parameters["newton_solver"]["maximum_iterations"] = 500
            solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
            solver.solve()
 
            (u_0, p_0) = w.split() 
        elif coupled:
            Eu = 1.0
            Ep = 1.0
            # Iterate until solution is converged
            while (Eu > picard_tol or Ep > picard_tol):
                solve(F == 0.0, w, bcw)
            
                (u_0, p_0) = w.split()
                Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)
                Ep = errornorm(p_0, p_star, norm_type="L2", degree=shape_U + 1)
                print max(Eu, Ep)

                w_star.assign(w)
                (u_star, p_star) = w_star.split()
        else:
            Eu = 1.0
            Ep = 1.0
            # Iterate until solution is converged
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
                #print max(Eu, Ep)
                
                u_star.assign(u_0)
                p_star.assign(p_0)  
                
        # Compute error
        Eu = errornorm(u_0, u_s, norm_type="L2", degree=shape_U + 1)
        Ep = errornorm(p_0, p_s, norm_type="L2", degree=shape_P + 1)

        # Compute change in error
        dE = max(abs(Eu - Eu1), abs(Ep - Ep1))
        print 'dE: ', dE
        Eu1 = Eu; Ep1 = Ep

        # Store values for next timestep
        if coupled:
            w_1.assign(w)
        else:
            u_1.assign(u_0)
            p_1.assign(p_0)
        
        # Save to file
        u_file << u_0
        p_file << p_0

    return Eu, Ep

# mms test
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_P = int(sys.argv[2]) 
picard_tol = float(sys.argv[3])
ss_tol = float(sys.argv[4])
CFL = float(sys.argv[5])
if sys.argv[6] == 'coupled':
    coupled = True
    if sys.argv[7] == 'newton':
        newton = True
    elif sys.argv[7] == 'picard':
        newton = False
    else:
        raise ValueError('sysarg 5 - picard or newton')
elif sys.argv[6] == 'ipcs':
    coupled = False
else:
    raise ValueError('sysarg 4 - coupled or ipcs')

# show parameters
info(parameters, False)
set_log_active(False)

# describe initial conditions (also analytical solutions) - as strings
u_sS = mms.u_s()
v_sS = mms.v_s()
p_sS = mms.p_s()

# describe source terms - as strings
u_fS = mms.u_fs()
v_fS = mms.v_fs()

# set constants
nu = Constant(1.0)
theta = Constant(0.5)

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression((u_sS, v_sS), degree = shape_U + 1)
p_s = Expression((p_sS), degree = shape_P + 1)
w_s = Expression((u_sS, v_sS, p_sS), degree = shape_U + 1)
f = Expression((u_fS, v_fS), degree = shape_U + 1)

label = 'a','b','c','d','e','f','g'
h = [] # element sizes
E = [] # errors
for i, nx in enumerate([4, 8, 16]):#, 32, 64, 128]):
    u_file = File("results/u_" + label[i] + ".pvd") 
    p_file = File("results/p_" + label[i] + ".pvd") 
    dt = ((pi/nx))*CFL
    h.append(pi/nx)
    print 'dt is: ', dt, '; h is: ', h[-1]
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i][0], " Ep=", E[i][1] 
