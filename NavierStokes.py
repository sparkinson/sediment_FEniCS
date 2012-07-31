from dolfin import *
import NavierStokes_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True
 
def main():
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)

    # generate function spaces and functions
    V = VectorFunctionSpace(mesh, "CG", shape_U)
    Q = FunctionSpace(mesh, "CG", shape_P)
    if coupled:
        W = V*Q
        w = project(w_0, W)
        (u, p) = split(w)
        (v, q) = TestFunctions(W)
        w0 = project(w_0, W)
        (u0, p0) = w0.split()
    else:
        u = TrialFunction(V)
        p = TrialFunction(Q)
        v = TestFunction(V)
        q = TestFunction(Q)
        u0 = project(u_0, V)
        u1 = project(u_0, V)
        p0 = project(p_0, Q)
        p1 = project(p_0, Q)

    # set values of non-linear variables and define solver
    if coupled:
        p_nl = p
        if newton:
            u_nl = u
        else:
            u_nl = u0
    else:
        u_nl = u0
        p_nl = p0 

    # define equations to be solved
    # momentum equation
    F = ((inner(grad(u_nl)*u, v)
          + nu*inner(grad(u), grad(v))
          - inner(v, f_0)
          - div(v)*p_nl
          )*dx
         + p_nl*inner(v, n)*ds)
    if coupled:
        # coupled momentum equation
        F = F - div(u)*q*dx
    else:
        # pressure equation
        P = ((inner(grad(p - p0), grad(q)) - 
              inner(u1, grad(q))
              )*dx 
             + q*inner(u1, n)*ds)
        # pressure correction
        F_2 = (inner(u, v) - 
               inner(u1, v) +
               inner(grad(p1 - p0), v)
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

    # prescribe dirichlet boundary conditions
    if coupled:
        V_ = W.sub(0)
        Q_ = W.sub(1)
    else:
        V_ = V
        Q_ = Q
    bcu  = [DirichletBC(V_, u_0, "on_boundary")]
    bcp = [DirichletBC(Q_, p_0, "on_boundary && near(x[0], 0.0)")]
    bcw = bcu + bcp

    # solve equations
    if coupled and newton:
        dF = derivative(F, w)
        pde = NonlinearVariationalProblem(F, w, bcw, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 500
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.solve()

        w0.assign(w)  
        (u0, p0) = w0.split() 
    elif coupled:
        Eu = 1.0
        Ep = 1.0
        # Iterate until solution is converged
        while (Eu > picardTol or Ep > picardTol):
            solve(F == 0.0, w, bcw)
            
            (u1, p1) = w.split()
            Eu = errornorm(u1, u0, norm_type="L2", degree=shape_U + 1)
            Ep = errornorm(p1, p0, norm_type="L2", degree=shape_U + 1)
            print max(Eu, Ep)

            w0.assign(w)
            (u0, p0) = w0.split()
    else:
        Eu = 1.0
        Ep = 1.0
        # Iterate until solution is converged
        while (Eu > picardTol or Ep > picardTol):
            # Compute tentative velocity step
            b1 = assemble(L1)
            [bc.apply(A1, b1) for bc in bcu]
            solve(A1, u1.vector(), b1, "gmres", "default")

            # Pressure correction
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcp]
            solve(A2, p1.vector(), b2, "gmres", "default")

            # Velocity correction
            b3 = assemble(L3)
            [bc.apply(A3, b3) for bc in bcu]
            solve(A3, u1.vector(), b3, "gmres", "default")

            Eu = errornorm(u1, u0, norm_type="L2", degree=shape_U + 1)
            Ep = errornorm(p1, p0, norm_type="L2", degree=shape_P + 1)
            print max(Eu, Ep)

            u0.assign(u1)
            p0.assign(p1)  
    
    # Compute error
    Eu = errornorm(u0, u_0, norm_type="L2", degree=shape_U + 1)
    Ep = errornorm(p0, p_0, norm_type="L2", degree=shape_P + 1)

    # Save to file
    u_file << u0
    p_file << p0

    return Eu, Ep

# mms test
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_P = int(sys.argv[2]) 
picardTol = float(sys.argv[3])
if sys.argv[4] == 'coupled':
    coupled = True
    if sys.argv[5] == 'newton':
        newton = True
    elif sys.argv[5] == 'picard':
        newton = False
    else:
        raise ValueError('sysarg 5 - picard or newton')
elif sys.argv[4] == 'ipcs':
    coupled = False
else:
    raise ValueError('sysarg 4 - coupled or ipcs')

# show parameters
info(parameters, False)

# initialise save files
u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")

# describe initial conditions (also analytical solutions) - as strings
u0s = mms.u_s()
u1s = mms.v_s()
ps = mms.p_s()

# describe source terms - as strings
u0fs = mms.u_fs()
u1fs = mms.v_fs()

# set constants
nu = Constant(1.0)

# generate expressions for initial conditions, boundary conditions and source terms
u_0 = Expression((u0s, u1s), degree = shape_U + 1)
p_0 = Expression((ps), degree = shape_P + 1)
w_0 = Expression((u0s, u1s, ps), degree = shape_U + 1)
f_0 = Expression((u0fs, u1fs), degree = shape_U + 1)

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32, 64, 128]: 
    h.append(pi/nx)
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i][0], " Ep=", E[i][1] 
