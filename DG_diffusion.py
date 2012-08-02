from dolfin import *
import DG_diffusion_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    mesh = Rectangle(0.,0.,2*pi,2*pi,nx,nx,'right')
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    
    # generate function spaces and functions
    D = FunctionSpace(mesh, ele_type, shape_C)
    W = VectorFunctionSpace(mesh, ele_type, shape_C)
    G = D*W
    g = project(g_s, G)
    (c, q) = split(g)
    (d, w) = TestFunctions(G)   

    # define boundary condition
    bcc  = [DirichletBC(D, c_s, "on_boundary")]
    bcq  = [DirichletBC(W, q_s, "on_boundary")]

    # define equations to be solved and solve  
    if ele_type == "DG":
        # generate function spaces and functions
        D = FunctionSpace(mesh, ele_type, shape_C)
        W = VectorFunctionSpace(mesh, ele_type, shape_C)
        G = D*W
        g = project(g_s, G)
        (c, q) = split(g)
        (d, w) = TestFunctions(G)  

        R_gd = q - grad(c)
        
        # F = (inner(grad(d), kappa*q)*dx 
        #      - inner(jump(d, n), avg(kappa*q))*dS 
        #      - inner(d*n, kappa*q)*ds
        #      - d*fc*dx
        #      + jump(w, n)*avg(c)*dS 
        #      + inner(w, n)*c*ds 
        #      - div(w)*c*dx
        #      - inner(w, q)*dx
        #      )

        # bcg = bcc + bcq

        # dF = derivative(F, g)
        # pde = NonlinearVariationalProblem(F, g, bcg, dF)
        # solver = NonlinearVariationalSolver(pde)
        # solver.parameters["newton_solver"]["maximum_iterations"] = 500
        # solver.parameters["newton_solver"]["relaxation_parameter"] = 0.5
        # solver.solve()
        
        # (c, q) = g.split()

        # Ec = errornorm(c, c_s, norm_type="L2", degree=shape_C + 1)
        # Eq = errornorm(q, q_s, norm_type="L2", degree=shape_C + 1)

        # c_file << c
        
        # # generate function spaces and functions
        # D = FunctionSpace(mesh, ele_type, shape_C)
        # c = TrialFunction(D)
        # d = TestFunction(D)
        # c_0 = project(c_s, D)
        
        # # Define parameters
        # alpha = 4.0
        # gamma = 8.0
        
        # a = (inner(grad(d), kappa*grad(c))*dx
        #      - inner(jump(d, n), avg(kappa*grad(c)))*dS
        #      - inner(avg(kappa*grad(d)), jump(c, n))*dS
        #      - inner(d*n, kappa*grad(c))*ds
        #      - inner(grad(d), n*kappa*c)*ds
        #      + alpha*avg(kappa/h)*inner(jump(d, n), jump(c, n))*dS
        #      + gamma/h*d*c*ds
        #      )
        # L = d*fc*dx 

        # solve(a == L, c_0, bcc)

        # Ec = errornorm(c_0, c_s, norm_type="L2", degree=shape_C + 1)
        # Eq = 1.0
    
        # # Save to file
        # c_file << c_0

    else:
        # generate function spaces and functions
        D = FunctionSpace(mesh, ele_type, shape_C)
        c = project(c_s, D)
        d = TestFunction(D) 

        F = inner(grad(d), kappa*grad(c))*dx - inner(d*n, kappa*grad(c))*ds - d*fc*dx
        solve(F == 0, c, bcc)

        Ec = errornorm(c, c_s, norm_type="L2", degree=shape_C + 1)
        Eq = 1.0
    
        # Save to file
        c_file << c

    return Ec, Eq

# MMS TEST
   
# read command line variables
import sys
shape_C = int(sys.argv[1]) 
ele_type = sys.argv[2] 

kappa = Constant(1.0)

# show parameters
info(parameters, False)
set_log_active(True)

# generate expressions for initial conditions, boundary conditions and source terms
q_s = Expression((mms.q0_s(), mms.q1_s()), degree = shape_C + 1)
c_s = Expression(mms.c_s(), t=0, degree = shape_C + 1)
g_s = Expression((mms.c_s(), mms.q0_s(), mms.q1_s()), degree = shape_C + 1)
fc = Expression(mms.c_fs(), degree = shape_C + 1)

label = 'a','b','c','d','e','f','g'
h = [] # element sizes
E = [] # errors
for i, nx in enumerate([4, 8, 16, 32]):#, 64]):
    c_file = File("results/" + label[i] + ".pvd")
    h.append(pi/nx)
    print 'Edge lengths: ', h[-1]
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    rc = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rq = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E rc=%.2f rq=%.2f Ec=%.2e Eq=%.2e" % (h[i], rc, rq, E[i][0], E[i][1])
