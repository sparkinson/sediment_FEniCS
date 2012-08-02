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
    W = V*Q
    (v, q) = TestFunctions(W)
    w = Function(W)
    (u, p) = split(w)
    w_1 = Function(W)
    (u_1, p_1) = w_1.split()

    ############################################################
    # define equations to be solved

    k = Constant(dt)

    ############################################################
    # define dirichlet boundary conditions
    
    V_ = W.sub(0)
    Q_ = W.sub(1)
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
        
        # time-averaged values
        u_ta = theta*u+(1.0-theta)*u_1
        
        F = ((inner(v, u)
              + k*inner(grad(u_ta)*u_ta, v)
              + k*nu*inner(grad(u_ta), grad(v))
              - k*div(u_ta)*q
              - k*div(v)*p
              - inner(v, u_1)
              - k*inner(v, f)
              )*dx
             + k*p*inner(v, n)*ds)
    
        dF = derivative(F, w)
        pde = NonlinearVariationalProblem(F, w, bcw, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.solve()
        
        (u_0, p_0) = w.split()  
                
        # Compute error
        Eu = errornorm(u_0, u_s, norm_type="L2", degree=shape_U + 1)
        Ep = errornorm(p_0, p_s, norm_type="L2", degree=shape_P + 1)

        # Compute change in error
        dE = max(abs(Eu - Eu1), abs(Ep - Ep1))
        print 'dE: ', dE
        Eu1 = Eu; Ep1 = Ep

        # Store values for next timestep
        w_1.assign(w)
        (u_1, p_1) = w_1.split(deepcopy = True)
        
        # Save to file
        u_file << u_0
        p_file << p_0

    return Eu, Ep

# mms test
   
# read command line variables
import sys
shape_U = 2 
shape_P = 1
picard_tol = 1e-3
ss_tol = 1e-4
CFL = 1.0

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
theta = Constant(1.0)

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
