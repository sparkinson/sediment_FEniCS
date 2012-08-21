from dolfin import *
import DG_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    
    # generate function spaces and functions
    D = FunctionSpace(mesh, ele_type, shape_C)
    V = VectorFunctionSpace(mesh, 'CG', shape_U)
    c = TrialFunction(D)   
    d = TestFunction(D)  
    c_0 = project(c_s, D)
    c_1 = project(c_s, D)
    c_0 = Function(D)

    # define equations to be solved and solve  
    k = Constant(dt)
    kappa = Constant(1.0)
    alpha = 1.0
    gamma = 5.0
    
    # time averaged values
    c_ta = c*theta + (1.0-theta)*c_1
    
    bcc = [DirichletBC(D, c_s, 'on_boundary')]
    #bcc = None
    
    un_up = (inner(u_s, n) + abs(inner(u_s, n)))/2.0
    un_down = (inner(u_s, n) - abs(inner(u_s, n)))/2.0
    F = ((1/k)*d*(c-c_1)*dx
         + inner(grad(d), kappa*grad(c_ta) - u_s*c_ta)*dx 
         - d*f*dx)
    if ele_type == 'DG':         
        # advection
        F += (jump(d)*(un_up('+')*c_ta('+') - un_up('-')*c_ta('-'))*dS
              + d*un_up*c_ta*ds
              #+ d*un_down*c_s*ds #for weak bcs inlets
              )
        # diffusion
        F += (alpha*avg(kappa/h)*inner(jump(d, n), jump(c, n))*dS
              #+ gamma/h*d*c*ds
              - inner(jump(d, n), avg(kappa*grad(c)))*dS
              #- inner(avg(grad(d)), jump(n, kappa*c))*dS
              - inner(d*n, kappa*grad(c))*ds
              # - inner(grad(d), n*kappa*c)*ds
              )
        # F += (kappa('+')*avg(alpha/h)*inner(jump(d, n), jump(c_ta, n))*dS 
        #       - kappa('+')*inner(avg(grad(d)), jump(c_ta, n))*dS 
        #       - kappa('+')*inner(jump(d, n), avg(kappa*grad(c_ta)))*dS
        #       )
    else:
        F += inner(d*n, u_s*c_ta)*ds - inner(d*n, kappa*grad(c_ta))*ds
    a = lhs(F)
    L = rhs(F)

    # t = dt; Ec_1 = 1e6; dE = 1.0
    # c_file << c_0
    # while dE > ss_tol:
    #     solve(a == L, c_0, bcc)
    #     c_1.assign(c_0)
    #     t += dt
    
    #     # Save to file
    #     c_file << c_0

    #     # Compute error
    #     Ec = errornorm(c_1, c_s, norm_type="L2", degree=shape_C + 1)
    #     print 'Error: ', Ec

    #     dE = abs(Ec_1 - Ec)
    #     Ec_1 = Ec

    solve(a == L, c_0, bcc)
    # Compute error
    Ec = errornorm(c_0, c_s, norm_type="L2", degree=shape_C + 1)
    print 'Error: ', Ec

    return Ec
    

# MMS TEST
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_C = int(sys.argv[2]) 
CFL = float(sys.argv[3])
T = float(sys.argv[4])
theta = float(sys.argv[5])
ele_type = sys.argv[6]
ss_tol = float(sys.argv[7])

# show parameters
info(parameters, False)
set_log_active(False)

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression((mms.u0_s(), mms.u1_s()), degree = shape_U + 1)
c_s = Expression(mms.c_s(), degree = shape_C + 1)
f = Expression(mms.c_f(), degree = shape_C + 1)

label = 'a','b','c','d','e','f','g'
h = [] # element sizes
E = [] # errors
for i, nx in enumerate([4, 8, 16, 32, 64]):
    c_file = File("results/" + label[i] + ".pvd")
    dt = CFL*(1./nx)
    h.append(pi/nx)
    print 'Edge lengths: ', h[-1], ' dt: ', dt
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    rc = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
    print "h=%10.2E rc=%.2f" % (h[i], rc), " Ec=", E[i]
