from dolfin import *

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)
    h = CellSize(mesh)

    # generate function spaces and functions
    V = VectorFunctionSpace(mesh, "CG", shape_U)
    D = FunctionSpace(mesh, ele_type, shape_C)
    u = project(u_0, V)
    c = TrialFunction(D)   
    d = TestFunction(D) 
    c1 = project(c_0, D)

    # define equations to be solved    
    a_omega = (   inner(grad(d), k*grad(c) - u*c)   )*dx
    a_ext = (   inner(d*n, u*c)   )*ds
    a = a_omega + a_ext
    L = d*fc_0*dx
    if ele_type == "DG":
        a_int = (   inner(jump(d, n), avg(u*c) - avg(k)*avg(grad(c)))    )*dS
        a += a_int
    
    # prescribe dirichlet boundary conditions
    bcc  = [DirichletBC(D, c_0, "on_boundary")]

    solve(a == L, c1, bcc)
    
    # Compute error
    Ec = errornorm(c1, c_0, norm_type="L2", degree=shape_C + 1)

    # Save to file
    c_file << c1

    return Ec

# MMS TEST
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_C = int(sys.argv[2]) 
ele_type = sys.argv[3] 

# show parameters
info(parameters, False)

# initialise save files
c_file = File("results/tracer.pvd")

# define constants
k = Constant(1.0)
alpha = Constant(5.0)

# describe initial conditions (also analytical solutions) - as strings
u0s = "-sin(x[1]) + cos(x[0]) + 1.00"
u1s = "sin(x[1]) - cos(x[0]) + 1.00"
cs = "sin(x[1]) + 0.700*cos(x[0]) + 3.00"

# describe source terms - as strings
fcs = ("-0.700*(-sin(x[1]) + cos(x[0]) + 1.00)*sin(x[0]) + (sin(x[1]) - cos(x[0]) + 1.00)*cos(x[1]) - (sin(x[1]) + 0.700*cos(x[0]) + 3.00)*(sin(x[0]) - cos(x[1])) + sin(x[1]) + 0.700*cos(x[0])")

# generate expressions for initial conditions, boundary conditions and source terms
u_0 = Expression((u0s, u1s), degree = shape_U + 1)
c_0 = Expression((cs), degree = shape_C + 1)
fc_0 = Expression((fcs), degree = shape_C + 1)

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32]: 
    h.append(pi/nx)
    print h[-1]
    E.append(main())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    rc = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
    print "h=%10.2E rc=%.2f" % (h[i], rc), " Ec=", E[i]
