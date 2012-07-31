from dolfin import *
import DG_upwind_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    mesh = Rectangle(0.,0.,2*pi,pi/2,nx,nx/4,'right')
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    
    # generate function spaces and functions
    D = FunctionSpace(mesh, ele_type, shape_C)
    V = VectorFunctionSpace(mesh, 'CG', shape_U)
    c = TrialFunction(D)   
    d = TestFunction(D)  
    c_0 = project(c_s, D)
    c_1 = project(c_s, D)

    # define equations to be solved and solve  
    k = Constant(dt)
    
    # time averaged values
    c_ta = c*theta + (1.0-theta)*c_1
        
    boundary_parts = \
        MeshFunction("uint", mesh, mesh.topology().dim()-1)
    
    class Inlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) < DOLFIN_EPS

    inlet_boundary = Inlet()
    inlet_boundary.mark(boundary_parts, 1)
    
    bcc = [DirichletBC(D, c_s, Inlet(), 'geometric')]
    #bcc = None
    
    un_up = (inner(u_s, n) + abs(inner(u_s, n)))/2.0
    un_down = (inner(u_s, n) - abs(inner(u_s, n)))/2.0
    F = ((1/k)*d*(c-c_1)*dx
         - inner(grad(d), u_s*c_ta)*dx 
         - d*f*dx)
    if ele_type == 'DG':     
        # advection
        F += (jump(d)*(un_up('+')*c_ta('+') - un_up('-')*c_ta('-'))*dS
              + d*un_up*c_ta*ds
              # + d*un_down*c_s*ds #for weak bcs
              - d*f*dx
              )
    else:
        F += inner(d*n, u_s*c_ta)*ds
    a = lhs(F)
    L = rhs(F)

    t = dt
    while t < T:
        c_s.t = t
        solve(a == L, c_0, bcc)
        c_1.assign(c_0)
        t += dt
    
        # Save to file
        c_file << c_0

        # Compute error
        Ec = errornorm(c_1, c_s, norm_type="L2", degree=shape_C + 1)
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

# show parameters
info(parameters, False)
set_log_active(False)

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression(('2*pi', '0.0'), degree = shape_U + 1)
c_s = Expression('sin(x[0] - 2*pi*t)', t=0, degree = shape_C + 1)
f = Expression('0.0', degree = shape_C + 1)

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
