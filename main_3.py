from numpy import *
from dolfin import *

class NoSlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS or abs(x[0] - 1.0) < DOLFIN_EPS) and on_boundary

class FreeSlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 0.4) < DOLFIN_EPS and on_boundary

# Load mesh
mesh = Rectangle(0.,0.,1.,0.4,100,40,'left')

# Set parameter values
dt = 0.005                
T = Constant(10)           
nu = Constant(1e-6)        # viscosity
k = Constant(dt)           
kappa = Constant(1e-6)      # diffusivity
alpha = Constant(5.0)      # penalty term
f_c = Constant(0.0)        # scalar field source term
f_u = Constant((0.0,0.0))  # velocity source term
g = Constant((0, -9.81))

# Initial conditions
c_0 = Expression('(1.0-floor(x[0]+0.8))*0.007')
u_0 = Constant((0.0,0.0))
p_0 = Constant(0.0)
# u_0a = Constant((1.0,0.0))

# Save files
c_file = File("results/tracer.pvd")
u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
pe_file = File("results/peclet.pvd")

# Defining the function spaces
A_dg = FunctionSpace(mesh, "DG", 1)
A_cg = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q
V1 = SubSpace(V, 1)


