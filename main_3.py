from numpy import *
from dolfin import *

class NavierStokes():
    def __init__(self, W):
        self.w = Function(W)
        self.u, self.p = split(w)
        
    
    def F(alpha = 0.5, alpha_nl = 0.5, mode = "Standard", scheme = 'FL'):
        
        
    def tau(self, u, nu):
        return nu*(nabla_grad(u) + nabla_grad(u).T)

    def time_discretisation(self, u_0 = None, u, u_1, u_2 = None, scheme = 'FL', alpha = 0.5, alpha_nl = 0.5):
        u_alpha = alpha*u_1 + (1.0-alpha)*u
        
        if scheme == 'AB': # Adams Bashforth - explicit scheme
            u_tilde = 1.5*u_1 - 0.5*u_2
            u_bar = 1.5*u_1 - 0.5*u_2
        elif scheme == 'FE': # Forward Euler - implicit scheme
            u_tilde = u_1
            u_bar = u_alpha
        elif scheme == 'ABP': # Adams Bashforth Projection - implicit scheme
            u_tilde = 1.5*u_1 - 0.5*u_2
            u_bar = u_alpha
        elif scheme == 'FL': # Fluidity - implicit scheme
            u_tilde = alpha_nl*u_1 + (1.0-alpha_nl)*u_0
            u_bar = u_alpha
        else:
            raise Exception('Unknown time-discretisation for velocity') 
        
        return u_alpha, u_tilde, u_bar 

    def conv(self, u_tilde, u_bar, v, mode="Standard"):
        if (mode == "Standard"):
            return inner(u_tilde*grad(u_bar), v)
        elif (mode == "Divergence"):
            return inner(div(outer(u_tilde, u_bar)), v)
        elif (mode == "Skew"):
            return 0.5*(inner(u_tilde*grad(u_bar), v) + \
                            inner(div(outer(u_tilde, u_bar)), v))
        else:
            raise Exception('Unknown convection mode') 
    
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
f_c = Constant(0.0)        # scalar field source term
f_u = Constant((0.0,0.0))  # velocity source term
g = Constant((0, -9.81))

# Initial conditions
class SedimentInitialCondition(Expression):
    def eval(self, values, x):
        if x[0] < 0.2:
            values[0] = 0.007
        else:
            values[0] = 0.0

self.V = VectorFunctionSpace(mesh, "CG", 1)
self.Q = FunctionSpace(mesh, "CG", 1)
self.W = V*Q

# Save files
c_file = File("results/tracer.pvd")
u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
pe_file = File("results/peclet.pvd")

# Defining the function spaces
A = FunctionSpace(mesh, "CG", 1)
V1 = SubSpace(V, 1)

# Set up boundary condition (apply strong BCs)
noslip = DirichletBC(V, (0, 0), NoSlipBoundary(), "geometric")
freeslip = DirichletBC(V1, 0, FreeSlipBoundary(), "geometric")
bcu = [noslip, freeslip]

# Test and trial functions
a = TestFunction(A)
c = TrialFunction(A)
u, p = split(w)

# Solution functions
w_s = Function(W)
w = Function(W)
u_1, p_1 = split(w_1)
c_s = Function(A)
c_1 = Function(A)

# Initialise and interpolate non-zero intial conditions
c_0 = SedimentInitialCondition()
c_s.interpolate(c_0)
c_1.interpolate(c_0)

# Non-linear variational Navier-Stokes
J = derivative(F, w_s, dw)

solver.solve()
