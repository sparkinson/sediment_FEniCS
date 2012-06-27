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
# u_0a = Constant((1.0,0.0))

# Save files
c_file = File("results/tracer.pvd")
u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
pe_file = File("results/peclet.pvd")

# Defining the function spaces
A = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q
V1 = SubSpace(V, 1)

# Set up boundary condition (apply strong BCs)
noslip = DirichletBC(V, (0, 0), NoSlipBoundary(), "geometric")
freeslip = DirichletBC(V1, 0, FreeSlipBoundary(), "geometric")
bcu = [noslip, freeslip]

# Test and trial functions
a = TestFunction(A)
c = TrialFunction(A)
dw = TestFunction(W)  # (v, q) = TestFunctions(W)
w = TrialFunction(W)  # (u, p) = TrialFunctions(W)
v, q = split(dw)
u, p = split(w)

# Solution functions
w_s = Function(W)
w_1 = Function(W)
u_1, p_1 = split(w_1)
c_s = Function(A)
c_1 = Function(A)

# Initialise and interpolate non-zero intial conditions
c_0 = SedimentInitialCondition()
c_s.interpolate(c_0)
c_1.interpolate(c_0)

# Non-linear variational Navier-Stokes
F = ((1/k)*inner(u - u_1, v)*dx + nu*inner(grad(u), grad(v))*dx + inner(grad(u_1)*u_1, v)*dx 
     + div(v)*p*dx + q*div(u)*dx - inner(f_u, v)*dx + inner(g*c_s, v)*dx)
F  = action(F, w_s)
J = derivative(F, w_s, dw)

problem = NonlinearVariationalProblem(F, w_s, J=J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'
    prm['krylov_solver']['absolute_tolerance'] = 1E-9
    prm['krylov_solver']['relative_tolerance'] = 1E-7
    prm['krylov_solver']['maximum_iterations'] = 1000
    prm['krylov_solver']['gmres']['restart'] = 40
    prm['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

solver.solve()
