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
V1 = SubSpace(V, 1)
Q = FunctionSpace(mesh, "CG", 1)

# Test and trial functions
a = TestFunction(A_cg)
c = TrialFunction(A_cg)
v = TestFunction(V)
u = TrialFunction(V)
q = TestFunction(Q)
pc = TrialFunction(Q)
pe = TrialFunction(V)

# Solution functions
c_s = Function(A_cg)
u_s = Function(V)
pc_s = Function(Q)
pe_s = Function(V)

# Initialise stored values
c_1 = project(c_0,A_cg)
u_1 = project(u_0,V)
p_1 = project(p_0,Q)
# u_1a = project(u_0a,V)

# Mesh-related functions
n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2

# ------------------------------------------------------------------------------------
# Tentative velocity step
# ------------------------------------------------------------------------------------
a1 = (1/k)*inner(u, v)*dx + nu*inner(grad(u), grad(v))*dx
L1 = (1/k)*inner(u_1, v)*dx - inner(grad(u_1)*u_1, v)*dx - inner(grad(p_1), v)*dx + inner(f_u, v)*dx + inner(g*c_s, v)*dx
# ------------------------------------------------------------------------------------
# Pressure correction
# ------------------------------------------------------------------------------------
a2 = inner(grad(pc), grad(q))*dx
L2 = (1/k)*inner(u_s, grad(q))*dx
# ------------------------------------------------------------------------------------
# Velocity update
# ------------------------------------------------------------------------------------
a3 = inner(u, v)*dx
L3 = inner(u_s, v)*dx - k*inner(grad(pc_s), v)*dx
# ------------------------------------------------------------------------------------
# Scalar advection/diffusion
# ------------------------------------------------------------------------------------
# # Bilinear form - DG        
# un = (dot(u_1, n) + abs(dot(u_1, n)))/2.0
# a4_int = (1/k)*a*c*dx + dot(grad(a), kappa*grad(c) - u_1*c)*dx + dot(a, dot(u_1, n)*c)*ds
# a4_fac = kappa('+')*(alpha('+')/h('+'))*dot(jump(a, n), jump(c, n))*dS \
#       - kappa('+')*dot(avg(grad(a)), jump(c, n))*dS \
#       - kappa('+')*dot(jump(a, n), avg(grad(c)))*dS
# a4_vel = dot(jump(a), un('+')*c('+') - un('-')*c('-'))*dS
# a4 = a4_int + a4_fac + a4_vel
# Bilinear form - CG (integrated twice)     
# stabilised_a = (1/tanh(pe_s) - 1/pe_s)*u_1*h
a4 = (1/k)*a*c*dx + dot(grad(a), kappa*grad(c) - u_1*c)*dx + dot(a, dot(u_1, n)*c)*ds
# Linear form
L4 = (1/k)*a*c_1*dx + a*f_c*dx
# ------------------------------------------------------------------------------------
# Grid Peclet Number
# ------------------------------------------------------------------------------------
a5 = inner(v, pe)*dx
L5 = inner(v, u_1*h*(1/(2*kappa)))*dx

# Set up boundary condition (apply strong BCs)
noslip = DirichletBC(V, (0, 0), NoSlipBoundary(), "geometric")
freeslip = DirichletBC(V1, 0, FreeSlipBoundary(), "geometric")
bcu = [noslip, freeslip]

# Assemble matrices that can be assembled once
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A5 = assemble(a5)

# Time-stepping
t = dt
t_save = -0.1
while t < T + DOLFIN_EPS:

    # Grid Peclet number
    b5 = assemble(L5)
    solve(A5, pe_s.vector(), b5, "gmres", "default")
    end()

    # Tracer advection/diffusion
    # Assemble
    A4 = assemble(a4)
    b4 = assemble(L4)
    # Solve system
    solve(A4, c_s.vector(), b4)

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u_s.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    b2 = assemble(L2)
    solve(A2, pc_s.vector(), b2, "gmres", "amg")
    end()

    # Velocity correction
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u_s.vector(), b3, "gmres", "default")
    end()

    # Store value of c_1
    c_1.assign(c_s)
    u_1.assign(u_s)
    p_1 = project(pc_s + p_1, Q)
    
    # Project solution to a continuous function space and save
    if t >= t_save + 0.1:
        # c_s_cg = project(c_s, V=A_cg)
        c_file << c_1
        u_file << u_1
        p_file << p_1
        pe_file << pe_s
        t_save = t

    t += dt
    print "t =", t
