from dolfin import *

# Exact solution
w_e = Expression('x[0] * x[0] + 1')

# Load mesh
mesh = UnitSquare(40, 40)

# Define inflow boundary
class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS and on_boundary

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DG", 1)
V_cg = FunctionSpace(mesh, "CG", 2)
V_u = VectorFunctionSpace(mesh, "CG", 1)

# Create velocity Function
u = Constant((1,0))

# Test and trial functions
w = TrialFunction(V_dg)
v = TestFunction(V_dg)

# Source term
f = Expression('2 * x[0]')

# Mesh-related functions
n = FacetNormal(mesh)

# (dot(v, n) + |dot(v, n)|)/2.0
un_u = (dot(u, n) + abs(dot(u, n)))/2.0
un_d = (dot(u, n) + abs(dot(u, n)))/2.0

# Bilinear form
a = -dot(grad(v), u*w)*dx + \
     dot(jump(v), un_u('+')*w('+') - un_u('-')*w('-') )*dS + \
     dot(v, un_u*w)*ds

# Linear form
L = v*f*dx #- dot(v, un_d*w_e)*ds

# Set up boundary condition (apply strong BCs)
bc = DirichletBC(V_dg, w_e, InflowBoundary(), "geometric")

#bc = None # Uncomment this line to get w = x**2

# Solution function
w_h = Function(V_dg)

# Assemble linear system
A, b = assemble_system(a, L, bc)

# Solve system
solve(A, w_h.vector(), b)

# Project solution to a continuous function space
w_p = project(w_h, V_cg)

# Print the error
print "Error (L2): ", errornorm(w_e, w_h, degree=3)

# Plot solution
plot(w_p, interactive=True)
