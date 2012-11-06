from dolfin import *

# Create mesh and define function space
mesh = Interval(10, 0.0, 1.0)
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "Lagrange", 1)

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 + DOLFIN_EPS and on_boundary

left_boundary = LeftBoundary()
exterior_facet_domains = FacetFunction("uint", mesh)
exterior_facet_domains.set_all(1)
left_boundary.mark(exterior_facet_domains, 0)

# Define variational problem
u_ = Expression('-0.001')
u = interpolate(u_, V)
c = TrialFunction(V)
v = TestFunction(V)
g = Expression("1.0")
a = div(v*u)*c*dx - v*u*n*c*ds(0)
L = g*v*ds(1)

# Compute solution
c = Function(V)
solve(a == L, c)

# Save solution in VTK format
file = File("poisson.pvd")
file << c

print u.vector().array()
print c.vector().array()
