from dolfin import *

mesh = Interval(10, 0.0, 1.0)
n = FacetNormal(mesh)

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 - DOLFIN_EPS and on_boundary

left_boundary = LeftBoundary()
exterior_facet_domains = FacetFunction("uint", mesh)
exterior_facet_domains.set_all(1)
left_boundary.mark(exterior_facet_domains, 0)

ds = Measure("ds")[exterior_facet_domains]

CG = FunctionSpace(mesh, "CG", 1)
v = TestFunction(CG)
u = TrialFunction(CG)

F = grad(v)*u*dx - v*(1.0)*ds(1) - v*u*n*ds(0)
print assemble(lhs(F)).array()
print assemble(rhs(F)).array()

R = FunctionSpace(mesh, "R", 0)
v = TestFunction(R)
u = TrialFunction(R)
u = Function(R)

F = v*u*ds(1) - v*(1.0)*ds(1)
# print assemble(lhs(F)).array()
# print assemble(rhs(F)).array()

# solve(assemble(lhs(F)), U.vector(), assemble(rhs(F)))
solve(F==0, u)

print u.vector().array()
