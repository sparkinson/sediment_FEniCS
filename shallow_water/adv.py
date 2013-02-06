#!/usr/bin/python

from dolfin import *

set_log_level(ERROR)

mesh = IntervalMesh(20, 0.0, 1.0)
n = FacetNormal(mesh)

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 + DOLFIN_EPS and on_boundary
left_boundary = LeftBoundary()
exterior_facet_domains = FacetFunction("uint", mesh)
exterior_facet_domains.set_all(1)
left_boundary.mark(exterior_facet_domains, 0)
ds = Measure("ds")[exterior_facet_domains] 

FS = FunctionSpace(mesh, "DG", 1)

v = TestFunction(FS)
u = dict([(i, project(Constant(1.0), FS)) for i in range(2)])

# bc = [DirichletBC(FS, '0.0', "near(x[0], 1.0) && on_boundary")]

u_td = 0.5*u[0] + 0.5*u[1]
u_N = Constant(-0.01)
S = Constant(0.1)
k = Constant(1e-2)

un_up = (u_N*n + abs(u_N*n))/2.0
F = v*(u[0] - u[1])*dx - grad(v)*u_N*u_td*k*dx 
# F += v*n*u_N*u_td*k*ds(0)     # CG
F += jump(v)*(un_up('+')*u_td('+') - un_up('-')*u_td('-'))*dS #+ v*un_up*u_td*ds(0)    # DG

while True:
    solve(F == 0, u[0]) #, bcs=bc)
    u[1].assign(u[0])

    plot(u[0], rescale=True)
    interactive()
