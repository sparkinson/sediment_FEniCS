#!/usr/bin/python

from dolfin import *

set_log_level(ERROR)

mesh = IntervalMesh(50, 0.0, 1.0)
n = FacetNormal(mesh)
h = CellSize(mesh)

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 + DOLFIN_EPS and on_boundary
left_boundary = LeftBoundary()
exterior_facet_domains = FacetFunction("uint", mesh)
exterior_facet_domains.set_all(1)
left_boundary.mark(exterior_facet_domains, 0)
ds = Measure("ds")[exterior_facet_domains] 

ele_disc = "DG"
ele_order = 1
SU = False
SU_scale = 0.1
FS = FunctionSpace(mesh, ele_disc, ele_order)

v = TestFunction(FS)
u = dict([(i, project(Constant(1.0), FS)) for i in range(2)])

# bc = [DirichletBC(FS, '0.0', "near(x[0], 1.0) && on_boundary")]

u_td = 0.5*u[0] + 0.5*u[1]
u_N = Constant(-0.1)
S = Constant(0.1)
k_ = 1e-1
k = Constant(k_)

un_up = (u_N*n + abs(u_N*n))/2.0
F = v*(u[0] - u[1])*dx - grad(v)*u_N*u_td*k*dx 
if ele_disc == "CG":
    F += v*n*u_N*u_td*k*ds(0)  
    if SU == True:
        F += Constant(SU_scale)*h/abs(u_N)*inner(u_N, grad(v))*inner(u_N, grad(u_td))*k*dx   # stabilisation
else:
    F += jump(v)*(un_up('+')*u_td('+') - un_up('-')*u_td('-'))*avg(k)*dS + v*un_up*u_td*k*ds(0)    

title = "{}{}".format(ele_order, ele_disc)
if SU == True:
    title = title + 'SU' + str(SU_scale)
title = title + "_"
file = File("{}.pvd".format(title))
file << u[0]

t = 0.0
t_save = 1.0

while t<20.0:

    solve(F == 0, u[0]) #, bcs=bc)
    u[1].assign(u[0])

    t += k_
    print t

    plot(u[0], rescale=True, hardcopy_prefix = title, title = title)
    if t > t_save:
        file << u[0]            
        t_save += 1.0
