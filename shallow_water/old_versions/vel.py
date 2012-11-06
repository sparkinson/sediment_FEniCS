from dolfin import *
from dolfin_tools import *
import mms_strings as mms
import numpy as np
import matplotlib.pyplot as plt

info(parameters, False)
set_log_active(True)

mesh = Interval(10, 0.0, 1.0)
n = FacetNormal(mesh)

Q = FunctionSpace(mesh, "CG", 1)
v = TestFunction(Q)
U = Function(Q)

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 + DOLFIN_EPS and on_boundary

left_boundary = LeftBoundary()
exterior_facet_domains = FacetFunction("uint", mesh)
exterior_facet_domains.set_all(1)
left_boundary.mark(exterior_facet_domains, 0)

# v*U*n*ds(0) + 
F_u_N = v*Constant(1.0)*ds - grad(v)*U*n*dx + v*Constant(1.0)*dx

solve(F_u_N == 0, U)

plot(h[0], rescale=False, interactive=True)
