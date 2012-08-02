from dolfin import *
import numpy

alpha = 3; beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0)
u0.t = 0

def boundary(x, on_boundary):  # define the Dirichlet boundary
    return on_boundary

mesh = UnitCircle(40)

V = FunctionSpace(mesh, "CG", 1)

bc = DirichletBC(V, u0, boundary)

u_1 = project(u0, V)

dt = 0.3      # time step

# u = TrialFunction(V)
# v = TestFunction(V)
# f = Constant(beta - 2 - 2*alpha)

# a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
# L = (u_1 + dt*f)*v*dx

# A = assemble(a)   # assemble only once, before the time stepping

# u = Function(V)   # the unknown at a new time level
# T = 2             # total simulation time
# t = dt

# while t <= T:
#     b = assemble(L)
#     u0.t = t
#     bc.apply(A, b)
#     solve(A, u.vector(), b)

u = Function(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx - (u_1 + dt*f)*v*dx

T = 2
t = dt

while t <= T:
    u0.t = t
    
    dF = derivative(F, u)
    pde = NonlinearVariationalProblem(F, u, bc, dF)
    solver = NonlinearVariationalSolver(pde)
    solver.solve()

    t += dt
    u_1.assign(u)

    u_e = interpolate(u0, V)
    maxdiff = numpy.abs(u_e.vector().array()-u.vector().array()).max()
    print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

