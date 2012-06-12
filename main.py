from dolfin import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
# mesh = Mesh("lshape.xml.gz")
mesh = Rectangle(0.,0.,1.,0.4,200,80,'left')

# Define function spaces (P1-P1)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
A = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
c = TrialFunction(A)
v = TestFunction(V)
q = TestFunction(Q)
a = TestFunction(A)

# Set parameter values
dt = 0.01
T = 3
nu = 0.01
D = 0.0 #1e-10

# # Define time-dependent pressure boundary condition
# p_in = Expression("sin(3.0*t)", t=0.0)

# # Define time-dependent tracer boundary condition
# c_in = Expression("10*sin(3.0*t)", t=0.0)

# Define boundary conditions
noslip = DirichletBC(V, (0, 0),
                     "on_boundary && \
                       (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | x[0] > 1.0 - DOLFIN_EPS)")
V1 = SubSpace(V,1);
freeslip = DirichletBC(V1, 0,
                     "on_boundary && (x[1] > 0.4 - DOLFIN_EPS)")
                       
# inflow  = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
# outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
# tracer_dirichlet = DirichletBC(A, 10.0, "x[1] > 1.0 - DOLFIN_EPS")
bcu = [noslip, freeslip]
# bcp = [inflow, outflow]
# bcc = [tracer_dirichlet]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)
c0 = project(Expression('(1.0-floor(x[0]+0.8))*0.1'),A)
c1 = Function(A)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))
fc = Constant(0)
g = Constant((0, -9.81))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx - inner(g*c1, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Tracer advection
G = Expression('0')
n = FacetNormal(mesh)
a4_int = c*a*dx + (a*inner(u1, grad(c)) + inner(D*grad(c), grad(a)))*k*dx
a4_dif = 0 #kappa('+')*(alpha('+')/h('+'))*dot(jump(v, n), jump(phi, n))*dS \
#       - kappa('+')*dot(avg(grad(v)), jump(phi, n))*dS \
#       - kappa('+')*dot(jump(v, n), avg(grad(phi)))*dS
a4_vel = 0 #jump(a)*inner(u1, n)*(c('+') - c('-'))*dS
a4 = a4_int
L4 = (c0 + k*fc)*a*dx + a*G*ds

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
cfile = File("results/tracer.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    # p_in.t = t

    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    # [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "gmres", "amg")
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()

    # Tracer advection
    begin("Computing tracer advection/diffusion")
    A4 = assemble(a4)
    b4 = assemble(L4)
    # [bc.apply(A4, b4) for bc in bcc]
    solve(A4, c1.vector(), b4, "gmres", "default")
    end()

    # # Plot solution
    # plot(p1, title="Pressure", rescale=True)
    # plot(u1, title="Velocity", rescale=True)
    # plot(c1, title="Tracer", rescale=True)

    # Save to file
    ufile << u1
    pfile << p1
    cfile << c1

    # Move to next time step
    u0.assign(u1)
    c0.assign(c1)
    t += dt
    print "t =", t

# # Hold plot
# interactive()
