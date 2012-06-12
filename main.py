from dolfin import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
# mesh = Mesh("lshape.xml.gz")
mesh = Rectangle(0.,0.,1.,0.4,100,40,'left')

# Define function spaces (P1-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
A = FunctionSpace(mesh, "DG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
c = TrialFunction(A)
v = TestFunction(V)
q = TestFunction(Q)
a = TestFunction(A)

# Set parameter values
dt = 0.001
T = Constant(10)
nu = Constant(1e-6)
D = Constant(0.0) 

# Define boundary conditions
noslip = DirichletBC(V, (0, 0),
                     "on_boundary && \
                       (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | x[0] > 1.0 - DOLFIN_EPS)")
V1 = SubSpace(V,1);
freeslip = DirichletBC(V1, 0,
                     "on_boundary && (x[1] > 0.4 - DOLFIN_EPS)")
                       
bcu = [noslip, freeslip]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)
c0 = project(Expression('(1.0-floor(x[0]+0.8))*0.0075'),A)
c1 = Function(A)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))
fc = Constant(0)
g = Constant((0, -9.81))

# Penalty term
alpha = Constant(5.0)

# Mesh-related functions
n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2
un = (dot(u1, n) + abs(dot(u1, n)))/2.0

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

# Bilinear form
a4_int = ((1/k)*c*a + dot(grad(a), D*grad(c) - u1*c))*dx
a4_dif = D('+')*(alpha('+')/h('+'))*dot(jump(a, n), jump(c, n))*dS \
      - D('+')*dot(avg(grad(a)), jump(c, n))*dS \
      - D('+')*dot(jump(a, n), avg(grad(c)))*dS
a4_vel =   #dot(jump(a), un('+')*c('+') - un('-')*c('-'))*dS #+ dot(a, un*c)*ds
a4 = a4_int + a4_dif + a4_vel 

# Linear form
L4 = ((1/k)*c0 + fc)*a*dx + a*G*ds

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

    # Compute tentative velocity step
    # begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    # begin("Computing pressure correction")
    b2 = assemble(L2)
    solve(A2, p1.vector(), b2, "gmres", "amg")
    end()

    # Velocity correction
    # begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()

    # Tracer advection
    # begin("Computing tracer advection/diffusion")
    A4 = assemble(a4)
    b4 = assemble(L4)
    solve(A4, c1.vector(), b4, "gmres", "default")
    end()

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
