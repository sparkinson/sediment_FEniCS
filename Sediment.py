from dolfin import *
import numpy as np
import Sediment_MMSFunctions as mms

parameters["std_out_all_processes"] = False;
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
# parameters["num_threads"] = 1;

# simulation parameters
dt_store = 0.1
shape_U = 1
shape_P = 1
shape_C = 1 
picard_tol = 1e-3
CFL = 0.2
min_dt = 5e-3
max_dt = 1e-1
nl_its = 2
dX = 2.5e-2
T = 5.0
nu_scale_ = 0.25
beta_ = 4./2.
L = 1.0

def Calc_timestep(u, h):
    dt = np.ma.fix_invalid(0.5*dX/abs(u.vector().array()))
    dt = CFL*dt.min()
    dt = max(dt, min_dt)
    dt = min(dt, max_dt)
    return dt

dt = min_dt

# show information
print_t = True
print_log = False
info(parameters, False)
set_log_active(print_log)

# set constants
nu = Constant(1e-6)
theta = Constant(0.5)
kappa = Constant(0.0)
nu_scale = Constant(nu_scale_)
beta = Constant(beta_)
g = Constant(9.81)
u_sink = Constant(5.5e-4)

# save files
u_file = File("results/u_con.pvd") 
p_file = File("results/p_con.pvd") 
c_file = File("results/c_con.pvd")

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression(('0.0', '0.0'), degree = shape_U + 1)
p_s = Expression('0.0', degree = shape_P + 1)
Mf = Expression(('0.0', '0.0'), degree = shape_U + 1)
Af = Expression('0.0', degree = shape_C + 1)

class sediment_initial(Expression):
    def eval(self, value, x):
        if x[0] < 0.2:
            value[0] = 0.007737
        else:
            value[0] = 0.0
c_s = sediment_initial()

# define gravity
g_vector_s = Expression(('0.0', '-1.0'), degree = shape_U + 1)

############################################################
# generate geometry

mesh = Rectangle(0.,0.,L,0.4,int(L/dX),int(0.4/dX),'right')
n = FacetNormal(mesh)
h = CellSize(mesh)

############################################################
# generate function spaces and functions

# MOMENTUM & CONSERVATION
V = VectorFunctionSpace(mesh, "CG", shape_U)
Q = FunctionSpace(mesh, "CG", shape_P)
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
u_0 = project(u_s, V)
u_star = project(u_s, V)
u_1 = project(u_s, V)
p_0 = project(p_s, Q)
p_star = project(p_s, Q)
p_1 = project(p_s, Q)
g_vector = project(g_vector_s, V)

# ADVECTION
D = FunctionSpace(mesh, "CG", shape_C)
c = TrialFunction(D)   
d = TestFunction(D) 
c_0 = interpolate(c_s, D)
c_1 = interpolate(c_s, D)

############################################################
# pressure reference node

coor = mesh.coordinates()
p_fix = 0
for i in range(mesh.num_vertices()):
    if coor[i][0] - L < DOLFIN_EPS and coor[i][1] - 0.4 < DOLFIN_EPS:
        p_fix = i

############################################################
# define dirichlet boundary conditions

free_slip = Expression('0.0', degree = shape_U + 1)
no_slip = Expression(('0.0','0.0'), degree = shape_U + 1)
# bcu  = [DirichletBC(V.sub(1), free_slip, "on_boundary && (near(x[1], 0.0) || near(x[1], 0.4))"), 
#         DirichletBC(V.sub(0), free_slip, "on_boundary && (near(x[0], 0.0) || near(x[0], 1.0))")]
# bcu  = [DirichletBC(V, no_slip, "on_boundary && !(near(x[1], 0.4))"),
#         DirichletBC(V.sub(1), free_slip, "on_boundary && near(x[1], 0.4)")]
bcu  = [DirichletBC(V, no_slip, "on_boundary")]
bcp = []
bcc  = []

############################################################
# store initial conditions

u_file << u_1
p_file << p_1
c_file << c_1
t_store = dt_store

############################################################
# time-loop
t = dt
while t < T:

    ############################################################
    # define equations to be solved
    k = Constant(dt)

    # time-averaged values
    u_ta = theta*u+(1.0-theta)*u_1
    c_ta = theta*c+(1.0-theta)*c_1
    u_0_ta = theta*u_0+(1.0-theta)*u_1
    c_0_ta = theta*c_0+(1.0-theta)*c_1
    # non-linear variables
    u_nl = theta*u_star+(1.0-theta)*u_1
    p_nl = p_star

    # MOMENTUM & CONSERVATION

    # momentum equation
    F = (((1./k)*inner(v, u - u_1)
          + inner(grad(u_nl)*u_ta, v)
          + nu*inner(grad(u_ta), grad(v))
          - inner(v, Mf)
          - div(v)*p_nl
          - inner(v, g*g_vector*c_0_ta)
          )*dx
         + p_nl*inner(v, n)*ds
         )
    # SU stabilisation
    # vnorm = sqrt(inner(u_0_ta, u_0_ta))
    # stab_F = nu_scale*h/vnorm*inner(u_0_ta, grad(v))*inner(u_0_ta, grad(u_ta))*dx
    # F += stab_F
    # pressure equation
    P = ((k*inner(grad(p - p_nl), grad(q)) - 
          inner(u_0, grad(q))
          )*dx 
         #+ q*inner(u_0, n)*ds # zero normal flow
         )
    stab_P = beta*h**2*inner(grad(p), grad(q))*dx
    P += stab_P
    # velocity correction
    F_2 = (inner(u, v) - 
           inner(u_0, v) +
           k*inner(grad(p_0 - p_nl), v)
           )*dx
    # seperate bilinear and linear forms of equations and preassemble bilinear form
    a1 = lhs(F)
    L1 = rhs(F)
    a2 = lhs(P)
    L2 = rhs(P)
    a3 = lhs(F_2)
    L3 = rhs(F_2)

    # ADVECTION-DIFFUSION

    # define equations to be solved   
    F_c = (d*(1./k)*(c - c_1)*dx 
           # + d*inner(u_0_ta - u_sink*g_vector, grad(c_ta))*dx         # advective form
           + d*div((u_0_ta - u_sink*g_vector)*c_ta)*dx                # conservative form
           + inner(grad(d), kappa*grad(c_ta))*dx 
           # - inner(d*n, kappa*grad(c))*ds  # zero-flux
           - d*Af*dx
           )

    # stabilisation
    # vnorm = norm(u_1) + 1e-7*e**-(norm(u_1)*1.0e5)
    vnorm = sqrt(inner(u_0_ta, u_0_ta) + 1e-7*e**-(inner(u_0_ta, u_0_ta)*1.0e5))
    stab_F_c = nu_scale*h/vnorm*inner(u_0_ta, grad(d))*inner(u_0_ta, grad(c_ta))*dx
    F_c += stab_F_c

    # seperate bilinear and linear forms of equations and preassemble bilinear form
    a4 = lhs(F_c)
    L4 = rhs(F_c)

    nl_it = 0
    while nl_it < nl_its:

        ############################################################
        # solve equations

        # MOMENTUM & CONSERVATION

        # Iterate until solution is converged
        Eu = 1e6
        Ep = 1e6
        while (Eu > picard_tol or Ep > picard_tol):
            # Compute tentative velocity step
            if print_log:
                print 'Computing tentative velocity step'
            # A1 = assemble(a1)
            # b1 = assemble(L1)
            # [bc.apply(A1, b1) for bc in bcu]
            # solve(A1, u_0.vector(), b1, "gmres", "default")
            solve(a1 == L1, u_0, bcu)

            # Pressure correction
            if print_log:
                print 'Computing pressure correction'
            # A2 = assemble(a2)
            # b2 = assemble(L2)
            # [bc.apply(A2, b2) for bc in bcp]
            # solve(A2, p_0.vector(), b2, "gmres", "default")
            solve(a2 == L2, p_0, bcp)

            # Velocity correction
            if print_log:
                print 'Computing velocity correction'
            # A3 = assemble(a3)
            # b3 = assemble(L3)
            # [bc.apply(A3, b3) for bc in bcu]
            # solve(A3, u_0.vector(), b3, "gmres", "default")
            solve(a3 == L3, u_0, bcu)

            Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)
            Ep = errornorm(p_0, p_star, norm_type="L2", degree=shape_P + 1)

            u_star.assign(u_0)
            p_star.assign(p_0)

            p_0.vector()[:] = p_0.vector().array() - p_0.vector().array()[p_fix]

        # ADVECTION-DIFFUSION

        if print_log:
            print 'Advection-diffusion of sediment'
        A4 = assemble(a4) 
        b4 = assemble(L4)
        [bc.apply(A4, b4) for bc in bcc]
        solve(A4, c_0.vector(), b4, "gmres", "default")

        nl_it += 1

    if t > t_store:
        # Save to file
        u_file << u_0
        p_file << p_0
        c_file << c_0

        t_store += dt_store

    # Store values for next timestep
    u_1.assign(u_0)
    p_1.assign(p_0)
    c_1.assign(c_0)

    dt = Calc_timestep(u_0, h)

    t += dt
    print t, dt
