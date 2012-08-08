from dolfin import *
import numpy as np
import Sediment_MMSFunctions as mms

parameters["std_out_all_processes"] = False;
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
# parameters["num_threads"] = 1;

# simulation parameters
dt_store = 1e-10
shape_U = 2
shape_P = 1
shape_C = 1 
picard_tol = 1e-5
CFL = 0.05
min_dt = 1e-4
max_dt = 1e-1
nl_its = 2
dX = 2.5e-2
T = 20.0
nu_scale_ = 0.20
beta_ = 4./2.
L = 2.0
R = 2.217
d = 25e-6
conc = 3.49e-03

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
u_file = File("results/u_a.pvd") 
p_file = File("results/p_a.pvd") 
c_file = File("results/c_a.pvd")
c_d_file = File("results/c_d_a.pvd")

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression(('0.0', '0.0'), degree = shape_U + 1)
p_s = Expression('0.0', degree = shape_P + 1)
Mf = Expression(('0.0', '0.0'), degree = shape_U + 1)
Af = Expression('0.0', degree = shape_C + 1)

class sediment_initial(Expression):
    def eval(self, value, x):
        if x[0] < 0.2:
            value[0] = conc #0.007737
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

# SEDIMENT BEDLOAD
c_d_0 = Function(D)
c_d_1 = Function(D)

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
    u_ta = theta*u_1+(1.0-theta)*u
    c_ta = theta*c_1+(1.0-theta)*c
    u_0_ta = theta*u_1+(1.0-theta)*u_0
    c_0_ta = theta*c_1+(1.0-theta)*c_0
    # non-linear variables
    u_nl = theta*u_1+(1.0-theta)*u_star

    # MOMENTUM & CONSERVATION

    # momentum equation
    F = (((1./k)*inner(v, u - u_1)
          + inner(grad(u_nl)*u_ta, v)
          + nu*inner(grad(u_ta), grad(v))
          - inner(v, Mf)
          - div(v)*p_star
          - inner(v, g*g_vector*c_0_ta*R)
          )*dx
         + p_star*inner(v, n)*ds
         )
    # SU stabilisation
    # vnorm = sqrt(inner(u_0_ta, u_0_ta))
    # stab_F = nu_scale*h/vnorm*inner(u_0_ta, grad(v))*inner(u_0_ta, grad(u_ta))*dx
    # F += stab_F
    # pressure equation
    P = ((k*inner(grad(p - p_star), grad(q)) - 
          inner(u_0, grad(q))
          )*dx 
         #+ q*inner(u_0, n)*ds # zero normal flow
         )
    # stab_P = beta*h**2*inner(grad(p), grad(q))*dx
    # P += stab_P
    # velocity correction
    F_2 = (inner(u, v) - 
           inner(u_0, v) +
           k*inner(grad(p_0 - p_star), v)
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
    u_sink_n_up = (inner(u_sink*g_vector, n) + abs(inner(u_sink*g_vector, n)))/2.0  
    F_c = (d*(1./k)*(c - c_1)*dx 
           # + d*inner(u_0_ta, grad(c_ta))*dx         # advective form - no sinking velocity
           + d*div(u_0_ta*c_ta)*dx                # conservative form - no sinking velocity
           # + d*inner(u_sink*g_vector, grad(c_ta))*dx # advective form - sinking vel only
           + inner(grad(d), - u_sink*g_vector*c_ta)*dx # conservative form integrated by parts - sinking vel only
           + d*u_sink_n_up*c_ta*ds # deposition of sediment (conservative form integrated by parts only)
           + inner(grad(d), kappa*grad(c_ta))*dx 
           # - inner(d*n, kappa*grad(c))*ds  # zero-flux
           - d*Af*dx
           )

    # reentrainment
    tau_b = abs(nu*grad(u_0_ta)*n)
    # R_p = 0.5829 # (R*g*d**3)**0.5/nu
    Z = inner(tau_b, tau_b)/u_sink*0.7234  # R_p**0.6
    A = Constant(1.3e-7)
    en = u_sink*( A*Z**5.0/(1 + A*Z**5.0/0.3) )

    En = assemble(en*ds)
    print "Integral of entrainment rate = %.2e" % En

    # energy = 0.5*inner(grad(u_0_ta), grad(u_0_ta))*dx
    # E = assemble(energy)
    # print E

    F_c -= d*en*ds

    # stabilisation
    # vnorm = norm(u_1) + 1e-7*e**-(norm(u_1)*1.0e5)
    vnorm = sqrt(inner(u_0_ta, u_0_ta) + 1e-7*e**-(inner(u_0_ta, u_0_ta)*1.0e5))
    stab_F_c = nu_scale*h/vnorm*inner(u_0_ta, grad(d))*inner(u_0_ta, grad(c_ta))*dx
    F_c += stab_F_c

    # seperate bilinear and linear forms of equations and preassemble bilinear form
    a4 = lhs(F_c)
    L4 = rhs(F_c)

    # DEPOSITION

    u_sink_n_up = (inner(u_sink*g_vector, n) + abs(inner(u_sink*g_vector, n)))/2.0
    D = d*(c_d_0 - c_d_1)*dx - k*d*u_sink_n_up*c_0_ta*ds

    nl_it = 0
    while nl_it < nl_its:

        ############################################################
        # solve equations

        # MOMENTUM & CONSERVATION

        # Iterate until solution is converged
        Eu = 1e6
        Ep = 1e6
        picard_it = 0
        while (Eu > picard_tol):
            # Compute tentative velocity step
            if print_log:
                print 'Computing tentative velocity step'
            solve(a1 == L1, u_0, bcu)

            # Pressure correction
            if print_log:
                print 'Computing pressure correction'
            solve(a2 == L2, p_0, bcp)

            # Velocity correction
            if print_log:
                print 'Computing velocity correction'
            solve(a3 == L3, u_0, bcu)

            Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)

            u_star.assign(u_0)
            p_star.assign(p_0)

            picard_it += 1

            if picard_it > 10:
                print picard_it
            if picard_it - picard_it/10*10 > 10:
                print "struggling to converge velocity field. Error=%.2e" % Eu

            # p_0.vector()[:] = p_0.vector().array() - p_0.vector().array()[p_fix]

        # ADVECTION-DIFFUSION

        if print_log:
            print 'Advection-diffusion of sediment'
        A4 = assemble(a4) 
        b4 = assemble(L4)
        [bc.apply(A4, b4) for bc in bcc]
        solve(A4, c_0.vector(), b4, "gmres", "default")

        # DEPOSITED SEDIMENT
        solve(D == 0, c_d_0)

        nl_it += 1

    # u_file << u_0
    # p_file << p_0
    # c_file << c_0
    # c_d_file << c_d_0

    if t > t_store:
        # Save to file
        u_file << u_0
        p_file << p_0
        c_file << c_0
        c_d_file << c_d_0

        t_store += dt_store

    # Store values for next timestep
    u_1.assign(u_0)
    p_1.assign(p_0)
    c_1.assign(c_0)
    c_d_1.assign(c_d_0)

    dt = Calc_timestep(u_0, h)

    t += dt
    print "t=%.5f, dt=%.2e, picard iterations=%d" % (t, dt, picard_it)
