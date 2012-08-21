from dolfin import *
from dolfin_adjoint import *
from dolfin_tools import *
import numpy as np

############################################################
# DOLFIN SETTINGS

parameters["linear_algebra_backend"]             = "PETSc"
parameters["form_compiler"]["optimize"]          = False
parameters["form_compiler"]["cpp_optimize"]      = True
parameters["form_compiler"]["quadrature_degree"] = 8
parameters["std_out_all_processes"]              = False;

# show information
print_t = True
print_progress = True
info(parameters, False)
set_log_active(False)
# list_krylov_solver_preconditioners()
# list_lu_solver_methods()
# list_krylov_solver_methods()

# Check how much memory is actually used by dolfin before we allocate anything
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

############################################################
# SIMULATION USER DEFINED PARAMETERS

# function spaces
shape_U = 2
shape_P = 1
shape_C = 1 

# output frequency
dt_store = 1e-1

# time-loop
min_timestep = 0.5e-1
max_timestep = 0.5e-1
T = 5
CFL = 0.5
nl_its = 1
picard_its = 2
picard_tol = 1e-7
theta = 0.5
adams_bashforth = False

# mesh
dX = 2e-2
L = 0.5

# stabilisation
nu_scale_u_ = 0.05
nu_scale_c_ = 0.20
stabilise_u = True
optimal_beta = False

# sediment
R = 2.217
d = 25e-6
conc = 3.49e-03
kappa_ = 0.0
u_sink_ = 5.5e-4

# viscosity
nu_ = 1e-4

# save files
u_file = File("results/u_adj.pvd") 
p_file = File("results/p_adj.pvd") 
c_file = File("results/c_adj.pvd")
c_d_file = File("results/c_d_adj.pvd")

############################################################
# INITIALISE DEPENDENT PARAMETERS

# set constants
nu = Constant(nu_)
kappa = Constant(kappa_)
nu_scale_u = Constant(nu_scale_u_)
nu_scale_c = Constant(nu_scale_c_)
g = Constant(9.81)
u_sink = Constant(u_sink_)

# initialise timestep
timestep = min_timestep

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression(('0.0', '0.0'), degree = shape_U + 1)
p_s = Expression('0.0', degree = shape_P + 1)
Mf = Expression(('0.0', '0.0'), degree = shape_U + 1)
Af = Expression('0.0', degree = shape_C + 1)
c_d_s = Expression('0.0', degree = shape_C + 1)

class sediment_initial(Expression):
    def eval(self, value, x):
        if x[0] < 0.2:
            value[0] = conc 
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
# generate function spaces

# MOMENTUM & CONSERVATION
V = VectorFunctionSpace(mesh, "CG", shape_U)
Q = FunctionSpace(mesh, "CG", shape_P)
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# ADVECTION
D = FunctionSpace(mesh, "CG", shape_C)
c = TrialFunction(D)   
d = TestFunction(D) 

############################################################
# generate functions

# MOMENTUM & CONSERVATION
g_vector = project(g_vector_s, V)
u_ = dict([[str(i), Function(V)] for i in range(3)])
u_['star'] = Function(V)
p_ = {'0': Function(Q), 'star': Function(Q)} 
i, j = indices(2)

# ADVECTION
c = TrialFunction(D)   
d = TestFunction(D) 
c_ = dict([[str(i), interpolate(c_s, D)] for i in range(3)])

# SEDIMENT BEDLOAD 
c_d_ = dict([[str(i), Function(D)] for i in range(3)])

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
bcu  = [DirichletBC(V, no_slip, "on_boundary && (near(x[0], 0.0) || near(x[0], " + str(L) + "))"),
        DirichletBC(V, no_slip, "on_boundary && near(x[1], 0.0)"),
        DirichletBC(V.sub(1), free_slip, "on_boundary && near(x[1], 0.4)")
        ]

############################################################
# store initial conditions

u_file << u_['0']
p_file << p_['0']
c_file << c_['0']
c_d_file << c_d_['0']
t_store = dt_store

############################################################
# initialise matrix memory

C_m = None
S_m = None
Adv_c = None
S_c = None
G_c = None

list_timings()
timer = Timer("run time") 

############################################################
# time-loop
t = timestep
while t < T:

    k = Constant(timestep)
    inv_k = 1.0/timestep

    nl_it = 0
    picard_its_store = []

    ############################################################
    # define equations to be solved

    # time-averaged values
    u_ta = theta*u_['1']+(1.0-theta)*u_['star']
    c_ta = theta*c_['1']+(1.0-theta)*c_['0']
    # non-linear variables
    u_nl = theta*u_['1']+(1.0-theta)*u_['0']

    # MOMENTUM & CONSERVATION

    # momentum equation
    F = (((1./k)*inner(v, u_['star'] - u_['1'])
          + inner(grad(u_nl)*u_ta, v)
          + nu*inner(grad(u_ta), grad(v))
          - inner(v, Mf)
          - div(v)*p_['star']
          - inner(v, g*g_vector*c_ta*R)
          )*dx
         )
    # pressure equation
    P = ((k*inner(grad(p_['0'] - p_['star']), grad(q)) - 
          inner(u_['star'], grad(q))
          )*dx 
         )
    # velocity correction
    F_2 = (inner(u_['0'], v) - 
           inner(u_['star'], v) +
           k*inner(grad(p_['0'] - p_['star']), v)
           )*dx

    # ADVECTION-DIFFUSION

    # define equations to be solved 
    vnorm_c = sqrt(inner(u_ta, u_ta) + 1e-14*e**-(inner(u_ta, u_ta)*1.0e5))
    abs_sink_u = sqrt(inner(u_sink*g_vector, n)**2 + 1e-14*e**-(inner(u_sink*g_vector, n)**2*1.0e5))
    u_sink_n_up = (inner(u_sink*g_vector, n) + abs_sink_u)/2.0  
    F_c = (d*(1./k)*(c_['0']- c_['1'])*dx 
           + d*div(u_ta*c_ta)*dx                
           + inner(grad(d), - u_sink*g_vector*c_ta)*dx  # sinking velocity
           + d*u_sink_n_up*c_ta*ds   # settling
           + inner(grad(d), kappa*grad(c_ta))*dx   
           - d*Af*dx   # forcing
           + nu_scale_c*h/vnorm_c*inner(u_ta, grad(d))*inner(u_ta, grad(c_ta))*dx   # stabilisation
           # - d*28.1e4*(inner(nu*grad(u_ta)*n, nu*grad(u_ta)*n))**1.25*ds   # reentrainment
           )

    # DEPOSITION
    D = d*(c_d_['0'] - c_d_['1'])*dx - k*d*u_sink_n_up*c_ta*ds

    nl_it = 0
    while nl_it < nl_its:

        ############################################################
        # solve equations

        # MOMENTUM & CONSERVATION

        # Iterate until solution is converged
        Eu = 1e6
        Ep = 1e6
        picard_it = 0
        while (Eu > picard_tol and picard_it < picard_its):
            # Compute tentative velocity step
            if print_progress:
                info_blue('Computing tentative velocity step')
            solve(F == 0, u_['star'], bcu)

            # Pressure correction
            if print_progress:
                info_blue('Computing pressure correction')
            solve(P == 0, p_['0'])

            # Velocity correction
            if print_progress:
                info_blue('Computing velocity correction')
            solve(F_2 == 0, u_['0'], bcu)

            Eu = errornorm(u_['0'], u_['star'], norm_type="L2", degree=shape_U + 1)

            # Assign new u_['star'] and p_['star']
            if print_progress:
                info_blue('Writing u_star and p_star')
            u_['star'].assign(u_['0'])
            p_['star'].assign(p_['0'])

            picard_it += 1

            if picard_it > 10:
                info_red("struggling to converge velocity field. Error=%.2e" % Eu)

        picard_its_store.append(picard_it)

        # ADVECTION-DIFFUSION

        if print_progress:
            info_blue('Compute advection-diffusion of sediment')
        solve(F_c == 0, c_['0'])

        # DEPOSITED SEDIMENT

        if print_progress:
            info_blue('Compute deposition of sediment')
        solve(D == 0, c_d_['0'])

        nl_it += 1

    if t > t_store:
        # Save to file
        u_file << u_['0']
        p_file << p_['0']
        c_file << c_['0']
        c_d_file << c_d_['0']

        t_store += dt_store

    # Store values for next timestep
    u_['1'].assign(u_['0'])
    c_['1'].assign(c_['0'])
    c_d_['1'].assign(c_d_['0'])

    ############################################################
    # Adaptive time step
    timestep = np.ma.fix_invalid(0.5*dX/abs(u_['0'].vector().array()))
    timestep = MPI.min(CFL*timestep.min())
    timestep = max(timestep, min_timestep)
    timestep = min(timestep, max_timestep)

    t += timestep
    info_green("t = %.5f, timestep = %.2e, picard iterations =" % (t, timestep) + str(picard_its_store) + ", Eu = %.3e" % Eu)

J = Functional(c_d_['0']*dx*dt[FINISH_TIME])
dJdic = compute_gradient(J, InitialConditionParameter(c_['0']), forget=False)
dJdu_s = compute_gradient(J, ScalarParameter(u_sink))

print dJdu_s
a_f = File("results/adj1.pvd") 
a_f << dJdic

list_timings()
timer.stop()
# Check how much memory is used
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory used = ' + dolfin_memory_use)
info_red('Total computing time = ' + str(timer.value()))
