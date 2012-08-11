############################################################
# init

from dolfin import *
import numpy as np
from os import getpid
from commands import getoutput

def getMyMemoryUsage():
    mypid = getpid()
    mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
    return mymemory

# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0. 
RED   = "\033[1;37;31m%s\033[0m"
BLUE  = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_blue(s):
    if MPI.process_number()==0:
        print BLUE % s

def info_green(s):
    if MPI.process_number()==0:
        print GREEN % s
    
def info_red(s):
    if MPI.process_number()==0:
        print RED % s

############################################################
# dolfin parameter settings

parameters["linear_algebra_backend"]             = "PETSc"
parameters["form_compiler"]["optimize"]          = False
parameters["form_compiler"]["cpp_optimize"]      = True
parameters["form_compiler"]["quadrature_degree"] = 8
parameters["std_out_all_processes"]              = False;
# parameters["num_threads"]                        = 4

# show information
print_t = True
print_progress = True
info(parameters, False)
set_log_active(True)
list_krylov_solver_preconditioners()
# list_lu_solver_methods()
# list_krylov_solver_methods()

# Check how much memory is actually used by dolfin before we allocate anything
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

############################################################
# simulation parameters
dt_store = 1e-1
shape_U = 2
shape_P = 1
shape_C = 1 
picard_tol = 1e-9
CFL = 0.5
min_dt = 5e-3
max_dt = 1e-1
nl_its = 1
picard_its = 3
dX = 3.5e-3
T = 0.2
nu_scale_ = 0.20
beta_ = 4./2.
L = 1.0
R = 2.217
d = 25e-6
conc = 3.49e-03
theta = 0.5

# set constants
nu = Constant(1e-6)
kappa = Constant(0.0)
nu_scale = Constant(nu_scale_)
g = Constant(9.81)
u_sink = Constant(5.5e-4)
dt = min_dt

# save files
u_file = File("results/u.pvd") 
p_file = File("results/p.pvd") 
c_file = File("results/c.pvd")
c_d_file = File("results/c_d.pvd")

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression(('0.0', '0.0'), degree = shape_U + 1)
p_s = Expression('0.0', degree = shape_P + 1)
Mf = Expression(('0.0', '0.0'), degree = shape_U + 1)
Af = Expression('0.0', degree = shape_C + 1)
c_d_s = Expression('0.0', degree = shape_C + 1)

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
bcu  = [DirichletBC(V, no_slip, "on_boundary && (near(x[0], 0.0) || near(x[0], " + str(L) + "))"),
        DirichletBC(V, no_slip, "on_boundary && near(x[1], 0.0)"),
        DirichletBC(V.sub(1), free_slip, "on_boundary && near(x[1], 0.4)")
        ]

############################################################
# define preassembled part of equations

# MOMENTUM & CONSERVATION

# momentum equation

a_M_m = inner(v, u)*dx
a_K_m = nu*inner(grad(u), grad(v))*dx
L_P_m = inner(v, grad(p))*dx
L_B_m = inner(v, g*g_vector*c*R)*dx

M_m = assemble(a_M_m)
K_m = assemble(a_K_m)
P_m = assemble(L_P_m)
B_m = assemble(L_B_m)
           
u_nl = theta*u_1 + (1.0-theta)*u_star
a_C_m = inner(v, grad(u_nl)*u)*dx

# pressure equation

a_p = inner(grad(p), grad(q))*dx
L_u = inner(u, grad(q))*dx

A_p = assemble(a_p)
B_u = assemble(L_u)

# velocity correction

a_vc = inner(u, v)*dx
L_vc_p = inner(grad(p), v)*dx

A_vc = assemble(a_vc)
B_vc_p = assemble(L_vc_p)

[bc.apply(A_vc) for bc in bcu]

# SEDIMENT ADVECTION AND DIFFUSION

a_M_c = inner(d, c)*dx
a_D_c = inner(grad(d), kappa*grad(c))*dx
a_AdvSink_c = inner(grad(d), u_sink*g_vector*c)*dx
u_sink_n_up = (inner(u_sink*g_vector, n) + abs(inner(u_sink*g_vector, n)))/2.0  
a_AdvSink_hat_c = d*u_sink_n_up*c*ds

M_c = assemble(a_M_c)
D_c = assemble(a_D_c)
AdvSink_c = assemble(a_AdvSink_c - a_AdvSink_hat_c)
        
u_0_ta = theta*u_1+(1.0-theta)*u_0
a_Adv_c = d*div(u_0_ta*c)*dx
vnorm = sqrt(inner(u_0_ta, u_0_ta) + 1e-7*e**-(inner(u_0_ta, u_0_ta)*1.0e5))
a_S_c = nu_scale*h/vnorm*inner(u_0_ta, grad(d))*inner(u_0_ta, grad(c))*dx

# DEPOSITION

a_SI_c_d = d*u_sink_n_up*c*ds 

SI_c_d = assemble(a_SI_c_d)

############################################################
# define solver settings

monitor_convergence = False

u_sol = KrylovSolver('gmres', 'sor')
u_sol.parameters['error_on_nonconvergence'] = False
u_sol.parameters['nonzero_initial_guess'] = True
u_sol.parameters['monitor_convergence'] = monitor_convergence
reset_sparsity = True

p_sol = KrylovSolver('bicgstab', 'hypre_amg')
p_sol.parameters['error_on_nonconvergence'] = False
p_sol.parameters['nonzero_initial_guess'] = True
p_sol.parameters['preconditioner']['reuse'] = True
p_sol.parameters['monitor_convergence'] = monitor_convergence

du_sol = KrylovSolver('cg', 'hypre_amg')
du_sol.parameters['error_on_nonconvergence'] = False
du_sol.parameters['nonzero_initial_guess'] = True
du_sol.parameters['preconditioner']['reuse'] = True
du_sol.parameters['monitor_convergence'] = monitor_convergence

c_sol = KrylovSolver('gmres', 'sor')
du_sol.parameters['error_on_nonconvergence'] = False
du_sol.parameters['nonzero_initial_guess'] = True
du_sol.parameters['preconditioner']['reuse'] = True
du_sol.parameters['monitor_convergence'] = monitor_convergence

############################################################
# store initial conditions

u_file << u_1
p_file << p_1
c_file << c_1
c_d_file << c_d_1
t_store = dt_store

############################################################
# initialise matrix memory

C_m = None
Adv_c = None
S_c = None
G_c = None

list_timings()
timer = Timer("run time") 

############################################################
# time-loop
t = dt
while t < T:

    k = Constant(dt)
    inv_k = 1.0/dt

    nl_it = 0
    picard_its_store = []

    while nl_it < nl_its:

        ############################################################
        # solve equations

        # MOMENTUM & CONSERVATION

        # Iterate until solution is converged
        Eu = 1e6
        Ep = 1e6
        picard_it = 0
        while (Eu > picard_tol and picard_it < picard_its):

            # Define tentative velocity step
            if print_progress and MPI.process_number() == 0:
                info_blue('Assembling tentative velocity step')

            C_m = assemble(a_C_m, tensor = C_m)
            c_0_ta = theta*c_1.vector()+(1.0-theta)*c_0.vector()

            A_m = inv_k*M_m + (1.0-theta)*K_m + (1.0-theta)*C_m
            b_m = inv_k*M_m*u_1.vector() - theta*K_m*u_1.vector() - \
                theta*C_m*u_1.vector() - P_m*p_star.vector() + B_m*c_0_ta

            reset_sparsity = False

            # Compute tentative velocity step
            if print_progress and MPI.process_number() == 0:
                info_blue('Computing tentative velocity step')

            [bc.apply(A_m, b_m) for bc in bcu]
            u_sol.solve(A_m, u_0.vector(), b_m)
            
            # Define pressure correction
            if print_progress and MPI.process_number() == 0:
                info_blue('Building pressure correction rhs')

            b_p = inv_k*B_u*u_0.vector() + A_p*p_star.vector()

            # Compute pressure correction
            if print_progress and MPI.process_number() == 0:
                info_blue('Computing pressure correction')
                            
            p_sol.solve(A_p, p_0.vector(), b_p)

            # Define velocity correction
            if print_progress and MPI.process_number() == 0:
                info_blue('Assembling velocity correction')

            dp = p_0.vector() - p_star.vector()
            b_vc = A_vc*u_0.vector() - dt*B_vc_p*dp

            # Compute velocity correction
            if print_progress and MPI.process_number() == 0:
                info_blue('Computing velocity correction')

            # [bc.apply(b_vc) for bc in bcu]
            du_sol.solve(A_vc, u_0.vector(), b_vc)

            Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)

            # Assign new u_star and p_star
            if print_progress and MPI.process_number() == 0:
                info_blue('Writing u_star and p_star')
            u_star.assign(u_0)
            p_star.assign(p_0)

            picard_it += 1

            if picard_it > 10:
                info_blue("struggling to converge velocity field. Error=%.2e" % Eu)

        picard_its_store.append(picard_it)

        # ADVECTION-DIFFUSION

        # Define advection-diffusion equation
        if print_progress and MPI.process_number() == 0:
            info_blue('Assembling advection-diffusion equation')

        Adv_c = assemble(a_Adv_c, tensor=Adv_c)
        S_c = assemble(a_S_c, tensor=S_c)

        A_c = inv_k*M_c + (1.0-theta)*Adv_c - (1.0-theta)*AdvSink_c + (1.0-theta)*D_c + \
            (1.0-theta)*S_c

        u_0_ta = theta*u_1+(1.0-theta)*u_0
        L_g_c = d*28.1e4*(inner(nu*grad(u_0_ta)*n, nu*grad(u_0_ta)*n))**1.25*ds
        G_c = assemble(L_g_c, tensor=G_c)

        max_E = MPI.max(G_c.array().max())
        if MPI.process_number() == 0:
            info_red('Maximum entrainment rate = %.3e m/s = %.3e m/s * dt' % 
                      (max_E, max_E*dt))

        b_c = inv_k*M_c*c_1.vector() - theta*Adv_c*c_1.vector() + theta*AdvSink_c*c_1.vector() \
            - theta*D_c*c_1.vector() - theta*S_c*c_1.vector() #+ G_c

        if print_progress and MPI.process_number() == 0:
            info_blue('Compute advection-diffusion of sediment')

        c_sol.solve(A_c, c_0.vector(), b_c)

        # DEPOSITED SEDIMENT

        # Define velocity correction
        if print_progress and MPI.process_number() == 0:
            info_blue('Assembling deposition equation')

        A_c_d = inv_k*M_c - (1.0-theta)*SI_c_d
        b_c_d = inv_k*M_c*c_d_1.vector() + theta*SI_c_d*c_d_1.vector()

        if print_progress and MPI.process_number() == 0:
            info_blue('Compute deposition of sediment')

        c_sol.solve(A_c_d, c_d_0.vector(), b_c_d)

        nl_it += 1

    ############################################################
    # Save/store values
        
    if t > t_store:
        u_file << u_0
        p_file << p_0
        c_file << c_0
        c_d_file << c_d_0

        t_store += dt_store

    # Storing values
    if print_progress and MPI.process_number() == 0:
        info_blue('Storing values for next timestep')

    # Store values for next timestep
    u_1.assign(u_0)
    p_1.assign(p_0)
    c_1.assign(c_0)
    c_d_1.assign(c_d_0)

    ############################################################
    # Adaptive time step
    dt = np.ma.fix_invalid(0.5*dX/abs(u_0.vector().array()))
    dt = MPI.min(CFL*dt.min())
    dt = max(dt, min_dt)
    dt = min(dt, max_dt)

    t += dt
    if MPI.process_number() == 0:
        info_green("t = %.5f, dt = %.2e, picard iterations =" % (t, dt) + str(picard_its_store) + ", Eu = %.3e" % Eu)

list_timings()
timer.stop()
if MPI.process_number() == 0:
    info_red('Total computing time = ' + str(timer.value()))
