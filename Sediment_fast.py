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
set_log_active(True)
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
min_dt = 1e-2
max_dt = 1e-1
T = 15.0
CFL = 0.5
nl_its = 1
picard_its = 2
picard_tol = 1e-7
theta = 0.5
adams_bashforth = False

# mesh
dX = 1.0e-2
L = 1.0

# stabilisation
nu_scale_u_ = 0.00
nu_scale_c_ = 0.20
stabilise_u = False
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
u_file = File("results/u_nu-4.pvd") 
p_file = File("results/p_nu-4.pvd") 
c_file = File("results/c_nu-4.pvd")
c_d_file = File("results/c_d_nu-4.pvd")

############################################################
# INITIALISE DEPENDENT PARAMETERS

# set constants
nu = Constant(nu_)
kappa = Constant(kappa_)
nu_scale_u = Constant(nu_scale_u_)
nu_scale_c = Constant(nu_scale_c_)
g = Constant(9.81)
u_sink = Constant(u_sink_)

# initialise dt
dt = min_dt

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
        
if adams_bashforth:
    u_nl = 3./2.*u_['1'] - 1./.2*u_['2']
else:   
    u_nl = theta*u_['1'] + (1.0-theta)*u_['star']
a_C_m = inner(v, grad(u_nl)*u)*dx
vnorm_u = sqrt(inner(u_nl, u_nl) + 1.0e-14*e**-(inner(u_nl, u_nl)*1.0e5))
u_dv = as_vector([u_nl[0]*v[i].dx(0) + u_nl[1]*v[i].dx(1) for i in range(2)])
u_du = as_vector([u_nl[0]*u[i].dx(0) + u_nl[1]*u[i].dx(1) for i in range(2)])
gRe = h*vnorm_u/nu
if optimal_beta:
    beta = atan(gRe)**-1.0 - 1.0*gRe**-1.0
else:
    beta = Constant(1.0)
a_S_m = nu_scale_u*beta*h/vnorm_u*inner(u_dv, u_du)*dx

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
        
u_ta = theta*u_['1']+(1.0-theta)*u_['0']
a_Adv_c = d*div(u_ta*c)*dx
vnorm_c = sqrt(inner(u_ta, u_ta) + 1e-14*e**-(inner(u_ta, u_ta)*1.0e5))
a_S_c = nu_scale_c*h/vnorm_c*inner(u_ta, grad(d))*inner(u_ta, grad(c))*dx

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
            if print_progress:
                info_blue('Assembling tentative velocity step')

            if not adams_bashforth or (picard_it == 0 and nl_it == 0):
                C_m = assemble(a_C_m, tensor = C_m)
            
            A_m = inv_k*M_m + (1.0-theta)*K_m + (1.0-theta)*C_m
            c_ta = theta*c_['1'].vector()+(1.0-theta)*c_['0'].vector()
            b_m = inv_k*M_m*u_['1'].vector() - theta*K_m*u_['1'].vector() - \
                theta*C_m*u_['1'].vector() - P_m*p_['star'].vector() + B_m*c_ta

            if stabilise_u:
                # F = d*temp*dx - d*vnorm_u*dx
                # solve(F == 0, temp)

                # print MPI.max(temp.vector().array().max())
                # print MPI.min(temp.vector().array().min())
                # print MPI.max(temp.vector().array().max())*dX/nu_

                # F = d*temp*dx - d*gRe*dx
                # solve(F == 0, temp)

                # print MPI.max(temp.vector().array().max())
                # print MPI.min(temp.vector().array().min())

                # F = d*temp*dx - d*beta*dx
                # solve(F == 0, temp)

                # print MPI.max(temp.vector().array().max())
                # print MPI.min(temp.vector().array().min())

                if not adams_bashforth or (picard_it == 0 and nl_it == 0):
                    S_m = assemble(a_S_m, tensor=S_m)
                
                A_m += (1.0-theta)*S_m
                b_m += -theta*S_m*u_['1'].vector()

            reset_sparsity = False

            # Compute tentative velocity step
            if print_progress:
                info_blue('Computing tentative velocity step')

            [bc.apply(A_m, b_m) for bc in bcu]
            u_sol.solve(A_m, u_['0'].vector(), b_m)
            
            # Define pressure correction
            if print_progress:
                info_blue('Building pressure correction rhs')

            b_p = inv_k*B_u*u_['0'].vector() + A_p*p_['star'].vector()

            # Compute pressure correction
            if print_progress:
                info_blue('Computing pressure correction')
                            
            p_sol.solve(A_p, p_['0'].vector(), b_p)

            # Define velocity correction
            if print_progress:
                info_blue('Assembling velocity correction')

            dp = p_['0'].vector() - p_['star'].vector()
            b_vc = A_vc*u_['0'].vector() - dt*B_vc_p*dp

            # Compute velocity correction
            if print_progress:
                info_blue('Computing velocity correction')

            # [bc.apply(b_vc) for bc in bcu]
            du_sol.solve(A_vc, u_['0'].vector(), b_vc)

            Eu = errornorm(u_['0'], u_['star'], norm_type="L2", degree=shape_U + 1)

            # Assign new u_['star'] and p_['star']
            if print_progress:
                info_blue('Writing u_star and p_star')
            u_['star'].assign(u_['0'])
            p_['star'].assign(p_['0'])

            picard_it += 1

            if picard_it > 10:
                info_blue("struggling to converge velocity field. Error=%.2e" % Eu)

        picard_its_store.append(picard_it)

        # ADVECTION-DIFFUSION

        # Define advection-diffusion equation
        if print_progress:
            info_blue('Assembling advection-diffusion equation')

        Adv_c = assemble(a_Adv_c, tensor=Adv_c)
        S_c = assemble(a_S_c, tensor=S_c)

        A_c = inv_k*M_c + (1.0-theta)*Adv_c - (1.0-theta)*AdvSink_c + (1.0-theta)*D_c + \
            (1.0-theta)*S_c

        L_g_c = d*28.1e4*(inner(nu*grad(u_ta)*n, nu*grad(u_ta)*n))**1.25*ds
        G_c = assemble(L_g_c, tensor=G_c)

        max_E = MPI.max(G_c.array().max())
        if MPI.process_number() == 0:
            info_red('Maximum entrainment rate = %.3e m/s = %.3e m/s * dt' % 
                      (max_E, max_E*dt))

        b_c = inv_k*M_c*c_['1'].vector() - theta*Adv_c*c_['1'].vector() + \
            theta*AdvSink_c*c_['1'].vector() - theta*D_c*c_['1'].vector() - \
            theta*S_c*c_['1'].vector() #+ G_c

        if print_progress:
            info_blue('Compute advection-diffusion of sediment')

        c_sol.solve(A_c, c_['0'].vector(), b_c)

        # DEPOSITED SEDIMENT

        # Define velocity correction
        if print_progress:
            info_blue('Assembling deposition equation')

        A_c_d = inv_k*M_c
        b_c_d = inv_k*M_c*c_d_['1'].vector() + (1.0-theta)*SI_c_d*c_['0'].vector() + \
            theta*SI_c_d*c_['1'].vector()

        if print_progress:
            info_blue('Compute deposition of sediment')

        c_sol.solve(A_c_d, c_d_['0'].vector(), b_c_d)

        nl_it += 1

    ############################################################
    # Save/store values
        
    if t > t_store:
        u_file << u_['0']
        p_file << p_['0']
        c_file << c_['0']
        c_d_file << c_d_['0']

        t_store += dt_store

    # Storing values
    if print_progress:
        info_blue('Storing values for next timestep')

    # Store values for next timestep
    u_['2'].assign(u_['1'])
    u_['1'].assign(u_['0'])
    c_['1'].assign(c_['0'])
    c_d_['1'].assign(c_d_['0'])

    ############################################################
    # Adaptive time step
    dt = np.ma.fix_invalid(dX/abs(u_['0'].vector().array()))
    dt = MPI.min(CFL*dt.min())
    dt = max(dt, min_dt)
    dt = min(dt, max_dt)

    t += dt
    info_green("t = %.5f, dt = %.2e, picard iterations =" % (t, dt) + str(picard_its_store) + ", Eu = %.3e" % Eu)

# J = FinalFunctional(inner(u, u)*dx)
# dJdic = compute_gradient(J, InitialConditionParameter(c_0), forget=False)
# dJdnu = compute_gradient(J, ScalarParameter(nu))

# a_f = File("results/dJdic.pvd") 
# b_f = File("results/dJdnu.pvd") 
# a_f << dJdic
# b_f << dJdnu

list_timings()
timer.stop()
# Check how much memory is used
dolfin_memory_use = getMyMemoryUsage()
if MPI.process_number() == 0:
    info_red('Memory used = ' + dolfin_memory_use)
    info_red('Total computing time = ' + str(timer.value()))
