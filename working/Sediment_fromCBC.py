from dolfin import *
import numpy as np
import Sediment_MMSFunctions as mms

parameters["std_out_all_processes"] = False;
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8
# parameters["num_threads"] = 1

# simulation parameters
dt_store = 1e-1
shape_U = 2
shape_P = 1
shape_C = 1 
picard_tol = 1e1
CFL = 0.5
min_dt = 5e-2
max_dt = 1e-1
nl_its = 3
dX = 2.5e-2
T = 5.0
nu_scale_ = 0.20
beta_ = 4./2.
L = 3.0
R = 2.217
d = 25e-6
conc = 3.49e-03
theta = 0.5

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
kappa = Constant(0.0)
nu_scale = Constant(nu_scale_)
g = Constant(9.81)
u_sink = Constant(5.5e-4)

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
# bcu  = [DirichletBC(V, no_slip, "on_boundary")]
bcu  = [DirichletBC(V, no_slip, "on_boundary && (near(x[0], 0.0) || near(x[0], " + str(L) + "))"),
        DirichletBC(V, no_slip, "on_boundary && near(x[1], 0.0)"),
        DirichletBC(V.sub(1), free_slip, "on_boundary && near(x[1], 0.4)")
        ]
bcp = []
bcc  = []

############################################################
# define preassembled part of equations

# MOMENTUM & CONSERVATION

# momentum equation

a_M_m = inner(v, u)*dx
a_K_m = nu*inner(grad(u), grad(v))*dx

M_m = assemble(a_M_m)
K_m = assemble(a_K_m)

# pressure equation
a_p = inner(grad(p), grad(q))*dx
A_p = assemble(a_p)

# velocity correction
a_vc = inner(u, v)*dx
A_vc = assemble(a_vc)

# SEDIMENT ADVECTION AND DIFFUSION

a_M_c = inner(d, c)*dx
a_D_c = inner(grad(d), kappa*grad(c))*dx
a_AdvSink_c = inner(grad(d), u_sink*g_vector*c)*dx
u_sink_n_up = (inner(u_sink*g_vector, n) + abs(inner(u_sink*g_vector, n)))/2.0  
a_AdvSink_hat_c = d*u_sink_n_up*c*ds

M_c = assemble(a_M_c)
D_c = assemble(a_D_c)
AdvSink_c = assemble(a_AdvSink_c - a_AdvSink_hat_c)

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
S_m = None
b_p_m = None
b_b_m = None
b_p = None
b_vc = None
Adv_c = None
S_c = None
G_c = None

############################################################
# time-loop
t = dt
while t < T:

    loop_timer = Timer("Loop")

    k = Constant(dt)
    inv_k = 1.0/dt

    nl_it = 0
    picard_its = []

    while nl_it < nl_its:

        ############################################################
        # solve equations

        # MOMENTUM & CONSERVATION

        # Iterate until solution is converged
        Eu = 1e6
        Ep = 1e6
        picard_it = 0
        while (Eu > picard_tol):

            # Define tentative velocity step
            if print_log:
                print 'Assembling tentative velocity step'

            assembly_timer_0 = Timer("Assembly")
            
            u_nl = theta*u_1 + (1.0-theta)*u_star
            a_C_m = inner(v, grad(u_nl)*u)*dx

            C_m = assemble(a_C_m, tensor = C_m)

            A_m = inv_k*M_m + (1.0-theta)*K_m + (1.0-theta)*C_m
            
            L_p_m = inner(v, grad(p_star))*dx 
            c_0_ta = theta*c_1+(1.0-theta)*c_0
            L_b_m = inner(v, g*g_vector*c_0_ta*R)*dx

            b_p_m = assemble(L_p_m, tensor=b_p_m)
            b_b_m = assemble(L_b_m, tensor=b_b_m)

            b_m = inv_k*M_m*u_1.vector() - theta*K_m*u_1.vector() - \
                theta*C_m*u_1.vector() - b_p_m + b_b_m

            assembly_timer_0.stop()

            # Compute tentative velocity step
            if print_log:
                print 'Computing tentative velocity step'
                
            solver_timer_0 = Timer("Solve")
            
            [bc.apply(A_m, b_m) for bc in bcu]
            solve(A_m, u_0.vector(), b_m)
                
            solver_timer_0.stop()
            
            # Define pressure correction
            if print_log:
                print 'Assembling pressure correction'

            assembly_timer_1 = Timer("Assembly")

            L_p = (1./k)*inner(u_0, grad(q))*dx + inner(grad(p_star), grad(q))*dx
            b_p = assemble(L_p, tensor=b_p)

            assembly_timer_1.stop()

            # Compute pressure correction
            if print_log:
                print 'Computing pressure correction'
                
            solver_timer_1 = Timer("Solve")
            
            [bc.apply(A_p, b_p) for bc in bcp]
            solve(A_p, p_0.vector(), b_p)
                
            solver_timer_1.stop()

            # Define velocity correction
            if print_log:
                print 'Assembling velocity correction'

            assembly_timer_2 = Timer("Assembly")

            L_vc = inner(u_0, v)*dx - k*inner(grad(p_0 - p_star), v)*dx
            b_vc = assemble(L_vc, tensor=b_vc)

            assembly_timer_2.stop()

            # Compute velocity correction
            if print_log:
                print 'Computing velocity correction'
                
            solver_timer_2 = Timer("Solve")

            [bc.apply(A_vc, b_vc) for bc in bcu]
            solve(A_vc, u_0.vector(), b_vc)
                
            solver_timer_2.stop()

            Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)

            u_star.assign(u_0)
            p_star.assign(p_0)

            picard_it += 1

            if picard_it > 10:
                print "struggling to converge velocity field. Error=%.2e" % Eu

            p_0.vector()[:] = p_0.vector().array() - p_0.vector().array()[p_fix]

        picard_its.append(picard_it)

        # ADVECTION-DIFFUSION

        # Define velocity correction
        if print_log:
            print 'Assembling advection-diffusion equation'

        assembly_timer_3 = Timer("Assembly")
        
        u_0_ta = theta*u_1+(1.0-theta)*u_0
        a_Adv_c = d*div(u_0_ta*c)*dx
        vnorm = sqrt(inner(u_0_ta, u_0_ta) + 1e-7*e**-(inner(u_0_ta, u_0_ta)*1.0e5))
        a_S_c = nu_scale*h/vnorm*inner(u_0_ta, grad(d))*inner(u_0_ta, grad(c))*dx

        Adv_c = assemble(a_Adv_c, tensor=Adv_c)
        S_c = assemble(a_S_c, tensor=S_c)

        A_c = inv_k*M_c + (1.0-theta)*Adv_c - (1.0-theta)*AdvSink_c + (1.0-theta)*D_c + \
            (1.0-theta)*S_c

        # reentrainment
        # tau_b = abs(nu*grad(u_0_ta)*n)
        # Z = inner(tau_b, tau_b)/u_sink*0.7234
        # A = Constant(1.3e-7)
        # L_g_c = u_sink*( A*Z**5.0/(1 + A*Z**5.0/0.3) )*ds
        # print L_g_c
        # G_c = assemble(L_g_c, tensor=G_c)
        # print G_c

        b_c = inv_k*M_c*c_1.vector() - theta*Adv_c*c_1.vector() + theta*AdvSink_c*c_1.vector() \
            - theta*D_c*c_1.vector() - theta*S_c*c_1.vector() #+ G_c

        assembly_timer_3.stop()

        if print_log:
            print 'Compute advection-diffusion of sediment'
                
        solver_timer_3 = Timer("Solve")

        [bc.apply(A_c, b_c) for bc in bcc]
        solve(A_c, c_0.vector(), b_c)
                
        solver_timer_3.stop()

        # DEPOSITED SEDIMENT
                
        solver_timer_4 = Timer("Solve")

        D = d*(c_d_0 - c_d_1)*dx - k*d*u_sink_n_up*c_0_ta*ds
        solve(D == 0, c_d_0)
                
        solver_timer_4.stop()

        nl_it += 1

    # u_file << u_0
    # p_file << p_0
    # c_file << c_0
    # c_d_file << c_d_0

    save_timer = Timer("Save")
    if t > t_store:
        # Save to file
        u_file << u_0
        p_file << p_0
        c_file << c_0
        c_d_file << c_d_0

        t_store += dt_store
    save_timer.stop()

    # Store values for next timestep
    u_1.assign(u_0)
    p_1.assign(p_0)
    c_1.assign(c_0)
    c_d_1.assign(c_d_0)
    
    # dt = Calc_timestep(u_0, h)

    t += dt
    print "t = %.5f, dt = %.2e, picard iterations =" % (t, dt), picard_its, ", Eu = %.3e" % Eu

    loop_timer.stop()
    sum_0 = (assembly_timer_0.value()+ 
              assembly_timer_1.value()+ 
              assembly_timer_2.value()+ 
              assembly_timer_3.value())
    print ("assembly times = %.3e, %.3e, %.3e, %.3e, total = %.3e"  
           % (assembly_timer_0.value(), 
              assembly_timer_1.value(), 
              assembly_timer_2.value(), 
              assembly_timer_3.value(),
              sum_0))
    sum_1 = (solver_timer_0.value()+ 
              solver_timer_1.value()+ 
              solver_timer_2.value()+ 
              solver_timer_3.value()+  
              solver_timer_4.value())
    print ("solve times = %.3e, %.3e, %.3e, %.3e, %.3e, total = %.3e"  
           % (solver_timer_0.value(), 
              solver_timer_1.value(), 
              solver_timer_2.value(), 
              solver_timer_3.value(),  
              solver_timer_4.value(),
              sum_1))
    print ("save time = %.3e, sum of assembly/solve = %.3e, time step = %.3e, lost time = %.3e" 
           % (save_timer.value(),
              (sum_0 + sum_1) * nl_its,
              loop_timer.value(),
              loop_timer.value() - (sum_0 + sum_1) * nl_its - save_timer.value()))
