from dolfin import *
from dolfin_tools import *
import numpy as np

############################################################
# DOLFIN SETTINGS

parameters["linear_algebra_backend"]             = "PETSc"
parameters["form_compiler"]["cpp_optimize"]      = True
parameters["form_compiler"]["quadrature_degree"] = 8
parameters["std_out_all_processes"]              = False;

# show information
print_time = True
print_progress = True
info(parameters, False)
set_log_active(True)

############################################################
# SIMULATION USER DEFINED PARAMETERS

# function spaces
shape_Q = 1
shape_H = 1
shape_PSI = 1 

# output
dt_store = 1e-1
q_file = File("results/q.pvd") 
h_file = File("results/h.pvd") 
psi_file = File("results/psi.pvd")
psi_d_file = File("results/psi_d.pvd")

# time-loop
min_timestep = 1e-2
max_timestep = 1e-2
T = 5.0
CFL = 0.5
theta = 0.5
nl_its = 2

# sediment
R = 2.217
phi_0 = 3.49e-03
h_0 = 0.4
u_sink = 5.5e-4
x_N_ = 0.2

# mesh
dX = 1e-2
L = 1.0

# gravity
g = 9.81

############################################################
# INITIALISE DEPENDENT PARAMETERS

# non-dimensionalising values
l_nd = 1./h_0
g_0 = R*phi_0
t_nd = 1./(h_0/g_0)**0.5
u_nd = 1./(g_0*h_0)**0.5
x_N = Constant(x_N_)
x_N_prime = Constant(0.0)

# set constants
beta = Constant(u_sink*u_nd)

# initialise timestep
timestep = min_timestep*t_nd

# generate expressions for initial conditions
class initial_condition(Expression):
    def __init__(self, c):
        self.c = c

    def eval(self, value, x):
        # if x[0] < 0.2:
        #     value[0] = self.c
        # else:
        #     value[0] = 0.0
        value[0] = self.c

psi_s = initial_condition(1.0)
h_s = initial_condition(1.0)

############################################################
# generate geometry

L_nd = L*l_nd
dX_nd = dX*l_nd
mesh = Interval(int(L_nd/dX_nd), 0.0, 1.0)

############################################################
# generate function spaces

V = FunctionSpace(mesh, "CG", shape_Q)
W = FunctionSpace(mesh, "CG", shape_H)
Z = FunctionSpace(mesh, "CG", shape_PSI)
v = TestFunction(V)
w = TestFunction(W)
z = TestFunction(Z)

############################################################
# generate functions

q_ = dict([[str(i), Function(V)] for i in range(2)])
h_ = dict([[str(i), interpolate(h_s, W)] for i in range(2)])
psi_ =  dict([[str(i), interpolate(psi_s, Z)] for i in range(2)])
psi_d_ =  dict([[str(i), Function(Z)] for i in range(2)])
X = Function(V)
X.vector()[:] = mesh.coordinates()[:,0]

############################################################
# define dirichlet boundary conditions

no_slip = Expression('0.0', degree = shape_Q + 1)
nose = Expression('Fr*pow(psi,0.5)*h', Fr=1.19, psi = 1.0, h = 1.0, degree = shape_Q + 1)
bcq  = [DirichletBC(V, no_slip, "near(x[0], 0.0)"),
        DirichletBC(V, nose, "near(x[0], 1.0)")]

############################################################
# store initial conditions

q_file << q_['0']
h_file << h_['0']
psi_file << psi_['0']
psi_d_file << psi_d_['0']
t_store = dt_store*t_nd

list_timings()
timer = Timer("run time") 

############################################################
# define equations to be solved

def F_q(q__, h__, psi__, k__, two_layer):
    if two_layer:
        tl = 1.0
    else:
        tl = 0.0

    q_ta = theta*q__['1']+(1.0-theta)*q__['0']
    psi_ta = theta*psi__['1']+(1.0-theta)*psi__['0']
    h_ta = theta*h__['1']+(1.0-theta)*h__['0']
    
    F_q = (v*(1./k__)*(q__['0'] - q__['1'])*h_ta*dx + 
           x_N**-1.0*(
            v*(q_ta*q_ta*h_ta**-1.0 + 0.5*psi_ta*h_ta).dx()*dx -
            v*X*x_N_prime*q_ta.dx()*dx
            )
           )

    return F_q

def F_h(q__, h__, k__):
    q_ta = theta*q__['1']+(1.0-theta)*q__['0']
    h_ta = theta*h__['1']+(1.0-theta)*h__['0']
    
    F_h = (w*(1./k__)*(h__['0'] - h__['1'])*dx + 
           x_N**-1.0*(
            w*q_ta.dx()*dx -
            w*X*x_N_prime*h_ta.dx()*dx
            )
           )

    return F_h

def F_psi(q__, h__, psi__, k__):
    q_ta = theta*q__['1']+(1.0-theta)*q__['0']
    h_ta = theta*h__['1']+(1.0-theta)*h__['0']
    psi_ta = theta*psi__['1']+(1.0-theta)*psi__['0']
    
    F_psi = (z*(1./k__)*(psi__['0'] - psi__['1'])*dx +
             x_N**-1.0*(
              z*(q_ta*psi_ta*h_ta**-1.0).dx()*dx -
              z*X*x_N_prime*psi_ta.dx()*dx
              ) + 
             z*beta*psi_ta*h_ta**-1.0*dx
             )

    return F_psi

############################################################
# time-loop

list_timings()
timer = Timer("run time") 

t = timestep
while t < T*t_nd:

    k = Constant(timestep)
    inv_k = 1.0/timestep

    nl_it = 0
    
    while nl_it < nl_its:

        # Compute velocity
        if print_progress:
            info_blue('Computing velocity')
        solve(F_q(q_, h_, psi_, k, False) == 0, q_['0'], bcq)

        # Compute height
        if print_progress:
            info_blue('Computing current height')
        solve(F_h(q_, h_, k) == 0, h_['0'])

        # Compute concentration
        if print_progress:
            info_blue('Computing current concentration')
        solve(F_psi(q_, h_, psi_, k) == 0, psi_['0'])

        nl_it += 1

    if t > t_store:
        q_file << q_['0']
        h_file << h_['0']
        psi_file << psi_['0']
        t_store += dt_store
    
    # Store values for next timestep
    q_['1'].assign(q_['0'])
    h_['1'].assign(h_['0'])
    psi_['1'].assign(psi_['0'])

    plot(h_['0'], rescale=False)
    
    x_N_prime_ = q_['0'].vector().array()[-1]/h_['0'].vector().array()[-1]
    x_N_ = x_N_ + x_N_prime_*timestep
    x_N = Constant(x_N_)
    x_N_prime = Constant(x_N_prime_)  

    t += timestep
    info_green("t = %.5f, timestep = %.2e" % (t/t_nd, timestep/t_nd))

list_timings()
timer.stop()
