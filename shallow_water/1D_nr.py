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
print_progress = False
info(parameters, False)
set_log_active(False)

############################################################
# SIMULATION USER DEFINED PARAMETERS

# function spaces
shape_Q = 1
shape_H = 1
shape_PSI = 1 

# output
dt_store = 1e-5
u_file = File("results/u.pvd") 
h_file = File("results/h.pvd") 
phi_file = File("results/phi.pvd")
phi_d_file = File("results/phi_d.pvd")

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
x_lg = 0.2

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
x_N = 1.0
x_N_ = dict([[str(i), Constant(x_N)] for i in range(2)])
x_N_prime_ = dict([[str(i), Constant(0.0)] for i in range(2)])

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

# non-dimensional
q_ = dict([[str(i), Function(V)] for i in range(2)])
h_ = dict([[str(i), interpolate(h_s, W)] for i in range(2)])
psi_ =  dict([[str(i), interpolate(psi_s, Z)] for i in range(2)])
psi_d_ =  dict([[str(i), Function(Z)] for i in range(2)])
X = Function(V)
X.vector()[:] = mesh.coordinates()[:,0]

# dimensional
u = Function(V)
h_d = Function(W)
phi = Function(Z)

############################################################
# define dirichlet boundary conditions

no_slip = Expression('0.0', degree = shape_Q + 1)
nose = Expression('Fr*pow(psi,0.5)*h', Fr = 1.19, psi = 1.0, h = 1.0, degree = shape_Q + 1)
bcq  = [DirichletBC(V, no_slip, "near(x[0], 0.0)"),
        DirichletBC(V, nose, "near(x[0], 1.0)")]

############################################################
# store initial conditions

u_file << u
h_file << h_d
phi_file << phi
t_store = dt_store*t_nd

list_timings()
timer = Timer("run time") 

############################################################
# define equations to be solved

def ta(vals):
    return theta*vals['1']+(1.0-theta)*vals['0']

def F_q(q, h, psi, k, x_N, x_N_prime):
    
    F_q = (v*(1./k)*(q['0'] - q['1'])*ta(h)*dx +
           v/x_N['0']*(
            (ta(q)*ta(q)/ta(h) + 0.5*ta(psi)*ta(h)).dx() 
            - X*x_N_prime['0']*ta(q).dx()
            )*dx
           )

    return F_q

def F_h(q, h, k, x_N, x_N_prime):    

    F_h = (w*(1./k)*(h['0'] - h['1'])*dx + 
           w/x_N['0']*(
            ta(q).dx() -
            X*x_N_prime['0']*ta(h).dx()
            )*dx
           )

    return F_h

def F_psi(q, h, psi, k, x_N, x_N_prime):

    F_psi = (z*(1./k)*(psi['0'] - psi['1'])*dx +
             z/x_N['0']*(
              (ta(q)*ta(psi)*ta(h)**-1.0).dx() -
              X*x_N_prime['0']*ta(psi).dx()
              )*dx + 
             z*beta*ta(psi)*ta(h)**-1.0*dx
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

        # Compute height
        if print_progress:
            info_blue('Computing current height')
        solve(F_h(q_, h_, k, x_N_, x_N_prime_) == 0, h_['0'])

        # Compute velocity
        if print_progress:
            info_blue('Computing velocity')
        solve(F_q(q_, h_, psi_, k, x_N_, x_N_prime_) == 0, q_['0'], bcq)

        # # Compute concentration
        # if print_progress:
        #     info_blue('Computing current concentration')
        # solve(F_psi(q_, h_, psi_, k, x_N_, x_N_prime_) == 0, psi_['0'])

        nl_it += 1
    
    # Store values for next timestep
    q_['1'].assign(q_['0'])
    h_['1'].assign(h_['0'])
    psi_['1'].assign(psi_['0'])
    
    x_N_prime = q_['0'].vector().array()[-1]/h_['0'].vector().array()[-1]
    x_N = x_N + x_N_prime*timestep
    x_N_['1'] = x_N_['0']
    x_N_prime_['1'] = x_N_prime_['0']
    x_N_['0'] = Constant(x_N)
    x_N_prime_['0'] = Constant(x_N_prime)  

    nose.psi = psi_['0'].vector().array()[-1]
    nose.h = h_['0'].vector().array()[-1]

    q_N = np.array([0.0])
    nose.eval(q_N, np.array([x_N, 0.0]))
    u_N = (q_N/h_['0'].vector().array()[-1])/u_nd

    t += timestep
    info_green("t = %.5f, timestep = %.2e, x_N = %.2e, u_N = %.2e, psi_N = %.2e, h_N = %.2e" % 
               (t/t_nd, timestep/t_nd, x_N*x_lg, u_N, psi_['0'].vector().array()[-1], h_['0'].vector().array()[-1]))

    u.vector()[:] = (q_['0'].vector().array()/h_['0'].vector().array())/u_nd
    h_d.vector()[:] = h_['0'].vector().array()/l_nd
    phi.vector()[:] = (psi_['0'].vector().array()/h_['0'].vector().array())*phi_0

    plot(h_d, rescale=False)    

    if t > t_store:
        u_file << u
        h_file << h_d
        phi_file << phi
        t_store += dt_store

list_timings()
timer.stop()
