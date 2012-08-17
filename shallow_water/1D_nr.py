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
shape_PHI = 1 

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
x_N = 1.0
x_N_ = dict([[str(i), Constant(x_N)] for i in range(2)])
x_N_prime_ = dict([[str(i), Constant(0.0)] for i in range(2)])

# set constants
beta = Constant(u_sink)

# initialise timestep
timestep = min_timestep

# generate expressions for initial conditions
class initial_condition(Expression):
    def __init__(self, c):
        self.c = c

    def eval(self, value, x):
        # if x[0] < 0.2:
        #     value[0] = self.c
        # else:
        #     value[0] = 0.0
        value[0] = self.c - x[0]*self.c

phi_s = initial_condition(phi_0)
h_s = initial_condition(h_0)

############################################################
# generate geometry

mesh = Interval(int(L/dX), 0.0, L)

############################################################
# generate function spaces

V = FunctionSpace(mesh, "CG", shape_Q)
W = FunctionSpace(mesh, "CG", shape_H)
Z = FunctionSpace(mesh, "CG", shape_PHI)
v = TestFunction(V)
w = TestFunction(W)
z = TestFunction(Z)

############################################################
# generate functions

# non-dimensional
q_ = dict([[str(i), Function(V)] for i in range(2)])
h_ = dict([[str(i), interpolate(h_s, W)] for i in range(2)])
phi_ =  dict([[str(i), interpolate(phi_s, Z)] for i in range(2)])
phi_d_ =  dict([[str(i), Function(Z)] for i in range(2)])
X = Function(V)
X.vector()[:] = mesh.coordinates()[:,0]

# dimensional
u = Function(V)

############################################################
# define dirichlet boundary conditions

no_slip = Expression('0.0', degree = shape_Q + 1)
nose = Expression('Fr*pow(phi,0.5)*h', Fr = 1.19, phi = 1.0, h = 1.0, degree = shape_Q + 1)
bcq  = [DirichletBC(V, no_slip, "near(x[0], 0.0) || near(x[0], 1.0)")]#,
        #DirichletBC(V, nose, "near(x[0], 1.0)")]

############################################################
# store initial conditions

u.vector()[:] = q_['0'].vector().array()/h_['0'].vector().array()

u_file << u
h_file << h_['0']
phi_file << phi_['0']
t_store = dt_store

list_timings()
timer = Timer("run time") 

############################################################
# define equations to be solved

def ta(vals):
    return theta*vals['1']+(1.0-theta)*vals['0']

def F_q(q, h, phi, k, x_N, x_N_prime):
    
    F_q = (v*(1./k)*(q['0'] - q['1'])*ta(h)*dx -
           #v/x_N['0']*
            v*((ta(q)*ta(q)/ta(h) + 0.5*R*g*ta(phi)*ta(h)**2).dx() 
            #- X*x_N_prime['0']*ta(q).dx()
            )*dx
           )

    return F_q

def F_h(q, h, k, x_N, x_N_prime):    

    F_h = (w*(1./k)*(h['0'] - h['1'])*dx + 
           #w/x_N['0']*
            w*(ta(q).dx()
            #- X*x_N_prime['0']*ta(h).dx()
            )*dx
           )

    return F_h

def F_phi(q, h, phi, k, x_N, x_N_prime):

    F_phi = (z*(1./k)*(phi['0'] - phi['1'])*dx +
             #z/x_N['0']*
              z*((ta(q)*ta(phi)*ta(h)**-1.0).dx()
              #- X*x_N_prime['0']*ta(phi).dx()
              )*dx + 
             z*beta*ta(phi)*ta(h)**-1.0*dx
             )

    return F_phi

############################################################
# time-loop

list_timings()
timer = Timer("run time") 

t = timestep
while t < T:

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
        solve(F_q(q_, h_, phi_, k, x_N_, x_N_prime_) == 0, q_['0'], bcq)

        # # Compute concentration
        # if print_progress:
        #     info_blue('Computing current concentration')
        # solve(F_phi(q_, h_, phi_, k, x_N_, x_N_prime_) == 0, phi_['0'])

        nl_it += 1
    
    # Store values for next timestep
    q_['1'].assign(q_['0'])
    h_['1'].assign(h_['0'])
    phi_['1'].assign(phi_['0'])
    
    x_N_prime = q_['0'].vector().array()[-1]/h_['0'].vector().array()[-1]
    x_N = x_N + x_N_prime*timestep
    x_N_['1'] = x_N_['0']
    x_N_prime_['1'] = x_N_prime_['0']
    x_N_['0'] = Constant(x_N)
    x_N_prime_['0'] = Constant(x_N_prime)  

    nose.phi = phi_['0'].vector().array()[-1]
    nose.h = h_['0'].vector().array()[-1]

    q_N = np.array([0.0])
    nose.eval(q_N, np.array([x_N, 0.0]))
    u_N = (q_N/h_['0'].vector().array()[-1])

    t += timestep
    info_green("t = %.5f, timestep = %.2e, x_N = %.2e, u_N = %.2e, phi_N = %.2e, h_N = %.2e" % 
               (t, timestep, x_N*x_lg, u_N, phi_['0'].vector().array()[-1], h_['0'].vector().array()[-1]))

    u.vector()[:] = q_['0'].vector().array()/h_['0'].vector().array()

    plot(h_['0'], rescale=False)    

    if t > t_store:
        u_file << u
        h_file << h_['0']
        phi_file << phi_['0']
        t_store += dt_store

list_timings()
timer.stop()
