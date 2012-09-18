from dolfin import *
from dolfin_tools import *
import mms_strings as mms
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

# time-loop
min_timestep = 1e-4
max_timestep = 1e-4
T = 5.0
CFL = 0.5
theta_ = 0.5
nl_its = 2

# mesh
dX = 1e-2
L = 1.0

# gravity
g = 1.0

############################################################
# INITIALISE DEPENDENT PARAMETERS

# non-dimensionalising values
x_N = 1.0
x_N_ = dict([[i, Constant(x_N)] for i in range(2)])
x_N_prime_ = dict([[i, Constant(0.0)] for i in range(2)])

# time-loop
timestep = min_timestep
theta = Constant(theta_)

# mms solutions
u_s = Expression(mms.u_s(), degree = shape_Q + 1)
h_s = Expression(mms.h_s(), degree = shape_H + 1)
q_s = Expression(mms.q_s(), degree = shape_Q + 1)
psi_s = Expression(mms.q_s(), degree = shape_PSI + 1)

# mms source terms
Sh = Expression(mms.Sh_s(), degree = shape_H + 1)
Sq = Expression(mms.Sq_s(), degree = shape_Q + 1)

############################################################
# generate geometry

mesh = Interval(int(L/dX), 0.0, L)

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
q_ = dict([[i, interpolate(q_s, V)] for i in range(2)])
h_ = dict([[i, interpolate(h_s, W)] for i in range(2)])
psi_ = dict([[i, interpolate(psi_s, Z)] for i in range(2)])
X = Function(V)
X.vector()[:] = mesh.coordinates()[:,0]

# functional
u = Function(V)

############################################################
# define dirichlet boundary conditions

bcq  = [DirichletBC(V, q_s, "near(x[0], 0.0) || near(x[0], 1.0)")]
bch  = [DirichletBC(W, h_s, "near(x[0], 0.0) || near(x[0], 1.0)")]

############################################################
# store initial conditions

u.vector()[:] = q_[0].vector().array()/h_[0].vector().array()

list_timings()
timer = Timer("run time") 

############################################################
# define equations to be solved

def ta(vals):
    return theta*vals[1]+(1.0-theta)*vals[0]

def F_h(q, h, k, x_N, x_N_prime):   

    F_h = (w*(1./k)*(h[0] - h[1])*dx + 
           (ta(q).dx())*w*dx -
           Sh*w*dx
           )

    return F_h

def F_q(q, h, psi, k, x_N, x_N_prime):

    F_q = (v*(1./k)*(q[0] - q[1])*ta(h)*dx -
           ta([q[0]**2.0, q[1]**2.0])/ta(h)*v*dx 
           - 0.5*g*ta(psi)*ta(h).dx()*v*dx
           - Sq*v*dx
           )

    return F_q

def F_complete_ss(q, h, psi):
    
    F = (v*( q**2.0/h 
         )

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
        solve(F_h(q_, h_, k, x_N_, x_N_prime_) == 0, h_[0])

        # Compute velocity
        if print_progress:
            info_blue('Computing velocity')
        solve(F_q(q_, h_, psi_, k, x_N_, x_N_prime_) == 0, q_[0], bcq)

        nl_it += 1
    
    u.vector()[:] = q_[0].vector().array()/h_[0].vector().array()
        
    # Store values for next timestep
    q_[1].assign(q_[0])
    h_[1].assign(h_[0])
    psi_[1].assign(psi_[0])

    t += timestep
    info_green("t = %.5f, timestep = %.2e" % 
               (t, timestep))

    plot(h_[0], rescale=False)    

list_timings()
timer.stop()
