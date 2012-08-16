from dolfin import *
from dolfin_adjoint import *
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
set_log_active(False)

############################################################
# SIMULATION USER DEFINED PARAMETERS

# function spaces
shape_U = 1
shape_H = 1
shape_PHI = 1 

# output
dt_store = 1e-1
u_file = File("results/u.pvd") 
h_file = File("results/h.pvd") 
phi_file = File("results/phi.pvd")
phi_d_file = File("results/phi_d.pvd")

# time-loop
min_timestep = 1.0
max_timestep = 1.0
T = 5
CFL = 0.5
theta = 0.5
nl_its = 2

# sediment
R = 2.217
phi_0 = 3.49e-03
h_0 = 0.4
u_sink_ = 5.5e-4

# mesh
dX = 2e-3
L = 1.0

############################################################
# INITIALISE DEPENDENT PARAMETERS

# set constants
u_sink = Constant(u_sink_)

# initialise timestep
timestep = min_timestep

# generate expressions for initial conditions
class initial_condition(Expression):
    def __init__(self, c):
        self.c = c

    def eval(self, value, x):
        if x[0] < 0.2:
            value[0] = self.c
        else:
            value[0] = 0.0

phi_s = initial_condition(phi_0)
h_s = initial_condition(h_0)

############################################################
# generate geometry

mesh = Interval(int(L/dX), 0.0, L)

n = FacetNormal(mesh)
h = CellSize(mesh)

############################################################
# generate function spaces

V = FunctionSpace(mesh, "CG", shape_U)
Q = FunctionSpace(mesh, "CG", shape_H)
Z = FunctionSpace(mesh, "CG", shape_PHI)
v = TestFunction(V)
q = TestFunction(Q)
z = TestFunction(Z)

############################################################
# generate functions

u_ = dict([[str(i), Function(V)] for i in range(2)])
# u_['star'] = Function(V)
h_ = dict([[str(i), interpolate(h_s, Q)] for i in range(2)])
phi_ =  dict([[str(i), interpolate(phi_s, Z)] for i in range(2)])
phi_d_ =  dict([[str(i), Function(Z)] for i in range(2)])

############################################################
# define dirichlet boundary conditions

no_slip = Expression('0.0', degree = shape_U + 1)
bcu  = [DirichletBC(V, no_slip, "near(x[0], 0.0)")]

############################################################
# store initial conditions

u_file << u_['0']
h_file << h_['0']
phi_file << phi_['0']
phi_d_file << phi_d_['0']
t_store = dt_store

list_timings()
timer = Timer("run time") 

############################################################
# define equations to be solved

def F_u(u__, h__, phi__, k__):
    u_ta = theta*u__['1']+(1.0-theta)*u__['0']
    # u_nl = theta*u__['1']+(1.0-theta)*u__['star']
    phi_ta = theta*phi__['1']+(1.0-theta)*phi__['0']
    h_ta = theta*h__['1']+(1.0-theta)*h__['0']
    
    F_u = v*(1./k__)*(u__['0'] - u__['1'])*h_ta*dx + \
        v*(u_ta*u_ta*h_ta + 0.5*phi_ta*h_ta**2).dx()*dx

    return F_u

def F_h(u__, h__, k__):
    u_ta = theta*u__['1']+(1.0-theta)*u__['0']
    h_ta = theta*h__['1']+(1.0-theta)*h__['0']
    
    F_h = q*(1./k__)*(h__['0'] - h__['1'])*dx + \
        q*(u_ta*h_ta).dx()*dx

    return F_h

def F_phi(u__, h__, phi__, k__):
    u_ta = theta*u__['1']+(1.0-theta)*u__['0']
    h_ta = theta*h__['1']+(1.0-theta)*h__['0']
    phi_ta = theta*phi__['1']+(1.0-theta)*phi__['0']
    
    F_h = z*(1./k__)*(phi__['0'] - phi__['1'])*dx + \
        z*u_ta*phi_ta.dx()*dx + z*u_sink*phi_ta*h_ta**-1.0*dx

    return F_h

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

        # Compute velocity
        if print_progress:
            info_blue('Computing velocity')
        solve(F_u(u_, h_, phi_, k) == 0, u_['0'], bcu)

        # Compute height
        if print_progress:
            info_blue('Computing current height')
        solve(F_h(u_, h_, k) == 0, h_['0'])

        # Compute concentration
        if print_progress:
            info_blue('Computing current concentration')
        solve(F_h(u_, h_, phi_, k) == 0, phi_['0'])

        nl_it += 1

    if t > t_store:
        u_file << u_['0']
        h_file << h_['0']
        phi_file << phi_['0']
        t_store += dt_store
    
    # Store values for next timestep
    u_['1'].assign(u_['0'])
    h_['1'].assign(h_['0'])
    phi_['1'].assign(phi_['0'])

    ############################################################
    # Adaptive time step
    timestep = np.ma.fix_invalid(dX/abs(u_['0'].vector().array()))
    timestep = MPI.min(CFL*timestep.min())
    timestep = max(timestep, min_timestep)
    timestep = min(timestep, max_timestep)

    t += timestep
    info_green("t = %.5f, timestep = %.2e, picard iterations =" % (t, timestep) + str(picard_its_store) + ", Eu = %.3e" % Eu)

list_timings()
timer.stop()
