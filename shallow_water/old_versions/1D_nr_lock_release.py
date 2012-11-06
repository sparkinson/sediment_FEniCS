from dolfin import *
from dolfin_tools import *
import mms_strings as mms
import numpy as np

############################################################
# DOLFIN SETTINGS

info(parameters, False)
set_log_active(True)

############################################################
# SIMULATION USER DEFINED PARAMETERS

# function spaces
shape = 1

# mesh
dX = 1e-3
L = 1.0

# save files
dt_store = 1e-1
q_file = File("results/q.pvd") 
h_file = File("results/h.pvd") 

############################################################
# SOLUTION

# # mms solutions
# u_s = Expression(mms.u_s(), degree = shape + 1)
# h_s = Expression(mms.h_s(), degree = shape + 1)
# q_s = Expression(mms.q_s(), degree = shape + 1)
# w_s = Expression((mms.h_s(), mms.q_s()), degree = shape + 1)

# # mms source terms
# Sh = Expression(mms.Sh_s(), degree = shape + 1)
# Sq = Expression(mms.Sq_s(), degree = shape + 1)

############################################################
# START CONDITIONS

class initial_condition(Expression):
    def eval(self, value, x):
        if x[0] < 0.05:
            value[0] = 1.05
        elif x[0] < 0.1:
            value[0] = 1.0 + 0.05 * (cos( pi*(x[0] - 0.05)/0.05 ) + 1.0)/2.0
        # elif x[0] < 0.6:
        #     value[0] = 1.0 + 0.01 * (1.0 - cos( 2*pi*(x[0] - 0.2)/0.2 ))
        else:
            value[0] = 1.0

h_s = initial_condition()

############################################################
# GEOMETRY

mesh = Interval(int(L/dX), 0.0, L)

############################################################
# FUNCTIONS

# H = FunctionSpace(mesh, "CG", shape)
# Q = FunctionSpace(mesh, "CG", shape)
# W = H*Q
# w = project(w_s, W)
# (h, q) = split(w)
# (v, z) = TestFunctions(W)

Q = FunctionSpace(mesh, "CG", shape)

h = dict([[i, project(h_s, Q)] for i in range(2)])
v = TestFunction(Q)

q = dict([[i, Function(Q)] for i in range(2)])
z = TestFunction(Q)

############################################################
# BC's

# H_ = W.sub(0)
# Q_ = W.sub(1)

# bch  = []#[DirichletBC(H_, h_s, "on_boundary")] #"near(x[0], 1.0)")]# || near(x[0], 1.0)")]
# bcq  = [DirichletBC(Q_, q_s, "on_boundary")] #"near(x[0], 0.0) || near(x[0], 1.0)")]
# bcw = bch + bcq

# bch  = [DirichletBC(Q, h_s, "on_boundary")]
bcq  = [DirichletBC(Q, 0.0, "on_boundary")]

############################################################
# EQUATIONS

# F = z*(q**2.0/h + 0.5*h**2.0).dx()*dx + v*q.dx()*dx - z*Sq*dx - v*Sh*dx
k = Constant(0.0005)
F_h = v*(h[0] - h[1])*dx + v*grad(q[0])*k*dx 
F_q = z*(q[0] - q[1])*dx + z*grad(q[0]**2.0/h[0] + 0.5*h[0]**2.0)*k*dx  
# F_h = v*(h[0] - h[1])*dx + v*grad(u[0]*h[0])*k*dx  
# F_u = z*(u[0] - u[1])*dx + z*u[0]*grad(u[0])*k*dx + z*grad(h[0])*k*dx

plot(q[0], rescale=False)
plot(h[0], rescale=False, interactive=True)

############################################################
# store initial conditions
q_file << q[0]
h_file << h[0]
    
ss = 1.0
# Iterate until solution is converged
while (ss > 1e-7):
    solve(F_q == 0.0, q[0], bcq)
    solve(F_h == 0.0, h[0])#, bch)
    
    dh = errornorm(h[1], h[0], norm_type="L2", degree=shape + 1)
    dq = errornorm(q[1], q[0], norm_type="L2", degree=shape + 1)
    ss = max(dh, dq)
    print ss

    h[1].assign(h[0])
    q[1].assign(q[0])

    plot(q[0], rescale=False)
    plot(h[0], rescale=False, interactive=False)
    
    q_file << q[0]
    h_file << h[0]

# (h, q) = w.split()
# Eh = errornorm(h, h_s, norm_type="L2", degree=shape + 1)
Eq = errornorm(q[1], q[0], norm_type="L2", degree=shape + 1)
print Eq





# dF = derivative(F, w)
# pde = NonlinearVariationalProblem(F, w, bcw, dF)
# solver = NonlinearVariationalSolver(pde)
# solver.parameters["newton_solver"]["maximum_iterations"] = 100
# solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
# solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
# solver.solve()
