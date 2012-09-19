from dolfin import *
from dolfin_tools import *
import mms_strings as mms
import numpy as np
import matplotlib.pyplot as plt

############################################################
# DOLFIN SETTINGS

info(parameters, False)
set_log_active(True)

############################################################
# TIME DISCRETISATION

class time_discretisation():
    
    def crank_nicholson(self, u):
        return 0.5*u[0] + 0.5*u[1]

    def backward_euler(self, u): #implicit
        return u[0]

    def forward_euler(self, u): #explicit
        return u[1]

    def calc(self, u):
        f = time_discretisation.crank_nicholson
        return f(self, u)

td = time_discretisation()

############################################################
# SIMULATION USER DEFINED PARAMETERS

# function spaces
shape = 2

# mesh
dX_ = 1e-2
L = 1.0

# save files
dt_store = 1e-1
q_file = File("results/q.pvd") 
h_file = File("results/h.pvd") 

# stabilisation
b_ = 0.3

# reduced gravity
phi_ = 9.81*0.0077

############################################################
# OTHER PARAMETERS

dX = Constant(dX_)

# non-dimensionalising values
x_N_ = 0.2
u_N_ = 0.0
x_N = dict([[i, Constant(x_N_)] for i in range(2)])
u_N = dict([[i, Constant(u_N_)] for i in range(2)])
q_N = np.array([0.0], dtype='d')

h_0 = 0.4
class initial_condition(Expression):
    def eval(self, value, x):
        # if x[0] < 0.95:
        #     value[0] = 1.05
        # else:
        #     value[0] = 1.0 + 0.05 * (cos( pi*(x[0] - 0.05)/0.05 ) + 1.0)/2.0
        value[0] = h_0

h_s = initial_condition()
phi = Constant(phi_)
b = Constant(b_)

############################################################
# GEOMETRY

mesh = Interval(int(L/dX_), 0.0, L)

############################################################
# FUNCTIONS

# H = FunctionSpace(mesh, "CG", shape)
# Q = FunctionSpace(mesh, "CG", shape)
# W = H*Q
# w = project(w_s, W)
# (h, q) = split(w)
# (v, z) = TestFunctions(W)

Q = FunctionSpace(mesh, "CG", shape)

h = dict([[i, interpolate(h_s, Q)] for i in range(2)])
v = TestFunction(Q)

q = dict([[i, Function(Q)] for i in range(2)])
z = TestFunction(Q)

X = interpolate(Expression('x[0]'), Q)

############################################################
# BC's

# H_ = W.sub(0)
# Q_ = W.sub(1)

# bch  = []#[DirichletBC(H_, h_s, "on_boundary")] #"near(x[0], 1.0)")]# || near(x[0], 1.0)")]
# bcq  = [DirichletBC(Q_, q_s, "on_boundary")] #"near(x[0], 0.0) || near(x[0], 1.0)")]
# bcw = bch + bcq

# bch  = [DirichletBC(Q, h_s, "on_boundary")]

no_slip = Expression('0.0', degree = shape + 1)
nose = Expression('Fr*pow(phi*h,0.5)*h', Fr = 1.19, phi = phi_, h = h_0, degree = shape + 1)
bcq  = [DirichletBC(Q, no_slip, "near(x[0], 0.0)"),
        DirichletBC(Q, nose, "near(x[0], 1.0)")]
bch = []

############################################################
# EQUATIONS (done each timestep)

timestep = 1./500.

############################################################
# PLOTTING STUFF

plot_x = np.linspace(0.0, 6.0, 1201)
plt.ion()
fig = plt.figure(figsize=(24, 12), dpi=50)
vel_plot = fig.add_subplot(211)
h_plot = fig.add_subplot(212)
vel_plot.set_autoscaley_on(False)
vel_plot.set_ylim([-0.1,0.4])
plot_freq = 0.01

def y_data(u):
    val_x = np.linspace(0.0, x_N_, L/dX_ + 1)
    val = u[:L/dX_ + 1]
    return np.interp(plot_x, val_x, val, right=0.0)

vel_line, = vel_plot.plot(plot_x, y_data(q[0].vector().array()/h[0].vector().array()), 'r-')
h_line, = h_plot.plot(plot_x, y_data(h[0].vector().array()), 'r-')
fig.canvas.draw()
fig.savefig('results/%06.2f.png' % (0.0))

plot_t = plot_freq

# plot(q[0], rescale=False)
# plot(h[0], rescale=False, interactive=True)

############################################################
# store initial conditions
# q_file << q[0]
# h_file << h[0]
    
t = 0.0
while (True):
    t += timestep
    k = Constant(timestep)

    ss = 1.0
    nl_its = 0
    while (ss > 1e-4):

        h_nl = h[0].copy(deepcopy=True)
        q_nl = q[0].copy(deepcopy=True)

        x_N_td = td.calc(x_N)
        u_N_td = td.calc(u_N)
        h_td = td.calc(h)
        q_td = td.calc(q)

        F_h = v*(x_N[0]*h[0] - x_N[1]*h[1])*dx + v*grad(q_td)*k*dx - v*X*u_N_td*grad(h_td)*k*dx
        F_q = z*(x_N[0]*q[0] - x_N[1]*q[1])*dx + z*grad(q_td**2.0/h_td + 0.5*phi*h_td**2.0)*k*dx - v*X*u_N_td*grad(q_td)*k*dx

        # stabilisation
        u = q_td/h_td
        alpha = b*dX*(abs(u)+u+h_td**0.5)*h_td
        F_q = F_q - v*1./x_N_td*grad((alpha*grad(u)))*k*dx

        dF = derivative(F_q, q[0])
        pde = NonlinearVariationalProblem(F_q, q[0], bcq, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 1000
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
        solver.solve()

        dF = derivative(F_h, h[0])
        pde = NonlinearVariationalProblem(F_h, h[0], bch, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 1000
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
        solver.solve()

        # solve(F_q == 0.0, q[0], bcq)
        # solve(F_h == 0.0, h[0])

        nose.h = h[0].vector().array()[-1]
        q_N_ = nose(Point(1.0))
        u_N_ = q_N_/h[0].vector().array()[-1]
        x_N_ = x_N_ + u_N_*timestep
        x_N[0] = Constant(x_N_)
        u_N[0] = Constant(u_N_)
        
        dh = errornorm(h_nl, h[0], norm_type="L2", degree=shape + 1)
        dq = errornorm(q_nl, q[0], norm_type="L2", degree=shape + 1)
        ss = max(dh, dq)
        print ss

        nl_its += 1

    h[1].assign(h[0])
    q[1].assign(q[0])
    x_N[1].assign(x_N[0])
    u_N[1].assign(u_N[0])

    timestep = 1./(10*((q[0].vector().array()/h[0].vector().array())/(x_N_*dX_)).max())

    if t > plot_t:
        vel_line.set_ydata(y_data(q[0].vector().array()/h[0].vector().array()))
        h_line.set_ydata(y_data(h[0].vector().array()))
        fig.canvas.draw()
        fig.savefig('results/%06.2f.png' % (t))
        plot_t += plot_freq

        info_blue("plotted")

    # plot(q[0], rescale=False)
    # plot(h[0], rescale=False, interactive=False)
    
    # q_file << q[0]
    # h_file << h[0]

    mass = (h[0].vector().array()[:L/dX_ + 1]*(x_N_*dX_)).sum()
    info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, h_N = %.2e, nl_its = %1d, mass = %.2e" % 
               (t, timestep, x_N_, u_N_, h[0].vector().array()[-1], nl_its, mass))
