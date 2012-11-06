import sys
from dolfin import *
from dolfin_tools import *
import mms_strings as mms
import numpy as np
import matplotlib.pyplot as plt

############################################################
# DOLFIN SETTINGS

info(parameters, False)
set_log_active(False)

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
shape = 1

# mesh
dX_ = 5e-2
L = 1.0

# save files
dt_store = 1e-1
q_file = File("results/q.pvd") 
h_file = File("results/h.pvd") 

# stabilisation
b_ = 0.3

# current properties
g_prime_ = 0.81 # 9.81*0.0077
h_0 = 20.0
x_N_ = 10.0
Fr_ = 1.19

# time step
timestep = 5.0e-2 #1./1000.

############################################################
# OTHER PARAMETERS

dX = Constant(dX_)

class initial_condition(Expression):
    def eval(self, value, x):
        value[0] = h_0

h_s = initial_condition()
x_N_s = Expression(str(x_N_))
g_prime = Constant(g_prime_)
b = Constant(b_)
Fr = Constant(Fr_)

############################################################
# GEOMETRY

mesh = Interval(int(L/dX_), 0.0, L)
n = FacetNormal(mesh)

############################################################
# FUNCTIONS

Q = FunctionSpace(mesh, "CG", shape)
R = FunctionSpace(mesh, "R", 0)
v = TestFunction(Q)
r = TestFunction(R)

h = dict([[i, interpolate(h_s, Q)] for i in range(2)])
q = dict([[i, Function(Q)] for i in range(2)])
x_N = dict([[i, project(x_N_s, R)] for i in range(2)])
u_N = dict([[i, Function(R)] for i in range(2)])

X = interpolate(Expression('x[0]'), Q)

############################################################
# BC's

no_slip = Expression('0.0', degree = shape + 1)
bcq = [DirichletBC(Q, no_slip, "near(x[0], 0.0) && on_boundary")]
bch = []

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 + DOLFIN_EPS and on_boundary

left_boundary = LeftBoundary()
exterior_facet_domains = FacetFunction("uint", mesh)
exterior_facet_domains.set_all(1)
left_boundary.mark(exterior_facet_domains, 0)

ds = Measure("ds")[exterior_facet_domains]

############################################################
# PLOTTING STUFF

plot_x = np.linspace(0.0, 100.0, 10001)
plt.ion()
fig = plt.figure(figsize=(26, 5), dpi=50)
vel_plot = fig.add_subplot(211)
h_plot = fig.add_subplot(212)
h_plot.set_autoscaley_on(False)
h_plot.set_ylim([0.0,25.0])
vel_plot.set_autoscaley_on(False)
vel_plot.set_ylim([0.0,5.0])
plot_freq = 100000.0

def y_data(u):
    val_x = np.linspace(0.0, x_N[0].vector().array()[-1], L/dX_ + 1)
    val = u[:L/dX_ + 1]
    return np.interp(plot_x, val_x, val, right=0.0)

vel_line, = vel_plot.plot(plot_x, y_data(q[0].vector().array()/h[0].vector().array()), 'r-')
h_line, = h_plot.plot(plot_x, y_data(h[0].vector().array()), 'r-')
fig.canvas.draw()
# fig.savefig('results/%06.2f.png' % (0.0))

plot_t = plot_freq
    
t = 0.0
while (True):
    t += timestep
    k = Constant(timestep)

    ss = 1.0
    nl_its = 0
    while (nl_its < 2 or ss > 1e-4):

        # VALUES FOR CONVERGENCE TEST
        h_nl = h[0].copy(deepcopy=True)
        q_nl = q[0].copy(deepcopy=True)
        x_N_nl = x_N[0].copy(deepcopy=True)

        # DEFINE EQUATIONS
        # time discretisation of values
        x_N_td = td.calc(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = td.calc(u_N)
        h_td = td.calc(h)
        q_td = td.calc(q)

        # momentum
        F_q = (v*(q[0] - q[1])*dx +
               inv_x_N*grad(v*X)*u_N_td*q_td*k*dx +
               inv_x_N*v*grad(q_td**2.0/h_td + 0.5*g_prime*h_td**2.0)*k*dx -
               inv_x_N*v*X*u_N_td**2.0*h_td*n*k*ds(1))
        # momentum stabilisation
        u = q_td/h_td
        alpha = b*dX*(abs(u)+u+h_td**0.5)*h_td
        F_q = F_q + inv_x_N*grad(v)*alpha*grad(u)*k*dx

        # conservation
        F_h = v*(h[0] - h[1])*dx + \
            inv_x_N*v*grad(q_td)*k*dx - \
            inv_x_N*v*X*u_N_td*grad(h_td)*k*dx

        # nose location/speed
        F_u_N = r*(Fr*(g_prime*h_td)**0.5)*ds(1) - r*u_N[0]*ds(1)
        F_x_N = r*(x_N[0] - x_N[1])*dx - r*u_N_td*k*dx

        # SOLVE EQUATIONS
        dF = derivative(F_q, q[0])
        pde = NonlinearVariationalProblem(F_q, q[0], bcq, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()

        dF = derivative(F_h, h[0])
        pde = NonlinearVariationalProblem(F_h, h[0], bch, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()

        solve(F_u_N == 0, u_N[0])
        solve(F_x_N == 0, x_N[0])
        
        dh = errornorm(h_nl, h[0], norm_type="L2", degree=shape + 1)
        dq = errornorm(q_nl, q[0], norm_type="L2", degree=shape + 1)
        dx_N = errornorm(x_N_nl, x_N[0], norm_type="L2", degree=shape + 1)
        ss = max(dh, dq, dx_N)

        nl_its += 1

    h[1].assign(h[0])
    q[1].assign(q[0])
    x_N[1].assign(x_N[0])
    u_N[1].assign(u_N[0])

    vel_line.set_ydata(y_data(q[0].vector().array()/h[0].vector().array()))
    h_line.set_ydata(y_data(h[0].vector().array()))
    fig.canvas.draw()
    # if t > plot_t:
        # fig.savefig('results/%06.2f.png' % (t))
        # plot_t += plot_freq
        # info_blue("plotted")

    mass = (h[0].vector().array()[:L/dX_ + 1]*(x_N[0].vector().array()[-1]*dX_)).sum()
    info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, u_N_2 = %.2e, h_N = %.2e, nl_its = %1d, mass = %.2e" % 
               (t, timestep, 
                x_N[0].vector().array()[-1], 
                u_N[0].vector().array()[0], 
                q[0].vector().array()[-1]/h[0].vector().array()[-1], 
                h[0].vector().array()[-1],
                nl_its, 
                mass))
