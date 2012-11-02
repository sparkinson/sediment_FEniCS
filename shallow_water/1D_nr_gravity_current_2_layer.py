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
dX_ = 1e-2
L = 1.0

# save files
dt_store = 1e-1
q_file = File("results/q.pvd") 
h_file = File("results/h.pvd") 

# stabilisation
b_ = 0.5

# current size
g_prime_ = 0.81 # 9.81*0.0077
h_0 = 20.0
H_ = 30.0
x_N__ = 10.0

############################################################
# OTHER PARAMETERS

dX = Constant(dX_)

x_N_ = dict([[i, x_N__] for i in range(2)])
u_N_ = dict([[i, 0.0] for i in range(2)])
x_N = dict([[i, Constant(x_N_[i])] for i in range(2)])
u_N = dict([[i, Constant(u_N_[i])] for i in range(2)])
q_N = np.array([0.0], dtype='d')

class initial_condition(Expression):
    def eval(self, value, x):
        value[0] = h_0

h_s = initial_condition()
g_prime = Constant(g_prime_)
b = Constant(b_)
H = Constant(H_)

############################################################
# GEOMETRY

mesh = Interval(int(L/dX_), 0.0, L)
n = FacetNormal(mesh)

############################################################
# FUNCTIONS

Q = FunctionSpace(mesh, "CG", shape)

h = dict([[i, interpolate(h_s, Q)] for i in range(2)])
v = TestFunction(Q)

q = dict([[i, Function(Q)] for i in range(2)])
z = TestFunction(Q)

X = interpolate(Expression('x[0]'), Q)

############################################################
# BC's

no_slip = Expression('0.0', degree = shape + 1)
nose = Expression('Fr*pow(g_prime*h, 0.5)*h', Fr = 0.5*(h_0/H_)**-0.5, g_prime = g_prime_, h = h_0, degree = shape + 1)
# nose = Expression('Fr*pow(g_prime*h, 0.5)*h', Fr = 1.19, g_prime = g_prime_, h = h_0, degree = shape + 1)
bcq  = [DirichletBC(Q, no_slip, "near(x[0], 0.0)"),
        DirichletBC(Q, nose, "near(x[0], 1.0)")]
bch = []

############################################################
# EQUATIONS (done each timestep)

timestep = 5.0e-4 #1./1000.

############################################################
# PLOTTING STUFF

# plot_x = np.linspace(0.0, 11.0, 10001)
# plt.ion()
# fig = plt.figure(figsize=(26, 5), dpi=50)
# vel_plot = fig.add_subplot(211)
# h_plot = fig.add_subplot(212)
# h_plot.set_autoscaley_on(False)
# h_plot.set_ylim([0.0,22.0])
# vel_plot.set_autoscaley_on(False)
# vel_plot.set_ylim([0.0,10.0])
# plot_freq = 100000.0

# def y_data(u):
#     val_x = np.linspace(0.0, x_N_[0], L/dX_ + 1)
#     val = u[:L/dX_ + 1]
#     return np.interp(plot_x, val_x, val, right=0.0)

# vel_line, = vel_plot.plot(plot_x, y_data(q[0].vector().array()/h[0].vector().array()), 'r-')
# h_line, = h_plot.plot(plot_x, y_data(h[0].vector().array()), 'r-')
# fig.canvas.draw()
# fig.savefig('results/%06.2f.png' % (0.0))

# plot_t = plot_freq

plot(q[0], rescale=False)
plot(h[0], rescale=False, interactive=True)
    
t = 0.0
while (True):
    t += timestep
    k = Constant(timestep)

    ss = 1.0
    nl_its = 0
    shallow_Fr = True
    while (nl_its < 2 or ss > 1e-4):

        h_nl = h[0].copy(deepcopy=True)
        q_nl = q[0].copy(deepcopy=True)

        x_N_td = td.calc(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = td.calc(u_N)
        h_td = td.calc(h)
        q_td = td.calc(q)

        F_h = v*(h[0] - h[1])*dx + \
            inv_x_N*v*grad(q_td)*k*dx - \
            inv_x_N*v*X*u_N_td*grad(h_td)*k*dx
        F_q = z*(q[0] - q[1])*dx + \
            (H - h_td)*inv_x_N*z*grad(q_td**2.0/h_td + 0.5*g_prime*h_td**2.0)*k*dx + \
            h_td*inv_x_N*z*grad(q_td*(H - h_td))*k*dx - \
            inv_x_N*v*X*u_N_td*grad(q_td)*k*dx

        # stabilisation
        u = q_td/h_td
        alpha = b*dX*(abs(u)+u+h_td**0.5)*h_td
        F_q = F_q + inv_x_N*grad(v)*alpha*grad(u)*k*dx

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

        # nose.h = h[0].vector().array()[-1]
        # if nose.h/H_ > 0.075 and shallow_Fr:
        #     nose.Fr = 0.5*(min(h[0].vector().array()[-1],0.1)/H_)**-0.5
        # else:
        #     nose.Fr = 1.19
        #     shallow_Fr = False
        try:
            nose.Fr = 0.5*(h[0].vector().array(),0.1)/H_)**-0.5
        except:
            nose.Fr = 

        q_N_ = nose(Point(1.0))
        u_N_[0] = q_N_/h[0].vector().array()[-1]
        x_N_[0] = x_N_[1] + u_N_[0]*timestep
        x_N[0] = Constant(x_N_[0])
        u_N[0] = Constant(u_N_[0])
        
        dh = errornorm(h_nl, h[0], norm_type="L2", degree=shape + 1)
        dq = errornorm(q_nl, q[0], norm_type="L2", degree=shape + 1)
        ss = max(dh, dq)
        # print ss

        nl_its += 1

    # DH = np.abs(h[0].vector().array() - h[1].vector().array()).max()/timestep
    # DX = (q[0].vector().array()/h[0].vector().array()).max()
    # CFL = 0.5
    # t1 = 10000.0 # CFL*(h[0].vector().array().max()*dX_)/DH
    # t2 = CFL*(x_N_[0]*dX_)/DX
    # # info_red("%.2e, %.2e, %.2e, %.2e" % (t1, t2, DH, DX))
    # timestep = min(t1, t2, plot_freq, timestep + 0.05*timestep)

    h[1].assign(h[0])
    q[1].assign(q[0])
    x_N_[1] = x_N_[0]
    u_N_[1] = u_N_[0]
    x_N[1].assign(x_N[0])
    u_N[1].assign(u_N[0])

    # vel_line.set_ydata(y_data(q[0].vector().array()/h[0].vector().array()))
    # h_line.set_ydata(y_data(h[0].vector().array()))
    # fig.canvas.draw()
    # if t > plot_t:
    #     fig.savefig('results/%06.2f.png' % (t))
    #     plot_t += plot_freq
    #     info_blue("plotted")

    plot(q[0], rescale=False)
    plot(h[0], rescale=False, interactive=False)
    
    # q_file << q[0]
    # h_file << h[0]

    mass = (h[0].vector().array()[:L/dX_ + 1]*(x_N_[0]*dX_)).sum()
    info_green("t = %.2e, timestep = %.2e, x_N = %.2e, u_N = %.2e, h_N = %.2e, nl_its = %1d, mass = %.2e" % 
               (t, timestep, x_N_[0], u_N_[0], h[0].vector().array()[-1], nl_its, mass))
