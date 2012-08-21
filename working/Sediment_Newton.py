from dolfin import *
import numpy as np
import Sediment_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# simulation parameters
dt_store = 0.1
shape_U = 1
shape_P = 1
shape_C = 2 
picard_tol = 1e-7
min_dt = 5e-3
max_dt = 5e-2
nl_its = 3
dX = 2.5e-2
T = 15.0
nu_scale_ = 0.5

def Calc_timestep(u, h):
    dt = np.ma.fix_invalid(dX/abs(u.vector().array()))
    print dt
    dt = dt.min()
    print dt
    dt = max(dt, min_dt)
    dt = min(dt, max_dt)
    print dt
    return dt

dt = max_dt

# show information
print_t = True
print_log = True
info(parameters, False)
set_log_active(print_log)

# set constants
nu = Constant(1e-6)
theta = Constant(1.0)
kappa = Constant(0.0)
nu_scale = Constant(nu_scale_)

# save files
u_file = File("results/u.pvd") 
p_file = File("results/p.pvd") 
c_file = File("results/c.pvd")

# generate expressions for initial conditions, boundary conditions and source terms
# u_s = Expression(('0.0', '0.0'), degree = shape_U + 1)
# # p_s = Expression('-floor(1.2 - x[0])*(0.02*9.81)*(x[1]-0.2)', degree = shape_P + 1)
# p_s = Expression('0.0', degree = shape_P + 1)
w_s = Expression(('0.0', '0.0', '0.0'), degree = shape_U + 1)
c_s = Expression('floor(1.2 - x[0])*0.007737', degree = shape_C + 1)
Mf = Expression(('0.0', '0.0'), degree = shape_U + 1)
Af = Expression('0.0', degree = shape_C + 1)

# define gravity
g_s = Expression(('0.0', '-9.81'), degree = shape_U + 1)

############################################################
# generate geometry

mesh = Rectangle(0.,0.,1.0,0.4,int(1.0/dX),int(0.4/dX),'right')
n = FacetNormal(mesh)
h = CellSize(mesh)

############################################################
# generate function spaces and functions

# MOMENTUM & CONSERVATION
V = VectorFunctionSpace(mesh, "CG", shape_U)
Q = FunctionSpace(mesh, "CG", shape_P)
W = V*Q
(v, q) = TestFunctions(W)
w_0 = project(w_s, W)
(u_0, p_0) = split(w_0)
w_1 = project(w_s, W)
(u_1, p_1) = w_1.split()

g = project(g_s, V)

# ADVECTION
D = FunctionSpace(mesh, "CG", shape_C)
c = TrialFunction(D)   
d = TestFunction(D) 
c_0 = project(c_s, D)
c_1 = project(c_s, D)
beta = Function(V)

############################################################
# define equations to be solved
k = Constant(dt)

# time-averaged values
u_0_ta = theta*u_0+(1.0-theta)*u_1
# u_ta = theta*u_0+(1.0-theta)*u_1
# c_0_ta = theta*c_0+(1.0-theta)*c_1

# # MOMENTUM & CONSERVATION

# F = ((inner(v, u_0)
#       + k*inner(grad(u_ta)*u_ta, v)
#       + k*nu*inner(grad(u_ta), grad(v))
#       - k*div(u_ta)*q
#       - k*div(v)*p_0
#       - inner(v, u_1)
#       - k*inner(v, Mf)
#       - k*inner(v, g*c_0_ta)
#       )*dx
#      + k*p_0*inner(v, n)*ds
#      )

# ADVECTION-DIFFUSION

# define equations to be solved   
a_c = (d*c*dx 
       + theta*k*d*inner(u_0_ta, grad(c))*dx
       + theta*k*inner(grad(d), kappa*grad(c))*dx 
       )
L_c = (d*c_1*dx 
       - (1.0-theta)*k*d*inner(u_0_ta, grad(c_1))*dx
       - (1.0-theta)*k*inner(grad(d), kappa*grad(c_1))*dx 
       + d*Af*dx
       )

# stabilisation
vnorm = sqrt(inner(u_0_ta, u_0_ta))

# SU stabilisation
stab = nu_scale*h/vnorm*inner(u_0_ta, grad(d))*inner(u_0_ta, grad(c))*dx

# Apply stabilistaion
a_c += stab

############################################################
# define dirichlet boundary conditions

V_ = W.sub(0)
Q_ = W.sub(1)
free_slip = Expression('0.0', degree = shape_U + 1)
no_slip = Expression(('0.0', '0.0'), degree = shape_U + 1)
bcu  = [DirichletBC(V_, no_slip, "on_boundary && !(near(x[1], 0.4))"), 
        DirichletBC(V_.sub(1), free_slip, "on_boundary && near(x[1], 0.4)")]
bcp = [DirichletBC(Q_, 0.0, "on_boundary && near(x[0], 1.0) && x[1] > 0.37")]
bcw = bcu+bcp
bcc  = []

############################################################
# store initial conditions

u_file << u_1
p_file << p_1
c_file << c_1
t_store = dt_store

############################################################
# time-loop

t = dt
while t < T:

    nl_it = 0
    while nl_it < nl_its:

        ############################################################
        # solve equations

        # MOMENTUM & CONSERVATION

        if print_log:
            print 'Momentum and conservation'

        u_ta = theta*u_0+(1.0-theta)*u_1
        c_0_ta = theta*c_0+(1.0-theta)*c_1
        
        F = ((inner(v, u_0)
              + k*inner(grad(u_ta)*u_ta, v)
              + k*nu*inner(grad(u_ta), grad(v))
              - k*div(u_ta)*q
              - k*div(v)*p_0
              - inner(v, u_1)
              #- k*inner(v, Mf)
              - k*inner(v, g*c_0_ta)
              )*dx
             + k*p_0*inner(v, n)*ds
             )

        dF = derivative(F, w_0)
        pde = NonlinearVariationalProblem(F, w_0, bcw, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 10
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
        #print solver.parameters["newton_solver"].keys()
        solver.solve()

        #(u_0, p_0) = w_0.split() 

        # ADVECTION-DIFFUSION

        if print_log:
            print 'Advection-diffusion of sediment'
        A = assemble(a_c) 
        b = assemble(L_c)
        [bc.apply(A, b) for bc in bcc]
        solve(A, c_0.vector(), b, "gmres", "default")

        nl_it += 1

    # Store values for next timestep
    w_1.assign(w_0)
    (u_1, p_1) = w_1.split(deepcopy = True)
    c_1.assign(c_0)

    # dt = Calc_timestep(u_0, h)
    # k = Constant(dt)

    t += dt

    if t > t_store:
        if print_t:
            print t

        # Save to file
        u_file << u_1
        p_file << p_1
        c_file << c_1

        t_store += dt_store
