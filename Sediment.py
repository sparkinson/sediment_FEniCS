from dolfin import *
import numpy as np
import Sediment_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# simulation parameters
dt_store = 0.1
shape_U = 2
shape_P = 1
shape_C = 2 
picard_tol = 1e-7
min_dt = 5e-3
max_dt = 1e-1
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
u_s = Expression(('0.0', '0.0'), degree = shape_U + 1)
# p_s = Expression('-floor(1.2 - x[0])*(0.02*9.81)*(x[1]-0.2)', degree = shape_P + 1)
p_s = Expression('0.0', degree = shape_P + 1)
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
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
u_0 = project(u_s, V)
u_star = project(u_s, V)
u_1 = project(u_s, V)
p_0 = project(p_s, Q)
p_star = project(p_s, Q)
p_1 = project(p_s, Q)
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
u_ta = theta*u+(1.0-theta)*u_1
c_ta = theta*c+(1.0-theta)*c_1
u_0_ta = theta*u_0+(1.0-theta)*u_1
c_0_ta = theta*c_0+(1.0-theta)*c_1
# non-linear variables
u_nl = theta*u_star+(1.0-theta)*u_1
p_nl = p_star

# MOMENTUM & CONSERVATION

# # pressure guess
# a = inner(v, grad(p))*dx 
# L = inner(v, g*c_0)*dx
# solve(a == L, p_0)
# p_1.assign(p_0)

# momentum equation
F = (((1./k)*inner(v, u - u_1)
      + inner(grad(u_nl)*u_ta, v)
      + nu*inner(grad(u_ta), grad(v))
      - inner(v, Mf)
      - div(v)*p_nl
      - inner(v, g*c_0_ta)
      )*dx
     + p_nl*inner(v, n)*ds
     )
# pressure equation
P = ((k*inner(grad(p - p_nl), grad(q)) - 
      inner(u_0, grad(q))
      )*dx 
     #+ q*inner(u_0, n)*ds # zero normal flow
     )
# velocity correction
F_2 = (inner(u, v) - 
       inner(u_0, v) +
       k*inner(grad(p_0 - p_nl), v)
       )*dx
# seperate bilinear and linear forms of equations and preassemble bilinear form
a1 = lhs(F)
L1 = rhs(F)
a2 = lhs(P)
L2 = rhs(P)
a3 = lhs(F_2)
L3 = rhs(F_2)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3) 

# ADVECTION-DIFFUSION

# define equations to be solved   
F_c = (d*(1./k)*(c - c_1)*dx 
       + d*inner(u_0_ta, grad(c_ta))*dx
       + inner(grad(d), kappa*grad(c_ta))*dx 
       # - inner(d*n, kappa*grad(c))*ds  # zero-flux
       - d*Af*dx
       )

# stabilisation
vnorm = sqrt(inner(u_0_ta, u_0_ta))

# SU stabilisation
# beta = 1.0/(atan(Pe) - Pe)
# Can't calculate using dolfin functions
# Pe = 0.5*u_0_ta*h/kappa
# beta.vector().array()[:] = np.sign(u_0.vector().array()) # sign(Pe) (xi with zero diffusion)
# stab = dot(nu_scale*beta*h, grad(d))*inner(u_0_ta, grad(c_ta))*dx
stab = nu_scale*h/vnorm*inner(u_0_ta, grad(d))*inner(u_0_ta, grad(c_ta))*dx

# SUPG stabilisation
# r = ((1./k)*(c - c_1) 
#      + inner(u_0_ta, grad(c_ta))
#      - div(kappa*grad(c_ta)) 
#      - Af
#      ) 
# p = inner(u_0_ta, grad(d))
# tau = nu_scale*h/vnorm
# stab = p*tau*r*dx

# Apply stabilistaion
F_c += stab

# seperate bilinear and linear forms of equations and preassemble bilinear form
a4 = lhs(F_c)
L4 = rhs(F_c)

############################################################
# define dirichlet boundary conditions

free_slip = Expression('0.0', degree = shape_U + 1)
no_slip = Expression(('0.0', '0.0'), degree = shape_U + 1)
# bcu  = [DirichletBC(V.sub(1), free_slip, "on_boundary && (near(x[1], 0.0) || near(x[1], 0.4))"), 
#         DirichletBC(V.sub(0), free_slip, "on_boundary && (near(x[0], 0.0) || near(x[0], 1.0))")]
bcu  = [DirichletBC(V, no_slip, "on_boundary && !(near(x[1], 0.4))"), 
        DirichletBC(V.sub(1), free_slip, "on_boundary && near(x[1], 0.4)")]
bcp = []
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

        # Iterate until solution is converged
        Eu = 1.0
        Ep = 1.0
        while (Eu > picard_tol or Ep > picard_tol):
            # Compute tentative velocity step
            if print_log:
                print 'Computing tentative velocity step'
            b1 = assemble(L1)
            [bc.apply(A1, b1) for bc in bcu]
            solve(A1, u_0.vector(), b1, "gmres", "default")

            # Pressure correction
            if print_log:
                print 'Computing pressure correction'
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcp]
            solve(A2, p_0.vector(), b2, "gmres", "default")

            # Velocity correction
            if print_log:
                print 'Computing velocity correction'
            b3 = assemble(L3)
            [bc.apply(A3, b3) for bc in bcu]
            solve(A3, u_0.vector(), b3, "gmres", "default")

            Eu = errornorm(u_0, u_star, norm_type="L2", degree=shape_U + 1)
            Ep = errornorm(p_0, p_star, norm_type="L2", degree=shape_P + 1)

            u_star.assign(u_0)
            p_star.assign(p_0)

        # ADVECTION-DIFFUSION

        if print_log:
            print 'Advection-diffusion of sediment'
        A4 = assemble(a4) 
        b4 = assemble(L4)
        [bc.apply(A4, b4) for bc in bcc]
        solve(A4, c_0.vector(), b4, "gmres", "default")

        nl_it += 1

    if t > t_store:
        if print_t:
            print t

        # Save to file
        u_file << u_0
        p_file << p_0
        c_file << c_0

        t_store += dt_store

    # Store values for next timestep
    u_1.assign(u_0)
    p_1.assign(p_0)
    c_1.assign(c_0)

    # dt = Calc_timestep(u_0, h)
    # k = Constant(dt)

    t += dt
    print t
