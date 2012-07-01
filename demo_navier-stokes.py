from dolfin import *

info(parameters, False)

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
parameters["form_compiler"]["cpp_optimize"] = True

u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
f_file = File("results/forcing.pvd")

u0s = "sin(x[0])*cos(x[1])"
u1s = "-cos(x[0])*sin(x[1])"
ps = "cos(x[0])*cos(x[1])"
u0f = "sin(x[0])*pow(sin(x[1]),2)*cos(x[0]) + sin(x[0])*cos(x[0])*pow(cos(x[1]),2) + sin(x[0])*cos(x[1])"
u1f = "pow(sin(x[0]),2)*sin(x[1])*cos(x[1]) + sin(x[1])*pow(cos(x[0]),2)*cos(x[1]) - 3*sin(x[1])*cos(x[0])"
nu = Constant(1.0)
         
# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# class u_forcing(Expression):
#     def eval(self, value, x):
#         value[0] = (sin(x[0])*sin(x[1])**2*cos(x[0]) + 
#                     sin(x[0])*cos(x[0])*cos(x[1])**2 + 
#                     sin(x[0])*cos(x[1])
#                     )
#         value[1] = (sin(x[0])**2*sin(x[1])*cos(x[1]) + 
#                     sin(x[1])*cos(x[0])**2*cos(x[1]) - 
#                     3*sin(x[1])*cos(x[0])
#                     )
#     def value_shape(self):
#         return (2,)

# class NavierStokes():
#     def __init__(self, method = "Coupled", IterativeMethod = "PicardIteration", degreeU = 2, degreeP = 1):
#         if method = "Coupled":
#             self.V = VectorFunctionSpace(mesh, "CG", degreeU)
#             self.Q = FunctionSpace(mesh, "CG", degreeP)
#             self.W = self.V*self.Q
 
#             self.w = project(w_e, self.W)
#             (self.u, self.p) = split(self.w)
#             (self.v, self.q) = TestFunctions(self.W)

def NavierStokes(nx, degreeU, degreeP, method):
    if method == "IPCS":
        return IPCS(nx, degreeU, degreeP)
    if method == "Coupled":
        return Coupled(nx, degreeU, degreeP)

def IPCS(nx, degreeU, degreeP):

    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
        
    n = FacetNormal(mesh)
    
    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", degreeU)
    Q = FunctionSpace(mesh, "CG", degreeP)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Define boundary conditions
    bcu  = [DirichletBC(V, u_e, "on_boundary")]
    bcp  = [DirichletBC(Q, p_e, "on_boundary")]

    # Create functions
    u0 = project(u_e, V)
    u1 = project(u_e, V)
    p0 = project(p_e, Q)
    p1 = project(p_e, Q)

    # Tentative velocity step
    F1 = (inner(grad(u0)*u0, v)*dx + 
          nu*inner(grad(u), grad(v))*dx - 
          inner(f_e, v)*dx +
          inner(grad(p0), v)*dx
          )
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q))*dx
    L2 = ((inner(u1, grad(q)) +
           inner(grad(p0), grad(q))
           )*dx 
          - q*inner(u1, n)*ds)

    # Velocity update
    a3 = inner(u, v)*dx
    L3 = (inner(u1, v) - 
          inner(grad(p1 - p0), v)
          )*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    Eu = 1.0
    Ep = 1.0
    while (Eu > tol or Ep > tol):

        # Compute tentative velocity step
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "gmres", "default")
        end()

        # Pressure correction
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        solve(A2, p1.vector(), b2, "gmres", "default")
        end()

        # Velocity correction
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "gmres", "default")
        end()

        Eu = errornorm(u1, u0, norm_type="L2", degree=degreeU + 1)
        Ep = errornorm(p1, p0, norm_type="L2", degree=degreeU + 1)
        print max(Eu, Ep)

        # Move to next time step
        u0.assign(u1)
        p0.assign(p1)

    Eu = errornorm(u0, u_e, norm_type="L2", degree=degreeU + 1)
    Ep = errornorm(p0, p_e, norm_type="L2", degree=degreeP + 1)

    # Save to file
    ufile << u0
    pfile << p0

    return Eu, Ep         
  
def Coupled(nx, degreeU, degreeP):

    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
        
    n = FacetNormal(mesh)
    
    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "CG", degreeU)
    Q = FunctionSpace(mesh, "CG", degreeP)
    W = V*Q   
 
    w = project(w_e, W)
    (u, p) = split(w)
    (v, q) = TestFunctions(W)
   
    # Define boundary conditions 
    bcu  = [DirichletBC(W.sub(0), u_e, "on_boundary")]

    if itMethod == "Newton":
        F= ((inner(dot(u,grad(u)), v)
             + nu*inner(grad(u), grad(v))
             - inner(v, f_e)
             - div(u)*q - div(v)*p
             )*dx
            + p_e*inner(v, n)*ds)

        # solve(F == 0.0, w, bcu)
        dF = derivative(F, w)
        pde = NonlinearVariationalProblem(F, w, bcu, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 200
        solver.parameters["newton_solver"]["relative_tolerance"] = 1.0e-12
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1.0e-12
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        print solver.parameters["newton_solver"].keys()

        solver.solve()

    elif itMethod == "Picard":
        u0 = project(u_e, V)
        p0 = project(p_e, Q)
        
        F= ((inner(dot(u,grad(u0)), v)
             + nu*inner(grad(u), grad(v))
             - inner(v, f_e)
             - div(u)*q - div(v)*p
             )*dx
            + p_e*inner(v, n)*ds)

        Eu = 1.0
        Ep = 1.0
        while (Eu > tol or Ep > tol):
            solve(F == 0.0, w, bcu)

            (u_1, p_1) = w.split(deepcopy = True)
            Eu = errornorm(u_1, u0, norm_type="L2", degree=degreeU + 1)
            Ep = errornorm(p_1, p0, norm_type="L2", degree=degreeU + 1)
            print max(Eu, Ep)

            # Move to next time step
            u0.assign(u_1)
            p0.assign(p_1)        

    (u_1, p_1) = w.split(deepcopy = True)
    Eu = errornorm(u_1, u_e, norm_type="L2", degree=degreeU + 1)
    Ep = errornorm(p_1, p_e, norm_type="L2", degree=degreeP + 1)

    # Save to file
    ufile << u_1
    pfile << p_1

    return Eu, Ep
  
import sys
degreeU = int(sys.argv[1]) # read degreeU as 1st command-line arg
degreeP = int(sys.argv[2]) # read degreeP as 1st command-line arg
method = sys.argv[3]
tol = float(sys.argv[4])
itMethod = sys.argv[5]

u_e = Expression((u0s, u1s), degree = degreeU + 1)
p_e = Expression((ps), degree = degreeP + 1)
w_e = Expression((u0s, u1s, ps), degree = degreeU + 1)
f_e = Expression((u0f, u1f), degree = degreeU + 1)

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32]:   
    h.append(pi/nx)
    E.append(NavierStokes(nx, degreeU, degreeP, method))

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i][0], " Ep=", E[i][1] 
