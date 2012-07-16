from dolfin import *
from math import pi

parameters["form_compiler"]["cpp_optimize"] = True

u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
f_file = File("results/forcing.pvd")

u0s = "sin(x[0])*cos(x[1])"
u1s = "-cos(x[0])*sin(x[1])"
ps = "cos(x[0])*cos(x[1])"
u0fs = "2*sin(x[0])*cos(x[1])"
u1fs = "- sin(x[0])*cos(x[1]) - sin(x[1])*cos(x[0])"

def NavierStokes(nx, degreeU, degreeP):

    nu = Constant(1.0)
    w0 = Expression((u0s, u1s, ps), degree=2)
    u_e = Expression((u0s, u1s), degree=3)
    p_e = Expression((ps), degree=3)
    f_e = Expression((u0fs, u1fs), degree=3)
    
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'left')
    
    n = FacetNormal(mesh)
    
    V = VectorFunctionSpace(mesh, "CG", degreeU)
    # Q = FunctionSpace(mesh, "CG", degreeP)
    # W = V*Q
    
    # w = project(w0, W)
    # (u, p) = split(w)
    # (v, q) = TestFunctions(W)
    u = Function(V)
    v = TestFunction(V)
    
    f = project(f_e, V)
    
    # bcs = DirichletBC(W.sub(0), u_e, "on_boundary")
    bcs = DirichletBC(V, u_e, "on_boundary")
    
    F= ((#inner(dot(u,grad(u)), v)
         nu*inner(grad(u), grad(v))
         - inner(v, f)
         # - div(u)*q - div(v)*p
         )*dx)
        # + p_e*dot(v, n)*ds)

    print F 
    print v
    # solve(F == 0.0, w, bcs)
    solve(F == 0.0, u, bcs)
    # dF = derivative(F, w)
    # pde = NonlinearVariationalProblem(F, w, bcs, dF)
    # solver = NonlinearVariationalSolver(pde)
    # solver.solve()
    
    # (u_1, p_1) = w.split(deepcopy = True)
    # Eu = errornorm(u_1, u_e, norm_type="L2", degree=degreeU + 1)
    # Ep = errornorm(p_1, p_e, norm_type="L2", degree=degreeP + 1)
    Eu = errornorm(u, u_e, norm_type="L2", degree=degreeU + 1)

    # plot(u_1)
    # plot(p_1)
    # interactive()
    # u_file << u_1
    u_file << u
    # p_file << p_1
    f_file << f

    return Eu#, Ep               
    
import sys
degreeU = int(sys.argv[1]) # read degreeU as 1st command-line arg
degreeP = int(sys.argv[2]) # read degreeP as 1st command-line arg

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32, 64]:
    h.append(pi/nx)
    E.append(NavierStokes(nx, degreeU, degreeP))

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
    # ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    # rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    rp = 0.0    
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i]#[0], " Ep=", E[i][1] 
