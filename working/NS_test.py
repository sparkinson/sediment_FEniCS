from dolfin import *
from math import pi

parameters["form_compiler"]["cpp_optimize"] = True

u_file = File("results/velocity.pvd")
p_file = File("results/pressure.pvd")
f_file = File("results/forcing.pvd")

u0s = "x[1]"
u1s = "0.0"
ps = "-x[1]"
u0fs = "0.0"
u1fs = "1.0"

def NavierStokes(nx, degreeU, degreeP):

    w0 = Expression((u0s, u1s, ps))
    u0 = Expression((u0s, u1s), degree=3)
    p0 = Expression((ps), degree=3)
    f0 = Expression((u0fs, u1fs), degree=3)
    
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'left')
    
    n = FacetNormal(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q
    
    w = project(w0, W)
    (u, p) = split(w)
    (v, q) = TestFunctions(W)

    bcs = DirichletBC(W.sub(0), u0, "on_boundary")
    
    nu = 1.0

    F = ((nu*inner(grad(u), grad(v))
          - div(u)*q 
          - div(v)*p 
          - inner(f0, v)
          )*dx
         + p0*dot(v, n)*ds
         )

    # solve(F==0.0, w, bcs)
    dF = derivative(F, w)
    pde = NonlinearVariationalProblem(F, w, bcs, dF)
    solver = NonlinearVariationalSolver(pde)
    solver.solve()
    
    (u_1, p_1) = w.split(deepcopy = True)
    Ep = errornorm(p_1, p0, norm_type="L2", degree=degreeP + 1)
    Eu = errornorm(u_1, u0, norm_type="L2", degree=degreeU + 1)

    # plot(u_1)
    # plot(p_1)
    # interactive()
    u_file << u_1
    p_file << p_1
    # f_file << f0

    return Eu, Ep               
    
import sys
degreeU = int(sys.argv[1]) # read degreeU as 1st command-line arg
degreeP = int(sys.argv[2]) # read degreeP as 1st command-line arg

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32]:
    h.append(pi/nx)
    E.append(NavierStokes(nx, degreeU, degreeP))

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    # ru = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    rp = 0.0    
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i][0], " Ep=", E[i][1] 
