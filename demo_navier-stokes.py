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
u0fs = "sin(x[0])*pow(sin(x[1]),2)*cos(x[0]) + sin(x[0])*cos(x[0])*pow(cos(x[1]),2) + sin(x[0])*cos(x[1])"
u1fs = "pow(sin(x[0]),2)*sin(x[1])*cos(x[1]) + sin(x[1])*pow(cos(x[0]),2)*cos(x[1]) - 3*sin(x[1])*cos(x[0])"
nu = Constant(1.0)
         
# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

class NS_args():
    def __init__(self, u0s = "0.0", u1s = "0.0", ps = "0.0", u0fs = "0.0", u1fs = "0.0"):
        self.u0s = u0s
        self.u1s = u1s
        self.ps = ps
        self.u0fs = u0fs
        self.u1fs = u1fs

class NavierStokes():
    picardTol = 1e-7

    def __init__(self, degreeU, degreeP, method = "Coupled", itMethod = "Newton", args = NS_args()):
        self.degreeU = degreeU
        self.degreeP = degreeP
        self.method = method
        self.itMethod = itMethod
        self.args = args

        self.u_0 = Expression((args.u0s, args.u1s), degree = degreeU + 1)
        self.p_0 = Expression((args.ps), degree = degreeP + 1)
        self.w_0 = Expression((args.u0s, args.u1s, ps), degree = degreeU + 1)
        self.f_0 = Expression((args.u0fs, args.u1fs), degree = degreeU + 1)

        if self.method == "Coupled":
            self.InitialiseCoupledProblem()
        elif self.method == "IPCS":
            self.InitialiseIPCSProblem()

    def F(self):
        return ((inner(grad(self.u_bar)*self.u, self.v)
                 + nu*inner(grad(self.u), grad(self.v))
                 - inner(self.v, self.f_0)
                 - div(self.v)*self.p_bar
                 )*dx)
                # + self.p_0*inner(self.v, n)*ds)

    def F_coupled(self):
        return self.F() - div(self.u)*self.q*dx

    def P(self):
        return ((inner(grad(self.p - self.p0), grad(self.q)) - 
                 inner(self.u1, grad(self.q))
                 )*dx 
                + self.q*inner(self.u1, n)*ds)

    def F_update(self):
        return (inner(self.u, self.v) - 
                inner(self.u1, self.v) +
                inner(grad(self.p1 - self.p0), self.v)
                )*dx

    def InitialiseCoupledProblem(self):
        self.V = VectorFunctionSpace(mesh, "CG", self.degreeU)
        self.Q = FunctionSpace(mesh, "CG", self.degreeP)
        self.W = self.V*self.Q
        
        self.w = project(w_e, self.W)
        (self.u, self.p) = split(self.w)
        (self.v, self.q) = TestFunctions(self.W)

        self.bcs  = [DirichletBC(self.W.sub(0), self.u_0, "on_boundary"),
                     DirichletBC(self.W.sub(1), self.p_0, "on_boundary && near(x[0], 0.0)")]

        self.p_bar = self.p
        if self.itMethod == "Newton":
            self.u_bar = self.u
            self.solve = self.NewtonCoupledSolver
        elif self.itMethod == "Picard":
            self.u0 = project(self.u_0, self.V)
            self.p0 = project(self.p_0, self.Q)
            self.u_bar = self.u0
            self.solve = self.PicardCoupledSolver

    def NewtonCoupledSolver(self):
        dF = derivative(self.F_coupled(), self.w)
        pde = NonlinearVariationalProblem(self.F_coupled(), self.w, self.bcs, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 500
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.solve()

        return self.ErrCalcs(self.w.split(deepcopy = True))        
    
    def PicardCoupledSolver(self):
        Eu = 1.0
        Ep = 1.0
        while (Eu > self.picardTol or Ep > self.picardTol):
            solve(self.F_coupled() == 0.0, self.w, self.bcs)

            (u1, p1) = self.w.split(deepcopy = True)
            Eu = errornorm(u1, self.u0, norm_type="L2", degree=self.degreeU + 1)
            Ep = errornorm(p1, self.p0, norm_type="L2", degree=self.degreeU + 1)
            print max(Eu, Ep)

            self.u0.assign(u1)
            self.p0.assign(p1)  

        return self.ErrCalcs((self.u0, self.p0))

    def InitialiseIPCSProblem(self):
        self.V = VectorFunctionSpace(mesh, "CG", self.degreeU)
        self.Q = FunctionSpace(mesh, "CG", self.degreeP)

        # Define trial and test functions
        self.u = TrialFunction(self.V)
        self.p = TrialFunction(self.Q)
        self.v = TestFunction(self.V)
        self.q = TestFunction(self.Q)
        self.u0 = project(self.u_0, self.V)
        self.u1 = project(self.u_0, self.V)
        self.p0 = project(self.p_0, self.Q)
        self.p1 = project(self.p_0, self.Q)

        self.bcs = [DirichletBC(self.V, self.u_0, "on_boundary")]
        
        self.u_bar = self.u0
        self.p_bar = self.p0
        self.a1 = lhs(self.F())
        self.L1 = rhs(self.F())
        self.a2 = lhs(self.P())
        self.L2 = rhs(self.P())
        self.a3 = lhs(self.F_update())
        self.L3 = rhs(self.F_update())

        # Assemble matrices
        self.A1 = assemble(self.a1)
        self.A2 = assemble(self.a2)
        self.A3 = assemble(self.a3)

        self.solve = self.IPCSSolver
    
    def IPCSSolver(self):
        Eu = 1.0
        Ep = 1.0
        while (Eu > self.picardTol or Ep > self.picardTol):

            # Compute tentative velocity step
            b1 = assemble(self.L1)
            [bc.apply(self.A1, b1) for bc in self.bcs]
            solve(self.A1, self.u1.vector(), b1, "gmres", "default")
            end()

            # Pressure correction
            b2 = assemble(self.L2)
            solve(self.A2, self.p1.vector(), b2, "gmres", "default")
            end()

            # Velocity correction
            b3 = assemble(self.L3)
            [bc.apply(self.A3, b3) for bc in self.bcs]
            solve(self.A3, self.u1.vector(), b3, "gmres", "default")
            end()

            Eu = errornorm(self.u1, self.u0, norm_type="L2", degree=self.degreeU + 1)
            Ep = errornorm(self.p1, self.p0, norm_type="L2", degree=self.degreeP + 1)
            print max(Eu, Ep)

            # Move to next time step
            self.u0.assign(self.u1)
            self.p0.assign(self.p1)  

        return self.ErrCalcs((self.u0, self.p0))

    def ErrCalcs(self, (u, p)):
        
        Eu = errornorm(u, self.u_0, norm_type="L2", degree=degreeU + 1)
        Ep = errornorm(p, self.p_0, norm_type="L2", degree=degreeP + 1)

        # Save to file
        ufile << u
        pfile << p

        return Eu, Ep         
        
def NavierStokesTest(nx, degreeU, degreeP, method):
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
          inner(f_e, v)*dx -
          div(v)*p1*dx #+
          # p_e*inner(v, n)*ds
          )
    # F1 = (inner(grad(u0)*u0, v) + 
    #       nu*inner(grad(u), grad(v)) - 
    #       inner(f_e, v) +
    #       inner(grad(p0), v)
    #       )*dx
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
        # [bc.apply(A2, b2) for bc in bcp] # Applying strongly decreases convergence of velocity
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
    bcs  = [DirichletBC(W.sub(0), u_e, "on_boundary"),
            DirichletBC(W.sub(1), p_e, 
                        "on_boundary && near(x[0], 0.0)")]

    if itMethod == "Newton":
        F= ((inner(grad(u)*u, v)
             + nu*inner(grad(u), grad(v))
             - inner(v, f_e)
             - div(u)*q - div(v)*p
             )*dx)
            # + p_e*inner(v, n)*ds)

        dF = derivative(F, w)
        pde = NonlinearVariationalProblem(F, w, bcs, dF)
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["maximum_iterations"] = 500
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.solve()

    elif itMethod == "Picard":
        u0 = project(u_e, V)
        p0 = project(p_e, Q)
        
        F= ((inner(grad(u0)*u, v)
             + nu*inner(grad(u), grad(v))
             - inner(v, f_e)
             - div(u)*q - div(v)*p
             )*dx)
            # + p_e*inner(v, n)*ds)

        Eu = 1.0
        Ep = 1.0
        while (Eu > tol or Ep > tol):
            solve(F == 0.0, w, bcs)

            (u_1, p_1) = w.split(deepcopy = True)
            Eu = errornorm(u_1, u0, norm_type="L2", degree=degreeU + 1)
            Ep = errornorm(p_1, p0, norm_type="L2", degree=degreeU + 1)
            print max(Eu, Ep)

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
f_e = Expression((u0fs, u1fs), degree = degreeU + 1)

h = [] # element sizes
E = [] # errors
for nx in [4, 8, 16, 32]:   
    h.append(pi/nx)
    E.append(NavierStokesTest(nx, degreeU, degreeP, method))

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i][0], " Ep=", E[i][1] 

h = [] # element sizes
E = [] # errors
args = NS_args(u0s, u1s, ps, u0fs, u1fs)
for nx in [4, 8, 16, 32]:   
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)
    NS = NavierStokes(degreeU, degreeP, method, itMethod, args)
    NS.picardTol = tol
    h.append(pi/nx)
    E.append(NS.solve())

# Convergence rates
from math import log as ln # (log is a dolfin name too)

for i in range(1, len(E)):
    ru = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])
    rp = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])
    print "h=%10.2E ru=%.2f rp=%.2f" % (h[i], ru, rp), " Eu=", E[i][0], " Ep=", E[i][1] 
