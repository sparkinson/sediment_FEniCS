from dolfin import *
import DG_MMSFunctions as mms

parameters["std_out_all_processes"] = True;
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    mesh = Rectangle(0.,0.,pi,pi,nx,nx,'right')
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    
    # define equations to be solved and solve  
    kappa = Constant(1.0)
    alpha = 1.0
    gamma = 5.0
    
    bcc = [DirichletBC(D, c_s, 'on_boundary')]
    #bcc = None
    
    un_up = (inner(u_s, n) + abs(inner(u_s, n)))/2.0
    un_down = (inner(u_s, n) - abs(inner(u_s, n)))/2.0
    F = (inner(grad(d), kappa*grad(c) - u_s*c)*dx 
         - d*f*dx)
    if ele_type == 'DG':  
        # generate function spaces and functions
        D = FunctionSpace(mesh, ele_type, shape_C)
        V = VectorFunctionSpace(mesh, 'CG', shape_U)
        c = TrialFunction(D)   
        d = TestFunction(D)  
        c_0 = project(c_s, D)
        c_1 = project(c_s, D)
        c_0 = Function(D)
       
        # advection
        F += (jump(d)*(un_up('+')*c('+') - un_up('-')*c('-'))*dS
              + d*un_up*c*ds
              #+ d*un_down*c_s*ds #for weak bcs inlets
              )
        # diffusion
        
    else:
        # generate function spaces and functions
        D = FunctionSpace(mesh, ele_type, shape_C)
        V = VectorFunctionSpace(mesh, 'CG', shape_U)
        c = TrialFunction(D)   
        d = TestFunction(D)  
        c_0 = project(c_s, D)
        c_1 = project(c_s, D)
        c_0 = Function(D)
        
        F = (inner(grad(d), kappa*grad(c) - u_s*c)*dx 
             - d*f*dx
             + inner(d*n, u_s*c)*ds 
             - inner(d*n, kappa*grad(c))*ds
             )
        a = lhs(F)
        L = rhs(F)
        
        solve(a == L, c_0, bcc)
        
        # Compute error
        Ec = errornorm(c_0, c_s, norm_type="L2", degree=shape_C + 1)
        print 'Error: ', Ec    

# MMS TEST
   
# read command line variables
import sys
shape_U = int(sys.argv[1]) 
shape_C = int(sys.argv[2]) 
CFL = float(sys.argv[3])
T = float(sys.argv[4])
theta = float(sys.argv[5])
ele_type = sys.argv[6]
ss_tol = float(sys.argv[7])

# show parameters
info(parameters, False)
set_log_active(False)

# generate expressions for initial conditions, boundary conditions and source terms
u_s = Expression((mms.u0_s(), mms.u1_s()), degree = shape_U + 1)
c_s = Expression(mms.c_s(), degree = shape_C + 1)
f = Expression(mms.c_f(), degree = shape_C + 1)

label = 'a'
h = [] # element sizes
E = [] # errors
for i, nx in enumerate([16]):
    c_file = File("results/" + label[i] + ".pvd")
    dt = CFL*(1./nx)
    h.append(pi/nx)
    print 'Edge lengths: ', h[-1], ' dt: ', dt
    main()
