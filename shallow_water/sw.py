#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
from dolfin_adjoint import *
import numpy as np
from optparse import OptionParser
import json
import sw_io

############################################################
# DOLFIN SETTINGS

parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters['krylov_solver']['relative_tolerance'] = 1e-15
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
solver_parameters = {}
solver_parameters["linear_solver"] = "gmres"
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["maximum_iterations"] = 15
solver_parameters["newton_solver"]["relaxation_parameter"] = 1.0
# info(parameters, True)
# set_log_active(False)
set_log_level(ERROR)

############################################################
# TIME DISCRETISATION FUNCTIONS

class Model():
    
    # SIMULAITON USER DEFINED PARAMETERS

    # mesh
    dX_ = 1.0e-2
    L_ = 1.0

    # current properties
    # c_0 = 0.00349
    # rho_R_ = 1.717
    h_0 = 1.0
    # x_N_ = 0.8
    Fr_ = 1.19
    # g_ = 9.81
    # u_sink_ = 1e-3
    beta_ = 5e-3

    # time stepping
    t = 0.0
    timestep = dX_/50.0 #5e-3
    adapt_timestep = True
    cfl = Constant(0.05)

    # mms test (default False)
    mms = False

    # display plot
    plot = None

    # smoothing eps value
    eps = 1e-12

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    q_b = Constant(0.2)
    h_b = Constant(0.0)
    phi_b = Constant(0.0)
    phi_d_b = Constant(0.0)

    # discretisation
    q_degree = 2
    h_degree = 1
    phi_degree = 1
    phi_d_degree = 1
    h_disc = "CG"
    phi_d_disc = "CG"

    def initialise_function_spaces(self):

        # define geometry
        self.mesh = IntervalMesh(int(self.L_/self.dX_), 0.0, self.L_)
        self.n = FacetNormal(self.mesh)[0]

        self.dX = Constant(self.dX_)
        self.L = Constant(self.L_)

        # left boundary marked as 0, right as 1
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        left_boundary = LeftBoundary()
        exterior_facet_domains = FacetFunction("uint", self.mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains] 

        # define function spaces
        self.q_FS = FunctionSpace(self.mesh, "CG", self.q_degree)
        self.h_FS = FunctionSpace(self.mesh, self.h_disc, self.h_degree)
        self.phi_FS = FunctionSpace(self.mesh, "CG", self.phi_degree)
        self.phi_d_FS = FunctionSpace(self.mesh, self.phi_d_disc, self.phi_d_degree)
        self.var_N_FS = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.q_FS, self.h_FS, self.phi_FS, self.phi_d_FS, self.var_N_FS, self.var_N_FS])
        self.X_FS = FunctionSpace(self.mesh, "CG", 1)

        # get dof_maps for plots
        self.map_dict = dict()
        for i in range(6):
            if self.W.sub(i).dofmap().global_dimension() == len(self.mesh.cells()) + 1:   # P1CG 
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j)[0] for j in range(len(self.mesh.cells()))]
                self.map_dict[i].append(self.W.sub(i).dofmap().cell_dofs(len(self.mesh.cells()) - 1)[-1])
            elif self.W.sub(i).dofmap().global_dimension() == len(self.mesh.cells()) * 2 + 1:   # P2CG 
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j)[:-1] for j in range(len(self.mesh.cells()))]
                self.map_dict[i] = list(np.array(self.map_dict[i]).flatten())
                self.map_dict[i].append(self.W.sub(i).dofmap().cell_dofs(len(self.mesh.cells()) - 1)[-1])    
            elif self.W.sub(i).dofmap().global_dimension() == len(self.mesh.cells()) * 2:   # P1DG
                self.map_dict[i] = [self.W.sub(i).dofmap().cell_dofs(j) for j in range(len(self.mesh.cells()))]
                self.map_dict[i] = list(np.array(self.map_dict[i]).flatten())
            else:   # R
                self.map_dict[i] = self.W.sub(i).dofmap().cell_dofs(0)    

        # define test functions
        (self.q_tf, self.h_tf, self.phi_tf, self.phi_d_tf, self.x_N_tf, self.u_N_tf) = TestFunctions(self.W)

        # initialise functions
        X_ = project(Expression('x[0]'), self.X_FS)
        self.X = Function(X_, name='X')

        self.w = dict()
        self.w[0] = Function(self.W, name='U')

    def setup(self, h_ic = None, phi_ic = None, 
              q_a = Constant(0.0), q_pa = Constant(0.0), q_pb = Constant(1.0), 
              w_ic = None):
        # q_a between 0.0 and 1.0 
        # q_pa between 0.2 and 0.99 
        # q_pb between 1.0 and 

        # set current time to 0.0
        self.t = 0.0

        # define constants
        # self.g = Constant(self.g_, name="g")
        # self.rho_R = Constant(self.rho_R_, name="rho_R")
        self.Fr = Constant(self.Fr_, name="Fr")
        # self.u_sink = Constant(self.u_sink_, name="u_sink")
        self.beta = Constant(self.beta_, name="beta")

        if type(w_ic) == type(None):
            # define initial conditions
            if type(h_ic) == type(None):
                h_ic = 1.0 #Constant(self.h_0)
                h_N = 1.0 #self.h_0
            else:
                h_N = h_ic.vector().array()[-1]
            if type(phi_ic) == type(None): 
                # phi_ic = Constant(self.c_0*self.rho_R_*self.g_*self.h_0)
                phi_ic = 1.0 #Constant(self.c_0*self.h_0)
                phi_N = 1.0 #self.c_0*self.rho_R_*self.g_*self.h_0
            else:
                phi_N = phi_ic.vector().array()[-1]

            # set u_N component
            trial = TrialFunction(self.var_N_FS)
            test = TestFunction(self.var_N_FS)
            u_N_ic = Function(self.var_N_FS, name='u_N_ic')
            a = inner(test, trial)*self.ds(1)
            L = inner(test, self.Fr*phi_ic**0.5)*self.ds(1)             
            solve(a == L, u_N_ic)

            q_N_ic = Function(self.var_N_FS, name='q_N_ic')
            # a = inner(test, trial)*self.ds(1)
            # L = inner(test, u_N_ic*h_ic)*self.ds(1)             
            # solve(a == L, q_N_ic)

            trial = TrialFunction(self.q_FS)
            test = TestFunction(self.q_FS)
            q_ic = Function(self.q_FS, name='q_ic')
            # a = inner(test, trial)*dx
            # q_b = Constant(1.0) - q_a  
            # f = (1.0 - (q_a*cos(((self.X/self.L)**q_pa)*np.pi) + q_b*cos(((self.X/self.L)**q_pb)*np.pi)))/2.0
            # L = inner(test, f*q_N_ic)*dx             
            # solve(a == L, q_ic)

            self.w_ic = [
                q_ic, 
                h_ic, 
                phi_ic, 
                Function(self.phi_d_FS, name='phi_d_ic'), 
                Constant(0.5), #self.x_N_), 
                u_N_ic
                ]

            # for exp in self.w_ic:
            #     try:
            #         print exp.vector().array()
            #     except:
            #         print exp((0,0))
            
        else:
            self.w_ic = w_ic

        # define bc's
        bcphi_d = DirichletBC(self.W.sub(3), '0.0', "near(x[0], 1.0) && on_boundary")
        bcq = DirichletBC(self.W.sub(0), '0.0', "near(x[0], 0.0) && on_boundary")
        self.bc = [bcq]#, bcphi_d]

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self, rescale=True)
            self.plot_t = self.plot

    def generate_form(self):

        # galerkin projection of initial conditions on to w
        test = TestFunction(self.W)
        trial = TrialFunction(self.W)
        L = 0; a = 0
        for i in range(len(self.w_ic)):
            a += inner(test[i], trial[i])*dx
            L += inner(test[i], self.w_ic[i])*dx
        solve(a == L, self.w[0])

        # copy to w[1]
        self.w[1] = project(self.w[0], self.W)

        # smooth functions (also never hit zero)
        def smooth_pos(val):
            return (val + smooth_abs(val))/2.0
        def smooth_neg(val):
            return (val - smooth_abs(val))/2.0
        def smooth_abs(val):
            return (val**2.0 + self.eps)**0.5

        # time discretisation of values
        def time_discretise(u):
            return 0.5*u[0] + 0.5*u[1]

        q = dict()
        h = dict()
        phi = dict()
        phi_d = dict()
        x_N = dict()
        u_N = dict()

        q[0], h[0], phi[0], phi_d[0], x_N[0], u_N[0] = split(self.w[0])
        q[1], h[1], phi[1], phi_d[1], x_N[1], u_N[1] = split(self.w[1])

        # define adaptive timestep form
        if self.adapt_timestep:
            # self.k = Function(self.var_N_FS, name="k")
            self.k = project(Expression(str(self.timestep)), model.var_N_FS)
            self.k_tf = TestFunction(self.var_N_FS)
            self.k_trf = TrialFunction(self.var_N_FS)
            self.a_k = self.k_tf*self.k_trf*dx 
            self.L_k = self.k_tf*(x_N[0]*self.dX)/(self.L*u_N[0])*self.cfl*dx
        else:
            self.k = Constant(self.timestep)

        q_td = time_discretise(q)
        # h_td = smooth_pos(time_discretise(h))
        h_td = time_discretise(h)
        h_td_p = smooth_pos(time_discretise(h))
        phi_td = time_discretise(phi)
        phi_d_td = time_discretise(phi_d)
        x_N_td = time_discretise(x_N)
        inv_x_N = 1./x_N_td
        u_N_td = time_discretise(u_N)
        ux = Constant(-1.0)*u_N_td*self.X

        uxn_up = smooth_pos(ux*self.n)
        uxn_down = smooth_neg(ux*self.n)

        def transForm(u, v, index, disc, stab, weak_b, b_val = None):
            if type(b_val) == type(None):
                b_val = u

            F = - self.k*grad(v)[0]*ux*u*dx - self.k*v*grad(ux)[0]*u*dx 
            
            if disc == "CG":
                F += self.k*v*self.n*ux*b_val*weak_b
                if stab((0,0)) > 0.0:
                    tau = Constant(stab)*self.dX/smooth_abs(ux)
                    F += tau*ux*grad(v)[0]*ux*grad(u)[0]*self.k*dx - \
                        tau*ux*self.n*v*ux*grad(b_val)[0]*self.k*weak_b
            elif disc == "DG":
                F += avg(self.k)*jump(v)*(uxn_up('+')*u('+') - uxn_up('-')*u('-'))*dS 
                if self.mms:
                    F += self.k*v*uxn_down*self.w_ic[index]*(self.ds(0) + self.ds(1))
                else:
                    F += self.k*v*uxn_down*b_val*weak_b
            else:
                sys.exit("unknown element type for function index {}".format(index))

            return F

        # momentum 
        F_q = x_N_td*self.q_tf*(q[0] - q[1])*dx + \
            self.k*self.q_tf*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)[0]*dx
        F_q += transForm(q_td, self.q_tf, 0, "CG", Constant(0.0), self.ds(1), u_N_td*h_td)
        # stabilisation
        u = q_td/h_td
        alpha = self.q_b*self.dX*(smooth_abs(u)+u+(phi_td*h_td)**0.5)*h_td
        F_q += self.k*grad(self.q_tf)[0]*alpha*grad(u)[0]*dx - \
            self.k*self.q_tf*alpha*grad(u)[0]*self.n*self.ds(1)  
        if self.mms:
            F_q += x_N_td*self.q_tf*self.s_q*self.k*dx

        # conservation 
        F_h = x_N_td*self.h_tf*(h[0] - h[1])*dx + \
            self.k*self.h_tf*grad(q_td)[0]*dx
        F_h += transForm(h_td, self.h_tf, 1, self.h_disc, self.h_b, self.ds(0) + self.ds(1))
        if self.mms:
            F_h += x_N_td*self.h_tf*self.s_h*self.k*dx

        # mass volume per unit width of sediment = g'(phi)*h = R*g*c*h
        F_phi = x_N_td*self.phi_tf*(phi[0] - phi[1])*dx + \
            self.phi_tf*grad(q_td*phi_td/h_td)[0]*self.k*dx + \
            x_N_td*self.phi_tf*self.beta*phi_td/h_td*self.k*dx
        F_phi += transForm(phi_td, self.phi_tf, 2, "CG", self.phi_b, self.ds(0) + self.ds(1))
        if self.mms:
            F_phi += x_N_td*self.phi_tf*self.s_phi*self.k*dx

        # deposit = c
        F_phi_d = x_N_td*self.phi_d_tf*(phi_d[0] - phi_d[1])*dx - \
            x_N_td*self.phi_d_tf*self.beta*phi_td/h_td*self.k*dx 
            # x_N_td*self.phi_d_tf*self.beta*phi_td/(self.rho_R*self.g*h_td)*self.k*dx 
        F_phi_d += transForm(phi_d_td, self.phi_d_tf, 3, self.phi_d_disc, self.phi_d_b, self.ds(0))
        if self.mms:
            F_phi_d += x_N_td*self.phi_d_tf*self.s_phi_d*self.k*dx

        # nose location/speed
        if self.mms:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx 
        else:
            F_x_N = self.x_N_tf*(x_N[0] - x_N[1])*dx - self.x_N_tf*u_N_td*self.k*dx 
        F_u_N = self.u_N_tf*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            self.u_N_tf*u_N[0]*self.ds(1)
        # F_u_N = self.u_N_tf*(0.5*h_td**-0.5*(phi_td)**0.5)*self.ds(1) - \
        #     self.u_N_tf*u_N[0]*self.ds(1)

        # combine PDE's
        self.F = F_q + F_h + F_phi + F_phi_d + F_x_N + F_u_N
        # self.F = F_h

        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0], trial)

    def solve(self, T = None, tol = None, nl_tol = 1e-5):

        def time_finish(t):
            if T:
                if t >= T:
                    return True
            return False

        def converged(du):
            if tol:
                if du < tol:
                    return True
            return False

        tic()

        delta = 1e10
        while not (time_finish(self.t) or converged(delta)):
            
            # ADAPTIVE TIMESTEP
            if self.adapt_timestep and self.t > 0.0:
                solve(self.a_k == self.L_k, self.k)
                self.timestep = self.k.vector().array()[0]
            
            # SOLVE COUPLED EQUATIONS
            # solve(self.F == 0, self.w[0], bcs=self.bc, solver_parameters=solver_parameters)
            solve(self.F == 0, self.w[0], bcs=self.bc, J=self.J, solver_parameters=solver_parameters)
            
            if tol:
                delta = 0.0
                f_list = [[self.w[0].split()[i], self.w[1].split()[i]] for i in range(len(self.w[0].split()))]
                for f_0, f_1 in f_list:
                    delta = max(errornorm(f_0, f_1, norm_type="L2", degree_rise=1)/self.timestep, delta)

            self.w[1].assign(self.w[0])

            self.t += self.timestep

            # plot(self.w[0][1], interactive = False, rescale = True)

            # display results
            if self.plot:
                if self.t > self.plot_t:
                    self.plotter.update_plot(self)
                    self.plot_t += self.plot
            sw_io.print_timestep_info(self, delta)

            # q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(self)
            # import IPython
            # IPython.embed()

        print "\n* * * Initial forward run finished: time taken = {}".format(toc())
        list_timings(True)

        if self.plot:
            self.plotter.clean_up()

if __name__ == '__main__':

    parser = OptionParser()
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)
    parser.add_option('-a', '--adjoint',
                      dest='adjoint', type=int, default=None,
                      help='adjoint run')
    parser.add_option('-t', '--adjoint_test',
                      action='store_true', dest='adjoint_test', default=False,
                      help='test adjoint solution')
    parser.add_option('-z', '--phi_ic_test2',
                      action='store_true', dest='phi_ic_test2', default=False,
                      help='test phi initial conditions type 2')
    parser.add_option('-T', '--end_time',
                      dest='T', type=float, default=60.0,
                      help='simulation end time')
    parser.add_option('-p', '--plot',
                      dest='plot', action='store_true', default=False,
                      help='plot results in real-time')
    parser.add_option('-P', '--plot-freq',
                      dest='plot_freq', type=float, default=0.00001,
                      help='provide time between plots')
    parser.add_option('-s', '--show_plot',
                      dest='show_plot', action='store_true', default=False,
                      help='show plots')
    parser.add_option('-S', '--save_plot',
                      dest='save_plot', action='store_true', default=False,
                      help='save plots')
    (options, args) = parser.parse_args()
    
    model = Model()
    if options.plot:
        model.plot = options.plot_freq
    model.save_plot = options.save_plot
    model.show_plot = options.show_plot
    model.initialise_function_spaces()

    # Adjoint 
    if options.adjoint:

        if options.adjoint == 1:

            if options.adjoint_test:
                phi_ic = sw_io.create_function_from_file('phi_ic_adj{}_latest.json'.
                                                         format(options.adjoint), model.phi_FS)
            else:
                phi_ic = project(Expression('0.01'), model.phi_FS)
                
            model.setup(phi_ic = phi_ic)#, h_ic=h_ic)
            model.solve(T = options.T)
            (q, h, phi, phi_d, x_N, u_N) = split(model.w[0])

            # get model data
            phi_d_aim = sw_io.create_function_from_file('deposit_data.json', model.phi_d_FS)
            x_N_aim = sw_io.create_function_from_file('runout_data.json', model.var_N_FS)

            # form Functional integrals
            int_0_scale = Constant(1)
            int_1_scale = Constant(1)
            int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*int_0_scale*dx
            int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*int_1_scale*dx

            # determine scaling
            int_0_scale.assign(1e-2/assemble(int_0))
            int_1_scale.assign(1e-4/assemble(int_1)) # 1e-4 t=5.0, 1e-4 t=10.0
            ### int_0 1e-2, int_1 1e-4 - worked well

            # functional regularisation
            reg_scale = Constant(1)
            int_reg = inner(grad(phi), grad(phi))*reg_scale*dx
            reg_scale_base = 1e-2       # 1e-2 for t=10.0
            reg_scale.assign(reg_scale_base)

            # functional
            scaling = Constant(1e-1)  # 1e0 t=5.0, 1e-1 t=10.0
            J = Functional(scaling*(int_0 + int_1)*dt[FINISH_TIME] + int_reg*dt[START_TIME])

            # dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=True)
            # import IPython
            # IPython.embed()

        elif options.adjoint == 2:
            model.q_b.assign(0.2)
            model.h_b.assign(0.2)
            model.phi_b.assign(0.2)
            model.phi_d_b.assign(0.1)

            phi_ic_0 = project(Expression('0.005'), model.phi_FS)
            if options.adjoint_test:
                # h_ic = sw_io.create_function_from_file('h_ic_adj{}_latest.json'.
                #                                         format(options.adjoint), model.h_FS)
                phi_ic = sw_io.create_function_from_file('phi_ic_adj{}_latest.json'.
                                                          format(options.adjoint), model.phi_FS)
                # q_a, q_pa, q_pb = sw_io.read_q_vals_from_file('q_ic_adj{}_latest.json'.
                #                                                format(options.adjoint))
                # q_a = Constant(q_a); q_pa = Constant(q_pa); q_pb = Constant(q_pb)
            else:
                phi_ic = phi_ic_0.copy(deepcopy = True)
                # h_ic = project(Expression('0.2'), model.h_FS)
                # q_a = Constant(0.0)
                # q_pa = Constant(0.5)
                # q_pb = Constant(1.0)

            model.setup(phi_ic = phi_ic) #, phi_ic = phi_ic, q_a = q_a, q_pa = q_pa, q_pb = q_pb)
            model.solve(T = options.T)
            (q, h, phi, phi_d, x_N, u_N) = split(model.w[0])
            
            # positive gradients
            int_scale = Constant(1)
            inv_x_N = 1.0/x_N
            filter_val = 1.0-exp(1e1*(model.X*model.x_N_/x_N - 1.0))
            filter = (filter_val + abs(filter_val))/2.0
            pos_grads_on = (grad(phi_d) + abs(grad(phi_d)))/(2*grad(phi_d))
            neg_grads_on = (grad(phi_d) - abs(grad(phi_d)))/(2*grad(phi_d))
            pos_f = pos_grads_on*(1.0 - exp(-1e5*grad(phi_d)))
            neg_f = neg_grads_on*(exp(1e5*grad(phi_d)) - 1.0)/(x_N-model.L)
            int = (1.0 - filter)*(pos_f + neg_f)*int_scale
            int_scale.assign(1e-2/abs(assemble(int*dx)))

            # func = Function(model.phi_FS)
            # trial = TrialFunction(model.phi_FS)
            # test = TestFunction(model.phi_FS)
            # a = inner(test, trial)*dx
            # L = inner(test, int)*dx             
            # solve(a == L, func)
            # print func.vector().array()
            # sys.exit()

            # mass conservation
            # int_cons = Constant(-1e2)*(phi*dx - phi_ic_0*dx)

            # regulator    pow 4 scale -5e-2 / pow 3 scale -5e-2
            reg_scale = Constant(-5e-2)
            reg_power = Constant(2.0)
            int_reg = (inner(grad(phi),grad(phi))**reg_power*reg_scale + 
                       inner(grad(h),grad(h))**reg_power*reg_scale)

            scaling = Constant(1e-1)  
            J = Functional(scaling*int*dx*dt[FINISH_TIME] + scaling*int_reg*dx*dt[START_TIME])

            print int_scale((0,0))
            q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model)
            sw_io.write_array_to_file('deposit_data.json', phi_d, 'w')
            sw_io.write_array_to_file('runout_data.json', x_N, 'w')
            sys.exit()

            # dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=False)
            # import IPython
            # IPython.embed()
        
        else:

            print "which adjoint do you want?"
            sys.exit()

        if options.adjoint_test:

            g = Function(model.phi_FS)
            reg = Function(model.phi_FS)

            trial = TrialFunction(model.phi_FS)
            test = TestFunction(model.phi_FS)
            a = inner(test, trial)*dx

            L_g = inner(test, int)*dx  
            L_reg = inner(test, int_reg)*dx             
            solve(a == L_g, g)            
            solve(a == L_reg, reg)

            q_, h_, phi_, phi_d_, x_N_, u_N_ = sw_io.map_to_arrays(model)

            import matplotlib.pyplot as plt

            dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=False)
            # dJdh = compute_gradient(J, InitialConditionParameter(h_ic), forget=False)
            # dJdq_a = compute_gradient(J, ScalarParameter(q_a), forget=False)
            # dJdq_pa = compute_gradient(J, ScalarParameter(q_a), forget=False)
            # dJdq_pb = compute_gradient(J, ScalarParameter(q_a), forget=False)
            import IPython
            IPython.embed()

            sys.exit()

        # clear old data
        sw_io.clear_file('phi_ic_adj{}.json'.format(options.adjoint))
        sw_io.clear_file('h_ic_adj{}.json'.format(options.adjoint))
        sw_io.clear_file('q_ic_adj{}.json'.format(options.adjoint))
        j_log = []

        ##############################
        #### REDUCED FUNCTIONAL HACK
        ##############################
        from dolfin_adjoint.adjglobals import adjointer

        class MyReducedFunctional(ReducedFunctional):

            def __call__(self, value):

                try:
                    print "\n* * * Adjoint and optimiser time taken = {}".format(toc())
                    list_timings(True)
                except:
                    pass

                #### initial condition dump hack ####
                phi_ic = value[0].vector().array()
                sw_io.write_array_to_file('phi_ic_adj{}_latest.json'.format(options.adjoint),phi_ic,'w')
                sw_io.write_array_to_file('phi_ic_adj{}.json'.format(options.adjoint),phi_ic,'a')
                
                try:
                    h_ic = value[1].vector().array()
                    sw_io.write_array_to_file('h_ic_adj{}_latest.json'.format(options.adjoint),h_ic,'w')
                    sw_io.write_array_to_file('h_ic_adj{}.json'.format(options.adjoint),h_ic,'a')
                except:
                    pass

                try:
                    q_a_ = value[2]((0,0)); q_pa_ = value[3]((0,0)); q_pb_ = value[4]((0,0))
                    sw_io.write_q_vals_to_file('q_ic_adj{}_latest.json'.format(options.adjoint),q_a_,q_pa_,q_pb_,'w')
                    sw_io.write_q_vals_to_file('q_ic_adj{}.json'.format(options.adjoint),q_a_,q_pa_,q_pb_,'a')
                except:
                    pass

                tic()

                print "\n* * * Computing forward model"

                func_value = (super(MyReducedFunctional, self)).__call__(value)
                # model.setup(h_ic = value[1], phi_ic = value[0], q_a = value[2], q_pa = value[3], q_pb = value[4])
                # model.solve(T = options.T)

                # func_value = adjointer.evaluate_functional(self.functional, 0)

                print "* * * Forward model: time taken = {}".format(toc())
                
                list_timings(True)

                # sys.exit()

                j = self.scale * func_value
                j_log.append(j)
                sw_io.write_array_to_file('j_log{}.json'.format(options.adjoint), j_log, 'w')
                
                print "* * * J = {}".format(j)

                tic()

                return func_value                

        #######################################
        #### END OF REDUCED FUNCTIONAL HACK
        #######################################
        
        tic()

        if options.adjoint == 1:

            reduced_functional = MyReducedFunctional(J, 
                                                     [InitialConditionParameter(phi_ic),
                                                      # InitialConditionParameter(h_ic),
                                                      # ScalarParameter(q_a), 
                                                      # ScalarParameter(q_pa), 
                                                      # ScalarParameter(q_pb)
                                                      ])
            bounds = [[1e-3], 
                      [1e-1]]

            for i in range(15):
                reg_scale.assign(reg_scale_base*2**(0-i))
                
                m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                                 options = {'maxiter': 5,
                                            'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                                 bounds = bounds) 
                
        elif options.adjoint == 2:

            def d_cb(current_func_value, scaled_dfunc_value, parameters):
                import IPython
                IPython.embed()

            reduced_functional = MyReducedFunctional(J, 
                                                     [InitialConditionParameter(phi_ic),
                                                      # InitialConditionParameter(h_ic),
                                                      # ScalarParameter(q_a), 
                                                      # ScalarParameter(q_pa), 
                                                      # ScalarParameter(q_pb)
                                                      ],
                                                     # scale = 1e-4,
                                                     # derivative_cb = d_cb
                                                     )
            bounds = [[5e-3# , 0.1, 0.0, 0.2, 1.0
                       ], 
                      [5e-2# , 0.5, 1.0, 0.99, 5.0
                       ]]

            m_opt = maximize(reduced_functional, 
                         method = "L-BFGS-B", 
                         options = {'disp': True, 'gtol': 1e-20}, 
                         bounds = bounds)  

    else:  

        # h_ic = project(Expression('0.3 - 0.1*cos(pi*x[0])'), model.h_FS)
        phi_ic = project(Expression('0.02 - 0.007*cos(pi*x[0])'), model.phi_FS)

        # phi_ic = sw_io.create_function_from_file('phi_ic_adj1_latest.json'.
        #                                          format(options.adjoint), model.phi_FS)

        model.setup()#phi_ic = phi_ic) #, h_ic = h_ic)

        q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model)
        sw_io.write_array_to_file('phi_ic.json', phi_ic.vector().array(), 'w')
        # sw_io.write_array_to_file('h_ic.json', h_ic.vector().array(), 'w')
        # q_a_ = q_a((0,0)); q_pa_ = q_pa((0,0)); q_pb_ = q_pb((0,0))
        # sw_io.write_q_vals_to_file('q_ic.json',q_a_,q_pa_,q_pb_,'w')

        model.solve(T = options.T)

        q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model)
        sw_io.write_array_to_file('deposit_data.json', phi_d, 'w')
        sw_io.write_array_to_file('runout_data.json', x_N, 'w')
