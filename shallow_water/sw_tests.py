#!/usr/bin/python

import sw, sw_io
from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser
import sw_mms_exp as mms
import numpy as np
import sys

class MMS_Model(sw.Model):
    def setup(self, dX, dT, disc):
        self.mms = True

        # define constants
        self.dX_ = dX
        self.L_ = np.pi
        self.Fr_ = 1.0
        self.Fr = Constant(1.0)
        self.beta_ = 1.0
        self.beta = Constant(1.0)

        # reset time
        self.t = 0.0

        self.q_b = Constant(0.0) #1e-1 / dX)
        self.h_b = Constant(0.0)
        self.phi_b = Constant(0.0)
        self.phi_d_b = Constant(0.0)

        self.disc = disc
        self.slope_limiter = None
        
        self.initialise_function_spaces()

        self.w_ic = project((Expression(
                    (
                        mms.q(), 
                        mms.h(),
                        mms.phi(),
                        mms.phi_d(),
                        'pi',
                        mms.u_N(),
                        )
                    , self.W.ufl_element())), self.W)

        # define bc's
        bcq = DirichletBC(self.W.sub(0), Expression(mms.q(), degree=self.degree), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bch = DirichletBC(self.W.sub(1), Expression(mms.h(), degree=self.degree), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi = DirichletBC(self.W.sub(2), Expression(mms.phi(), degree=self.degree), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi_d = DirichletBC(self.W.sub(3), Expression(mms.phi_d(), degree=self.degree), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        self.bc = [bcq, bch, bcphi, bcphi_d]
        self.bc = []

        # define source terms
        s_q = Expression(mms.s_q(), self.W.sub(0).ufl_element())
        s_h = Expression(mms.s_h(), self.W.sub(0).ufl_element())
        s_phi = Expression(mms.s_phi(), self.W.sub(0).ufl_element())
        s_phi_d = Expression(mms.s_phi_d(), self.W.sub(0).ufl_element())
        self.S = [s_q, s_h, s_phi, s_phi_d]

        self.timestep = dT
        self.adapt_timestep = False

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self, rescale=True, file=self.save_loc)
            self.plot_t = self.plot

def mms_test(plot, show, save):

    def getError(model):
        Fq = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fh = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.disc, model.degree + 2)

        S_q = project(Expression(mms.q(), degree=5), Fq)
        S_h = project(Expression(mms.h(), degree=5), Fh)
        S_phi = project(Expression(mms.phi(), degree=5), Fphi)
        S_phi_d = project(Expression(mms.phi_d(), degree=5), Fphi_d)

        q, h, phi, phi_d, x_N, u_N = model.w[0].split()
        Eh = errornorm(h, S_h, norm_type="L2", degree_rise=2)
        Ephi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
        Eq = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        Ephi_d = errornorm(phi_d, S_phi_d, norm_type="L2", degree_rise=2)

        return Eh, Ephi, Eq, Ephi_d 

    set_log_level(ERROR)    

    model = MMS_Model() 
    model.plot = plot
    model.show_plot = show
    model.save_plot = save

    disc = 'CG'
    print disc
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([3, 6, 12, 24, 48, 96, 192]):
        h.append(pi/nx)
        print 'h is: ', h[-1]
        model.save_loc = 'results/{}'.format(h[-1])
        model.setup(h[-1], 2.0, disc)
        model.solve(T = 1.0)
        E.append(getError(model))

    print ( "h=%10.2E rh=0.00 rphi=0.00 rq=0.00 rphi_d=0.00 Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
            % (h[0], E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        rphi_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rphi_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
                    % (h[i], rh, rphi, rq, rphi_d, E[i][0], E[i][1], E[i][2], E[i][3]) )    

    disc = 'DG'
    print disc
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([3, 6, 12, 24, 48, 96, 192]):
        dT = (pi/nx) * 1.0
        h.append(pi/nx)
        print 'dt is: ', dT, '; h is: ', h[-1]
        model.setup(h[-1], 1.0, disc)
        model.solve(T = 0.1)
        E.append(getError(model))

    print ( "h=%10.2E rh=0.00 rphi=0.00 rq=0.00 rphi_d=0.00 Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
            % (h[0], E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        rphi_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rphi_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
                    % (h[i], rh, rphi, rq, rphi_d, E[i][0], E[i][1], E[i][2], E[i][3]) )

def taylor_tester(plot, show, save):

    model = sw.Model()

    model.dX = 5e-2
    model.timestep = 1e-2
    model.adapt_timestep = False
    model.plot = plot
    model.show_plot = show
    model.save_plot = save
    model.slope_limiter = True

    set_log_level(PROGRESS) 

    model.initialise_function_spaces()
    
    info_blue('Taylor test for phi')

    ic = project(Expression('0.5'), model.phi_FS)
    # ic = Constant(0.5)
    model.setup(q_a = ic)
    model.solve(T = 3e-2)  
    
    w_0 = model.w[0]
    J = Functional(inner(w_0, w_0)*dx*dt[FINISH_TIME])
    Jw = assemble(inner(w_0, w_0)*dx)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    parameters["adjoint"]["stop_annotating"] = True 

    # dJdphi = compute_gradient(J, ScalarParameter(ic), forget=False)
    dJdphi = compute_gradient(J, InitialConditionParameter(ic), forget=False)
  
    def Jhat(ic):
        model.setup(q_a = ic)
        model.solve(T = 3e-2, annotate = False)
        w_0 = model.w[0]
        print 'Jhat: ', assemble(inner(w_0, w_0)*dx)
        return assemble(inner(w_0, w_0)*dx)

    # conv_rate = taylor_test(Jhat, ScalarParameter(ic), Jw, dJdphi, value = ic, seed=1e-2)
    conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdphi, value = ic, seed=1e-5)

    info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
    if conv_rate > 1.9:
        info_blue('*** test passed ***')
    else:
        info_red('*** ERROR: test failed ***')

def similarity_test(plot, show, save):
    
    model = sw.Model()
    model.plot = plot
    model.show_plot = show
    model.save_plot = save

    # mesh
    model.dX_ = 5.0e-3
    model.L_ = 1.0

    # current properties
    model.Fr_ = 1.19
    model.beta_ = 0.0

    # time stepping
    model.timestep = 5e-4 #model.dX_/500.0
    model.adapt_timestep = False
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.5)

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    model.q_b = Constant(0.0)
    model.h_b = Constant(0.0)
    model.phi_b = Constant(0.0)
    model.phi_d_b = Constant(0.0)

    w_ic_e = [
        '(2./3.)*K*pow(t,-1./3.)*x[0]*(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
        '(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
        '(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
        '0.0',
        'K*pow(t, (2./3.))',
        '(2./3.)*K*pow(t,-1./3.)'
        ]

    def getError(model):
        Fq = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fh = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.disc, model.degree + 2)

        K = ((27.0*model.Fr_**2.0)/(12.0 - 2.0*model.Fr_**2.0))**(1./3.)
        
        S_q = project(Expression(w_ic_e[0], K = K, Fr = model.Fr_, t = model.t, degree=5), Fq)
        S_h = project(Expression(w_ic_e[1],  K = K, Fr = model.Fr_, t = model.t, degree=5), Fh)
        S_phi = project(Expression(w_ic_e[2], K = K, Fr = model.Fr_, t = model.t, degree=5), Fphi)

        q, h, phi, phi_d, x_N, u_N = model.w[0].split()
        E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)
        E_phi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)

        E_x_N = abs(x_N(0) - K*model.t**(2./3.))
        E_u_N = abs(u_N(0) - (2./3.)*K*model.t**(-1./3.))

        return E_q, E_h, E_phi, 0.0, E_x_N, E_u_N
    
    # long test
    T = 0.52
    dt = [1e-1/16, 1e-1/32, 1e-1/64, 1e-1/128, 1e-1/256, 1e-1/512]
    dX = [1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64]

    # # quick settings
    # T = 0.52
    # dt = [1e-1/512]
    # dX = [1.0/4, 1.0/8, 1.0/16]

    E = []
    for dt_ in dt:
        E.append([])
        for dX_ in dX:

            print dX_, dt_

            model.dX_ = dX_
            model.timestep = dt_
            model.t = 0.5

            model.initialise_function_spaces()

            w_ic_E = Expression(
                (
                    w_ic_e[0], 
                    w_ic_e[1], 
                    w_ic_e[2], 
                    w_ic_e[3], 
                    w_ic_e[4], 
                    w_ic_e[5], 
                    ),
                K = ((27.0*model.Fr_**2.0)/(12.0 - 2.0*model.Fr_**2.0))**(1./3.),
                Fr = model.Fr_,
                t = model.t,
                element = model.W.ufl_element(),
                degree = 5)

            w_ic = project(w_ic_E, model.W)
            model.setup(w_ic = w_ic, similarity = True)
            model.t = 0.5
            model.error_callback = getError
            E[-1].append(model.solve(T))

    sw_io.write_array_to_file('similarity_convergence.json', E, 'w')

    E = E[0]
    print ( "R = 0.00  0.00  0.00  0.00  0.00 E = %.2e %.2e %.2e %.2e %.2e" 
            % (E[0][0], E[0][1], E[0][2], E[0][4], E[0][5]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(dX[i]/dX[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(dX[i]/dX[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(dX[i]/dX[i-1])
        rx = np.log(E[i][4]/E[i-1][4])/np.log(dX[i]/dX[i-1])
        ru = np.log(E[i][5]/E[i-1][5])/np.log(dX[i]/dX[i-1])
        print ( "R = %-5.2f %-5.2f %-5.2f %-5.2f %-5.2f E = %.2e %.2e %.2e %.2e %.2e"
                % (rh, rphi, rq, rx, ru, E[i][0], E[i][1], E[i][2], 
                   E[i][4], E[i][5]) )   

def dam_break_test(plot, show, save):
    
    model = sw.Model()
    model.plot = plot
    model.show_plot = show
    model.save_plot = save

    # mesh
    model.L_ = 1.0
    model.x_N_ = 1.0

    # current properties
    model.Fr_ = 1.19
    model.beta_ = 0.0

    # time stepping
    model.adapt_timestep = False
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.5)

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    model.q_b = Constant(0.0)
    model.h_b = Constant(0.0)
    model.phi_b = Constant(0.0)
    model.phi_d_b = Constant(0.0)

    def getError(model):

        q, h, phi, phi_d, x_N_model, u_N_model = model.w[0].split()

        Fq = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fh = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.disc, model.degree + 2)

        u_N = model.Fr_/(1.0+model.Fr_/2.0)
        h_N = (1.0/(1.0+model.Fr_/2.0))**2.0

        x_N = u_N*model.t
        x_M = (2.0 - 3.0*h_N**0.5)*model.t
        x_L = -model.t

        class q_expression(Expression):
            def eval(self, value, x):
                x_gate = (x[0]*x_N_model(0) - 1.0)
                if x_gate <= x_L:
                    value[0] = 0.0
                elif x_gate <= x_M:
                    value[0] = 2./3.*(1.+x_gate/model.t) * 1./9.*(2.0-x_gate/model.t)**2.0
                else:
                    value[0] = model.Fr_/(1.0+model.Fr_/2.0) * (1.0/(1.0+model.Fr_/2.0))**2.0

        class h_expression(Expression):
            def eval(self, value, x):
                x_gate = (x[0]*x_N_model(0) - 1.0)
                if x_gate <= x_L:
                    value[0] = 1.0
                elif x_gate <= x_M:
                    value[0] = 1./9.*(2.0-x_gate/model.t)**2.0
                else:
                    value[0] = (1.0/(1.0+model.Fr_/2.0))**2.0
        
        S_q = project(q_expression(), Fq)
        S_h = project(h_expression(), Fh)

        E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)

        E_x_N = abs(x_N_model(0) - 1.0 - x_N)
        E_u_N = abs(u_N_model(0) - u_N)

        return E_q, E_h, E_x_N, E_u_N
    
    # long test
    T = 0.5
    dt = [1e-1/8, 1e-1/16, 1e-1/32, 1e-1/64, 1e-1/128, 1e-1/256, 1e-1/512]
    dX = [1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64]

    # # quick settings
    # dt = [1e-1/64]
    # dX = [1.0/16, 1.0/32, 1.0/64]
    # dX = [1.0/64]

    E = []
    for dt_ in dt:
        E.append([])
        for dX_ in dX:

            print dX_, dt_

            model.dX_ = dX_
            model.timestep = dt_
            model.t = 0.0
            model.initialise_function_spaces()

            model.setup(zero_q = True, dam_break = True)
            model.error_callback = getError
            E[-1].append(model.solve(T))

    sw_io.write_array_to_file('dam_break.json', E, 'w')

    E = E[0]
    print ( "R = 0.00  0.00  0.00  0.00  E = %.2e %.2e %.2e %.2e" 
            % (E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(dX[i]/dX[i-1])
        rq = np.log(E[i][1]/E[i-1][1])/np.log(dX[i]/dX[i-1])
        rx = np.log(E[i][2]/E[i-1][2])/np.log(dX[i]/dX[i-1])
        ru = np.log(E[i][3]/E[i-1][3])/np.log(dX[i]/dX[i-1])
        print ( "R = %-5.2f %-5.2f %-5.2f %-5.2f E = %.2e %.2e %.2e %.2e"
                % (rh, rq, rx, ru, E[i][0], E[i][1], E[i][2], E[i][3]) )   

if __name__ == '__main__':
    
    parser = OptionParser()
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)     

    parser.add_option('-m', '--mms',
                      action='store_true', dest='mms', default=False,
                      help='mms test')
    parser.add_option('-t', '--taylor_test',
                      action='store_true', dest='taylor_test', default=False,
                      help='adjoint taylor test')
    parser.add_option('-j', '--similarity_solution',
                      action='store_true', dest='similarity', default=False,
                      help='similarity solution test')
    parser.add_option('-d', '--dam_break',
                      action='store_true', dest='dam_break', default=False,
                      help='dam break test')
    parser.add_option('-p', '--plot',
                      dest='plot', type=float, default=None,
                      help='plot results in real-time - provide time between plots')
    parser.add_option('-s', '--show_plot',
                      dest='show_plot', action='store_true', default=False,
                      help='show plots')
    parser.add_option('-S', '--save_plot',
                      dest='save_plot', action='store_true', default=False,
                      help='save plots')
    (options, args) = parser.parse_args()

    # MMS test
    if options.mms == True:
        mms_test(options.plot, options.show_plot, options.save_plot)

    # taylor test
    if options.taylor_test == True:
        taylor_tester(options.plot, options.show_plot, options.save_plot)

    # similarity solution
    if options.similarity == True:
        similarity_test(options.plot, options.show_plot, options.save_plot)

    # dam_break solution
    if options.dam_break == True:
        dam_break_test(options.plot, options.show_plot, options.save_plot)
