#!/usr/bin/python

import sw_DG, sw_io
from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser
import sw_mms_exp as mms
import numpy as np
import sys

class MMS_Model(sw_DG.Model):
    def setup(self, dX, dT):
        self.mms = True

        # define constants
        self.dX_ = dX
        self.L_ = np.pi
        self.q_b = Constant(1e-1 / dX)
        self.Fr_ = 1.0
        self.Fr = Constant(1.0)
        self.beta_ = 1.0
        self.beta = Constant(1.0)

        # reset time
        self.t = 0.0

        self.h_b = Constant(0.0)
        self.phi_b = Constant(0.0)
        self.phi_d_b = Constant(0.0)

        self.q_degree = 1
        self.h_degree = 1
        self.phi_degree = 1
        self.phi_d_degree = 1

        self.q_disc = 'CG'
        self.h_disc = 'CG'
        self.phi_disc = 'CG'
        self.phi_d_disc = 'CG'
        
        self.initialise_function_spaces()

        self.w_ic = project((Expression(
                    (
                        mms.q(), 
                        mms.q(), 
                        mms.h(),
                        mms.h(),
                        mms.phi(),
                        mms.phi(),
                        mms.phi_d(),
                        'pi',
                        mms.u_N(),
                        )
                    , self.W.ufl_element())), self.W)

        # define bc's
        bcq = DirichletBC(self.W.sub(0), Expression(mms.q(), degree=self.q_degree), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bch = DirichletBC(self.W.sub(1), Expression(mms.h(), degree=self.h_degree), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi = DirichletBC(self.W.sub(2), Expression(mms.phi(), degree=self.phi_degree), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi_d = DirichletBC(self.W.sub(3), Expression(mms.phi_d(), degree=self.phi_d_degree), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        self.bc = [bcq, bch, bcphi, bcphi_d]
        self.bc = []

        # define source terms
        s_q = Expression(mms.s_q(), self.W.sub(0).ufl_element())
        s_h = Expression(mms.s_h(), self.W.sub(0).ufl_element())
        s_phi = Expression(mms.s_phi(), self.W.sub(0).ufl_element())
        s_phi_d = Expression(mms.s_phi_d(), self.W.sub(0).ufl_element())
        s_CG = Constant(0.0)
        self.S = [s_q, s_CG, s_h, s_CG, s_phi, s_CG, s_phi_d]

        self.timestep = dT
        self.adapt_timestep = False

        self.generate_form()
        
        # # initialise plotting
        # if self.plot:
        #     self.plotter = sw_io.Plotter(self, rescale=True, file=self.save_loc)
        #     self.plot_t = self.plot

def mms_test(plot, show, save):

    def getError(model):
        Fq = FunctionSpace(model.mesh, model.q_disc, model.q_degree + 2)
        Fh = FunctionSpace(model.mesh, model.h_disc, model.h_degree + 2)
        Fphi = FunctionSpace(model.mesh, model.phi_disc, model.phi_degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.phi_d_disc, model.phi_d_degree + 2)

        S_q = project(Expression(mms.q(), degree=5), Fq)
        S_h = project(Expression(mms.h(), degree=5), Fh)
        S_phi = project(Expression(mms.phi(), degree=5), Fphi)
        S_phi_d = project(Expression(mms.phi_d(), degree=5), Fphi_d)

        q, q_cg, h, h_cg, phi, phi_cg, phi_d, x_N, u_N = model.w[0].split()
        Eh = errornorm(h_cg, S_h, norm_type="L2", degree_rise=2)
        Ephi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
        Eq = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        Ephi_d = errornorm(phi_d, S_phi_d, norm_type="L2", degree_rise=2)

        return Eh, Ephi, Eq, Ephi_d 

    set_log_level(ERROR)    

    model = MMS_Model() 
    model.plot = plot
    model.show_plot = show
    model.save_plot = save
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([12, 24, 48, 96, 192]):
        h.append(pi/nx)
        print 'h is: ', h[-1]
        model.save_loc = 'results/{}'.format(h[-1])
        model.setup(h[-1], 1.0)
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

    # disc = 'DG'
    # print disc
    
    # h = [] # element sizes
    # E = [] # errors
    # for i, nx in enumerate([48, 96, 192]):
    #     dT = (pi/nx) * 1.0
    #     h.append(pi/nx)
    #     print 'dt is: ', dT, '; h is: ', h[-1]
    #     model.setup(h[-1], dT, disc)
    #     model.solve(tol = dT/50.0)
    #     E.append(getError(model))

    # for i in range(1, len(E)):
    #     rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
    #     rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
    #     rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
    #     rphi_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
    #     print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rphi_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
    #                 % (h[i], rh, rphi, rq, rphi_d, E[i][0], E[i][1], E[i][2], E[i][3]) )

def taylor_tester(plot, show, save):

    model = sw_DG.Model()

    model.dX = 5e-2
    model.timestep = 1e-2
    model.adapt_timestep = False
    model.plot = plot
    model.show_plot = show
    model.save_plot = save

    model.initialise_function_spaces()
    
    info_blue('Taylor test for phi')

    ic = project(Expression('0.5'), model.phi_FS)
    # ic = Constant(0.5)
    model.setup(q_a = ic)
    model.solve(T = 0.03)       

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
        model.solve(T = 0.03)
        w_0 = model.w[0]
        print 'Jhat: ', assemble(inner(w_0, w_0)*dx)
        return assemble(inner(w_0, w_0)*dx)

    # conv_rate = taylor_test(Jhat, ScalarParameter(ic), Jw, dJdphi, value = ic, seed=1e-2)
    conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdphi, value = ic, seed=1e-2)

    info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
    if conv_rate > 1.9:
        info_blue('*** test passed ***')
    else:
        info_red('*** ERROR: test failed ***')

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
    
