#!/usr/bin/python

import sw, sw_io
from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser
import sw_mms_exp as mms
import numpy as np
import sys

class MMS_Model(sw.Model):
    def setup(self, dX, dT):
        self.mms = True

        # define constants
        self.dX = dX
        self.L = np.pi
        self.g_ = 1.0
        self.g = Constant(1.0)
        self.rho_R_ = 1.0
        self.rho_R = Constant(1.0)
        self.q_b = Constant(1.0 / dX)
        self.Fr_ = 1.0
        self.Fr = Constant(1.0)
        self.u_sink_ = 1.0
        self.u_sink = Constant(1.0)
        
        self.initialise_function_spaces()

        self.w_ic = project((Expression(
                    (
                        mms.q(), 
                        mms.h(),
                        mms.phi(),
                        mms.c_d(),
                        'pi',
                        mms.u_N(),
                        )
                    , self.W.ufl_element())), self.W)

        # define bc's
        bch = DirichletBC(self.W.sub(1), Expression(mms.h()), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi = DirichletBC(self.W.sub(2), Expression(mms.phi()), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcc_d = DirichletBC(self.W.sub(3), Expression(mms.c_d()), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcq = DirichletBC(self.W.sub(0), Expression(mms.q()), "(near(x[0], 0.0)) && on_boundary")
        self.bc = [bcq, bch, bcphi, bcc_d]

        # define source terms
        self.s_q = Expression(mms.s_q(), self.W.sub(0).ufl_element())
        self.s_h = Expression(mms.s_h(), self.W.sub(0).ufl_element())
        self.s_phi = Expression(mms.s_phi(), self.W.sub(0).ufl_element())
        self.s_c_d = Expression(mms.s_c_d(), self.W.sub(0).ufl_element())

        self.timestep = dT
        self.adapt_timestep = False

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self)

def mms_test(plot):

    def getError(model):
        Fq = FunctionSpace(model.mesh, "CG", model.q_degree + 1)
        Fh = FunctionSpace(model.mesh, model.h_disc, model.h_degree + 1)
        Fphi = FunctionSpace(model.mesh, "CG", model.phi_degree + 1)
        Fc_d = FunctionSpace(model.mesh, model.c_d_disc, model.c_d_degree + 1)

        S_q = project(Expression(mms.q(), degree=5), Fq)
        S_h = project(Expression(mms.h(), degree=5), Fh)
        S_phi = project(Expression(mms.phi(), degree=5), Fphi)
        S_c_d = project(Expression(mms.c_d(), degree=5), Fc_d)

        q, h, phi, c_d, x_N, u_N = model.w[0].split()
        Eh = errornorm(h, S_h, norm_type="L2", degree_rise=2)
        Ephi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
        Eq = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        Ec_d = errornorm(c_d, S_c_d, norm_type="L2", degree_rise=2)

        return Eh, Ephi, Eq, Ec_d        

    model = MMS_Model()
    model.plot = plot

    set_log_level(ERROR)
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([4, 8, 16, 24]):
        dT = (pi/nx) * 0.5
        h.append(pi/nx)
        print 'dt is: ', dT, '; h is: ', h[-1]
        model.setup(h[-1], dT)
        model.solve(tol = 1e-3)
        E.append(getError(model))

    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        rc_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rc_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ec_d=%.2e" 
                    % (h[i], rh, rphi, rq, rc_d, E[i][0], E[i][1], E[i][2], E[i][3]) )

def taylor_tester(plot):

    model = sw.Model()

    model.dX = 5e-2
    model.timestep = 1e-2
    # model.adapt_timestep = False
    model.plot = plot

    model.initialise_function_spaces()
    
    info_blue('Taylor test for phi')

    phi_ic = project(Expression('1.0'), model.phi_FS)
    h_ic = project(Expression('0.4'), model.phi_FS)
    model.setup(phi_ic = phi_ic)
    model.solve(T = 0.03)       

    w_0 = model.w[0]
    J = Functional(inner(w_0, w_0)*dx*dt[FINISH_TIME])
    Jw = assemble(inner(w_0, w_0)*dx)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    parameters["adjoint"]["stop_annotating"] = True 

    dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=False)
  
    def Jhat(phi_ic):
        model.setup(phi_ic = phi_ic, h_ic = h_ic)
        model.solve(T = 0.03)
        w_0 = model.w[0]
        print 'Jhat: ', assemble(inner(w_0, w_0)*dx)
        return assemble(inner(w_0, w_0)*dx)

    conv_rate = taylor_test(Jhat, InitialConditionParameter(phi_ic), Jw, dJdphi, value = phi_ic, seed=1e-2)

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
                      action='store_true', dest='plot', default=False,
                      help='plot results in real-time')
    (options, args) = parser.parse_args()

    # MMS test
    if options.mms == True:
        mms_test(options.plot)

    # taylor test
    if options.taylor_test == True:
        taylor_tester(options.plot)
    
