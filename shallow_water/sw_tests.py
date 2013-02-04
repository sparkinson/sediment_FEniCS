#!/usr/bin/python

import sw, sw_io
from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser
import sw_mms_exp as mms
import numpy as np
import sys

class MMS_Model(sw.Model):
    def setup(self, dX_, dT):
        self.mms = True

        # define constants
        self.dX_ = dX_
        self.dX = Constant(self.dX_)
        self.L = np.pi
        self.g_ = 1.0
        self.g = Constant(1.0)
        self.rho_R_ = 1.0
        self.rho_R = Constant(1.0)
        self.b_ = 1.0 / dX_
        self.b = Constant(1.0 / dX_)
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
                        mms.u_N(), 
                        'pi'
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

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = sw_io.Plotter(self)

def mms_test():

    def getError(model):
        Fh = FunctionSpace(model.mesh, "CG", model.h_degree + 1)
        Fq = FunctionSpace(model.mesh, "CG", model.q_degree + 1)
        Fr = FunctionSpace(model.mesh, "R", 0)

        S_h = project(Expression(mms.h(), degree=5), Fh)
        S_phi = project(Expression(mms.phi(), degree=5), Fh)
        S_q = project(Expression(mms.q(), degree=5), Fq)
        S_u_N = project(Expression(mms.u_N(), degree=5), Fr)

        q, h, phi, c_d, u_N, x_N = model.w[0].split()
        Eh = errornorm(h, S_h, norm_type="L2", degree_rise=2)
        Ephi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
        Eq = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        Eu_N = errornorm(u_N, S_u_N, norm_type="L2", degree_rise=2)

        return Eh, Ephi, Eq, Eu_N        

    model = MMS_Model()
    model.plot = False
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([32, 64, 128, 256, 512, 1024]):
        dT = (pi/nx) * 0.5
        h.append(pi/nx)
        print 'dt is: ', dT, '; h is: ', h[-1]
        model.setup(h[-1], dT)
        model.solve(tol = 5e-2)
        E.append(getError(model))

    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e" 
                    % (h[i], rh, rphi, rq, E[i][0], E[i][1], E[i][2]) )

def taylor_tester():

    model = sw.Model()
    model.plot = False
    model.initialise_function_spaces()
    
    info_blue('Taylor test for phi')

    phi_ic = project(Constant(0.06), model.phi_FS)
    # phi_ic = Constant(0.06)
    model.setup(phi_ic = phi_ic)
    # model.solve(T = 0.03)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    J = Functional(inner(model.w[0][2], model.w[0][2])*dx*dt[FINISH_TIME])
    # replay_dolfin()
    dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic))
    # dJdphi = compute_gradient(J, ScalarParameter(phi_ic))
    print dJdphi
    Jphi = assemble(inner(model.w[0][2], model.w[0][2])*dx)

    parameters["adjoint"]["stop_annotating"] = True # stop registering equations
    
    def Jhat(phi_ic):
        model.setup(phi_ic = phi_ic)
        # model.solve(T = 0.03)
        # (q, h, phi, c_d, u_N, x_N) = split(model.w[0])
        print 'Jhat: ', assemble(inner(model.w[0][2], model.w[0][2])*dx)
        return assemble(inner(model.w[0][2], model.w[0][2])*dx)

    # conv_rate = taylor_test(Jhat, ScalarParameter(phi_ic), Jphi, dJdphi, value = phi_ic, seed=1e-4)
    conv_rate = taylor_test(Jhat, InitialConditionParameter(phi_ic), Jphi, dJdphi, value = phi_ic, seed=1e-4)
    
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
    (options, args) = parser.parse_args()

    # MMS test
    if options.mms == True:
        mms_test()

    # taylor test
    if options.taylor_test == True:
        taylor_tester()
    
