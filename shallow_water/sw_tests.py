#!/usr/bin/python

import sw
from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser
import sw_mms_exp as mms
import numpy as np
import sw_output

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

        self.phi_ic = project(Expression(mms.phi()), self.h_CG)
        self.h_ic = project(Expression(mms.h()), self.h_CG)
        self.c_d_ic = project(Expression(mms.c_d()), self.h_CG)
        self.q_ic = project(Expression(mms.q()), self.q_CG)
        self.x_N_ic = project(Expression('pi'), self.R)
        self.u_N_ic = project(Expression(mms.u_N()), self.R)
        self.initialise_functions()

        # define bc's
        self.bch = [DirichletBC(self.h_CG, Expression(mms.h()), 
                                "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcphi = [DirichletBC(self.h_CG, Expression(mms.phi()), 
                                  "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcc_d = [DirichletBC(self.h_CG, Expression(mms.c_d()), 
                                  "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]
        self.bcq = [DirichletBC(self.q_CG, Expression(mms.q()), "(near(x[0], 0.0)) && on_boundary")]
                                # "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")]

        # define source terms
        self.s_q = Expression(mms.s_q())
        self.s_h = Expression(mms.s_h())
        self.s_phi = Expression(mms.s_phi())
        self.s_c_d = Expression(mms.s_c_d())

        # initialise plot
        if self.plot:
            self.plotter = sw_output.Plotter(self)

        self.timestep = dT

        self.form()

def mms_test():

    def getError(model):
        Fh = FunctionSpace(model.mesh, "CG", model.h_degree + 1)
        Fq = FunctionSpace(model.mesh, "CG", model.q_degree + 1)

        S_h = project(Expression(mms.h(), degree=5), Fh)
        S_phi = project(Expression(mms.phi(), degree=5), Fh)
        S_q = project(Expression(mms.q(), degree=5), Fq)
        S_u_N = project(Expression(mms.u_N(), degree=5), model.R)

        Eh = errornorm(model.h[0], S_h, norm_type="L2", degree_rise=1)
        Ephi = errornorm(model.phi[0], S_phi, norm_type="L2", degree_rise=1)
        Eq = errornorm(model.q[0], S_q, norm_type="L2", degree_rise=1)
        Eu_N = errornorm(model.u_N[0], S_u_N, norm_type="L2", degree_rise=1)

        return Eh, Ephi, Eq, Eu_N        

    model = MMS_Model()
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([32, 64, 128, 256]):
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

    model.setup()
    model.solve(T = 0.05)

    J = Functional(model.c_d[0]*dx*dt[FINISH_TIME])
    dJdphi = compute_gradient(J, InitialConditionParameter(model.phi[0]))
    Jphi = assemble(model.c_d[0]*dx)

    parameters["adjoint"]["stop_annotating"] = True # stop registering equations
    
    def Jhat(phi_ic):
        model.setup(phi_ic = phi_ic)
        model.solve(T = 0.05)
        print 'Jhat: ', assemble(model.c_d[0]*dx)
        return assemble(model.c_d[0]*dx)

    conv_rate = taylor_test(Jhat, InitialConditionParameter(model.phi[0]), Jphi, dJdphi, value = model.phi_ic, seed=1e-1)
    
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

    # MMS test
    if options.taylor_test == True:
        taylor_tester()
    
