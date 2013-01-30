#!/usr/bin/python

import sys
from dolfin import *
from dolfin_tools import *
from dolfin_adjoint import *
import sw_mms_exp as mms
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from optparse import OptionParser

############################################################
# DOLFIN SETTINGS

parameters['krylov_solver']['relative_tolerance'] = 1e-15
dolfin.parameters["optimization"]["test_gradient"] = True
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.0001
info(parameters, True)
set_log_active(True)

############################################################
# TIME DISCRETISATION FUNCTIONS

def crank_nicholson(u):
    return 0.5*u[0] + 0.5*u[1]
def backward_euler(u): #implicit
    return u[0]
def forward_euler(u): #explicit
    return u[1]

# def model(c_0):

    # SIMULATION USER DEFINED PARAMETERS
# function spaces
shape = 1

# mesh
dX_ = 5e-2
L = 1.0

# stabilisation
b_ = 0.3

# current properties
# g_prime_ = 0.81 # 9.81*0.0077
c_0 = 0.00349
rho_R_ = 1.717
h_0 = 0.4
x_N_ = 0.2
Fr_ = 1.19
g_ = 9.81

# time step
timestep = 5.0e-2 #1./1000.

# define time discretisation
td = crank_nicholson

# mms test (default False)
mms = False

# define constants
dX = Constant(dX_)
g = Constant(g_)
rho_R = Constant(rho_R_)
b = Constant(b_)
Fr = Constant(Fr_)

# define geometry
mesh = Interval(int(L/dX_), 0.0, L)
n = FacetNormal(mesh)

# define function spaces
Q = FunctionSpace(mesh, "CG", shape)
G = FunctionSpace(mesh, "DG", shape - 1)
R = FunctionSpace(mesh, "R", 0)

# define test functions
v = TestFunction(Q)
z = TestFunction(G)
r = TestFunction(R)

def main(u_sink):

    h_exp = str(h_0)
    phi_exp = str(c_0*rho_R_*g_*h_0)
    c_d_exp = '0.0'
    q_exp = '0.0'
    x_N_exp = str(x_N_)
    u_N_exp = '0.0'

    # define function dictionaries for prognostic variables
    h = dict([[i, interpolate(Expression(h_exp), Q)] for i in range(2)])
    phi = dict([[i, interpolate(Expression(phi_exp), Q)] for i in range(2)])
    c_d = dict([[i, interpolate(Expression(c_d_exp), Q)] for i in range(2)])
    q = dict([[i, interpolate(Expression(q_exp), Q)] for i in range(2)])
    x_N = dict([[i, interpolate(Expression(x_N_exp), R)] for i in range(2)])
    u_N = dict([[i, interpolate(Expression(u_N_exp), R)] for i in range(2)])
    X = interpolate(Expression('x[0]'), Q) 

    # left boundary marked as 0, right as 1
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.5 + DOLFIN_EPS and on_boundary
    left_boundary = LeftBoundary()
    exterior_facet_domains = FacetFunction("uint", mesh)
    exterior_facet_domains.set_all(1)
    left_boundary.mark(exterior_facet_domains, 0)
    ds = Measure("ds")[exterior_facet_domains]    

    # define bc's
    bch = []
    bcphi = []
    bcc_d = [DirichletBC(Q, '0.0', "near(x[0], 1.0) && on_boundary")]
    bcq = [DirichletBC(Q, '0.0', "near(x[0], 0.0) && on_boundary")]

    T = 0.1
    tol = None
    nl_tol = 1e-5

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

    t = 0.0
    du = 1e10
    while not (time_finish(t) or converged(du)):
        t += timestep
        k = Constant(timestep)

        ss = 1.0
        nl_its = 0
        while (nl_its < 2 or du_nl > nl_tol):

            # VALUES FOR CONVERGENCE TEST
            h_nl = h[0].copy(deepcopy=True)
            phi_nl = phi[0].copy(deepcopy=True)
            q_nl = q[0].copy(deepcopy=True)
            x_N_nl = x_N[0].copy(deepcopy=True)

            # DEFINE EQUATIONS
            # time discretisation of values
            x_N_td = td(x_N)
            inv_x_N = 1./x_N_td
            u_N_td = td(u_N)
            h_td = td(h)
            phi_td = td(phi)
            c_d_td = td(c_d)
            q_td = td(q)

            # momentum
            q_N = u_N_td*h_td
            u = q_td/h_td
            alpha = 0.0 #b*dX*(abs(u)+u+(phi_td*h_td)**0.5)*h_td
            F_q = v*(q[0] - q[1])*dx + \
                inv_x_N*v*grad(q_td**2.0/h_td + 0.5*phi_td*h_td)*k*dx + \
                inv_x_N*u_N_td*grad(v*X)*q_td*k*dx - \
                inv_x_N*u_N_td*v*X*q_N*n*k*ds(1) + \
                inv_x_N*grad(v)*alpha*grad(u)*k*dx - \
                inv_x_N*v*alpha*grad(u)*n*k*ds(1) 
                # inv_x_N*v*alpha*Constant(-0.22602295050021465)*n*k*ds(1) 
            if mms:
                F_q = F_q + v*s_q*k*dx

            # conservation
            F_h = v*(h[0] - h[1])*dx + \
                inv_x_N*v*grad(q_td)*k*dx - \
                inv_x_N*v*X*u_N_td*grad(h_td)*k*dx 
            if mms:
                F_h = F_h + v*s_h*k*dx

            # concentration
            F_phi = v*(phi[0] - phi[1])*dx + \
                inv_x_N*v*grad(q_td*phi_td/h_td)*k*dx - \
                inv_x_N*v*X*u_N_td*grad(phi_td)*k*dx + \
                v*u_sink*phi_td/h_td*k*dx 
            if mms:
                F_phi = F_phi + v*s_phi*k*dx

            # deposit
            F_c_d = v*(c_d[0] - c_d[1])*dx - \
                inv_x_N*v*X*u_N_td*grad(c_d_td)*k*dx - \
                v*u_sink*phi_td/(rho_R*g*h_td)*k*dx
            if mms:
                F_c_d = F_c_d + v*s_c_d*k*dx

            # nose location/speed
            F_u_N = r*(Fr*(phi_td)**0.5)*ds(1) - \
                r*u_N[0]*ds(1)
            F_x_N = r*(x_N[0] - x_N[1])*dx - r*u_N_td*k*dx 

            # SOLVE EQUATIONS
            solve(F_q == 0, q[0], bcq)
            solve(F_h == 0, h[0], bch)
            solve(F_phi == 0, phi[0], bcphi)
            solve(F_c_d == 0, c_d[0], bcc_d)
            solve(F_u_N == 0, u_N[0])
            if not mms:
                solve(F_x_N == 0, x_N[0])

            dh = errornorm(h_nl, h[0], norm_type="L2", degree=shape + 1)
            dphi = errornorm(phi_nl, phi[0], norm_type="L2", degree=shape + 1)
            dq = errornorm(q_nl, q[0], norm_type="L2", degree=shape + 1)
            dx_N = errornorm(x_N_nl, x_N[0], norm_type="L2", degree=shape + 1)
            du_nl = max(dh, dphi, dq, dx_N)/timestep

            nl_its += 1

        dh = errornorm(h[0], h[1], norm_type="L2", degree=shape + 1)
        dphi = errornorm(phi[0], phi[1], norm_type="L2", degree=shape + 1)
        dq = errornorm(q[0], q[1], norm_type="L2", degree=shape + 1)
        dx_N = errornorm(x_N[0], x_N[1], norm_type="L2", degree=shape + 1)
        du = max(dh, dphi, dq, dx_N)/timestep

        h[1].assign(h[0])
        phi[1].assign(phi[0])
        c_d[1].assign(c_d[0])
        q[1].assign(q[0])
        x_N[1].assign(x_N[0])
        u_N[1].assign(u_N[0])

    return c_d[0]

# return c_d[0]

# c_d_desired = (model(0.00349)).copy(deepcopy=True)
# c_d = model(c_0 = 0.0001)
# J = Functional((c_d - c_d_desired)*dx*dt[FINISH_TIME])
# J = Functional(c_d[0]*dx*dt[FINISH_TIME])
# dJdu_sink = compute_gradient(J, ScalarParameter(u_sink), forget=False)
# dJdphi = compute_gradient(J, InitialConditionParameter(phi[1]))

# print dJdu_sink
# print dJdphi.vector().array()

if __name__ == "__main__":
    u_sink_ = 1e-3
    u_sink = Constant(u_sink_)
    c_d = main(u_sink)
    
    J = Functional(c_d*dx*dt[FINISH_TIME])
    dJdu_sink = compute_gradient(J, ScalarParameter(u_sink))

    Ju_sink = assemble(c_d*dx)

    parameters["adjoint"]["stop_annotating"] = True # stop registering equations

    def Jhat(u_sink): # the functional as a pure function of nu
        c_d = main(u_sink)
        return assemble(c_d*dx)

    conv_rate = taylor_test(Jhat, ScalarParameter(u_sink), Ju_sink, dJdu_sink)

