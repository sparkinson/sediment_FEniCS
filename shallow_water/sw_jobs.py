#!/usr/bin/python

import sw, sw_io
from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser
import sw_mms_exp as mms
import numpy as np
import sys

parser = OptionParser()
usage = (
'''usage: %prog [options] job_id'
job_id to run:
0 - Validation set up
1 - Create target deposit for adjoint simulation 1
2 - Adjoint simulation 1: match deposit
3 - Adjoint simulation 2: positive gradients
''')
parser = OptionParser(usage=usage)
parser.add_option('-t', '--adjoint_test',
                  action='store_true', dest='adjoint_test', default=False,
                  help='test adjoint solution')
parser.add_option('-T', '--end_time',
                  dest='T', type=float, default=2.0,
                  help='simulation end time')
parser.add_option('-p', '--plot',
                  dest='plot', action='store_true', default=False,
                  help='plot results in real-time')
parser.add_option('-P', '--plot-freq',
                  dest='plot_freq', type=float, default=0.00001,
                  help='provide time between plots')
parser.add_option('-s', '--save_plot',
                  dest='save_plot', action='store_true', default=False,
                  help='save plots')
parser.add_option('-w', '--write',
                  dest='write', action='store_true', default=False,
                  help='write results to json file')
parser.add_option('-W', '--write_freq',
                  dest='write_freq', type=float, default=0.00001,
                  help='time between writing data')
parser.add_option('-l', '--save_loc',
                  dest='save_loc', type=str, default='results/default',
                  help='save location')
parser.add_option('-A', '--adapt_timestep',
                  dest='adapt_timestep', action='store_true', default=False,
                  help='adaptive timestep')
(options, args) = parser.parse_args()

# GENERATE MODEL OBJECT
model = sw.Model()

# PARSE ARGUMENTS/OPTIONS
if len(args) != 1:
    parser.error("provide 1 job id")
job = eval(args[0])

model.save_loc = options.save_loc

if options.plot:
    model.plot = options.plot_freq
model.show_plot = not options.save_plot
model.save_plot = options.save_plot

if options.write:
    model.write = options.write_freq

model.adapt_timestep = options.adapt_timestep

# JOB DEFINITIONS
def adjoint_setup(model):

    # mesh
    model.dX_ = 2.0e-2
    model.L_ = 1.0

    # current properties
    model.x_N_ = 0.5
    model.Fr_ = 1.19
    model.beta_ = 5e-3

    # time stepping
    model.timestep = model.dX_*10.0
    model.adapt_timestep = False
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.2)

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    model.q_b = Constant(0.1)
    model.h_b = Constant(0.0)
    model.phi_b = Constant(0.0)
    model.phi_d_b = Constant(0.1)

    # discretisation
    model.q_degree = 2
    model.h_degree = 1
    model.phi_degree = 1
    model.phi_d_degree = 1
    model.h_disc = "CG"
    model.phi_d_disc = "CG"

if job == 0:  

    # SIMULAITON PARAMETERS

    # mesh
    model.dX_ = 1.0e-2
    model.L_ = 1.0

    # current properties
    model.x_N_ = 0.5
    model.Fr_ = 1.19
    model.beta_ = 5e-3

    # time stepping
    model.timestep = model.dX_/50.0
    model.adapt_timestep = True
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.2)

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    model.q_b = Constant(0.2)
    model.h_b = Constant(0.0)
    model.phi_b = Constant(0.0)
    model.phi_d_b = Constant(0.0)

    # discretisation
    model.q_degree = 2
    model.h_degree = 1
    model.phi_degree = 1
    model.phi_d_degree = 1
    model.h_disc = "CG"
    model.phi_d_disc = "CG"

    model.initialise_function_spaces()
    model.setup(zero_q = True)     

    T = 75.0
    if (options.T): T = options.T
    model.solve(T) 

elif job == 1:  

    adjoint_setup(model)
    model.initialise_function_spaces()

    phi_ic = project(Expression('1.0 - 0.1*cos(pi*x[0])'), model.phi_FS)

    model.setup(phi_ic = phi_ic) 

    q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model.w[0], model.map_dict) 
    sw_io.write_array_to_file('phi_ic.json', phi_ic.vector().array(), 'w')

    model.solve(T = options.T)

    q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model.w[0], model.map_dict) 
    sw_io.write_array_to_file('deposit_data.json', phi_d, 'w')
    sw_io.write_array_to_file('runout_data.json', x_N, 'w')

elif job == 4:  

    adjoint_setup(model)
    model.initialise_function_spaces()

    # phi_ic = project(Expression('1.0 - (0.8*cos(pow(x[0] +0.1,4.0)*pi))'), model.phi_FS)
    phi_ic = project(Expression('1.0'), model.phi_FS)

    model.setup(phi_ic = phi_ic) 

    q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model.w[0], model.map_dict) 
    sw_io.write_array_to_file('phi_ic_2_init.json', phi_ic.vector().array(), 'w')

    model.solve(T = options.T)

    q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model.w[0], model.map_dict) 
    sw_io.write_array_to_file('deposit_data_2_init.json', phi_d, 'w')
    sw_io.write_array_to_file('runout_data_2_init.json', x_N, 'w')

else:

    adjoint_setup(model)
    model.initialise_function_spaces()

    if job == 2 or job == 5:
        target = True
    else:
        target = False

    plotter = sw_io.Adjoint_Plotter('results/adj_{}_'.format(job), True, target=target)

    if job == 2:

        if options.adjoint_test:
            phi_ic = sw_io.create_function_from_file('phi_ic_adj{}_latest.json'.
                                                     format(job), model.phi_FS)
        else:
            phi_ic = project(Expression('1.0'), model.phi_FS)

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
        int_0_scale.assign(1e-5/assemble(int_0))
        int_1_scale.assign(1e-7/assemble(int_1)) # 1e-4 t=5.0, 1e-4 t=10.0
        ### int_0 1e-2, int_1 1e-4 - worked well for dimensionalised problem

        # functional regularisation
        reg_scale = Constant(1)
        int_reg = inner(grad(phi), grad(phi))*reg_scale*dx
        reg_scale_base = 1e-4       # 1e-2 for t=10.0
        reg_scale.assign(reg_scale_base)

        # functional
        scaling = Constant(1e-0)  # 1e0 t=5.0, 1e-1 t=10.0
        J = Functional(scaling*(int_0 + int_1)*dt[FINISH_TIME] + int_reg*dt[START_TIME])

    if job == 5:

        if options.adjoint_test:
            phi_ic = sw_io.create_function_from_file('phi_ic_adj{}_latest.json'.
                                                     format(job), model.phi_FS)
        else:
            phi_ic = project(Expression('1.0'), model.phi_FS)

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
        int_0_scale.assign(1e-5/assemble(int_0))
        int_1_scale.assign(1e-7/assemble(int_1)) # 1e-4 t=5.0, 1e-4 t=10.0
        ### int_0 1e-2, int_1 1e-4 - worked well for dimensionalised problem

        # functional regularisation
        reg_scale = Constant(1)
        int_reg = inner(grad(phi), grad(phi))*reg_scale*dx
        reg_scale_base = 1e-4       # 1e-2 for t=10.0
        reg_scale.assign(reg_scale_base)

        # functional
        scaling = Constant(1e-0)  # 1e0 t=5.0, 1e-1 t=10.0
        J = Functional(scaling*(int_0 + int_1)*dt[FINISH_TIME] + int_reg*dt[START_TIME])

    elif job == 3:

        model.q_b.assign(0.2)
        model.h_b.assign(0.2)
        model.phi_b.assign(0.2)
        model.phi_d_b.assign(0.1)

        phi_ic_0 = project(Expression('1.0'), model.phi_FS)# - 0.1*cos(pi*x[0])'), model.phi_FS)
        # if options.adjoint_test:
        #     # h_ic = sw_io.create_function_from_file('h_ic_adj{}_latest.json'.
        #     #                                         format(options.adjoint), model.h_FS)
        #     phi_ic = sw_io.create_function_from_file('phi_ic_adj{}_latest.json'.
        #                                               format(job), model.phi_FS)
        #     # q_a, q_pa, q_pb = sw_io.read_q_vals_from_file('q_ic_adj{}_latest.json'.
        #     #                                                format(options.adjoint))
        #     # q_a = Constant(q_a); q_pa = Constant(q_pa); q_pb = Constant(q_pb)
        # else:
        #     phi_ic = phi_ic_0.copy(deepcopy = True)
        #     # h_ic = project(Expression('0.2'), model.h_FS)
        #     # q_a = Constant(0.0)
        #     # q_pa = Constant(0.5)
        #     # q_pb = Constant(1.0)
        phi_ic = phi_ic_0.copy(deepcopy = True)
        h_ic = project(Expression('1.0'), model.h_FS)

        model.setup(phi_ic = phi_ic, h_ic = h_ic) #, phi_ic = phi_ic, q_a = q_a, q_pa = q_pa, q_pb = q_pb)
        model.solve(T = options.T)
        (q, h, phi, phi_d, x_N, u_N) = split(model.w[0])

        # functional
        def smooth_pos(val):
            return (val + smooth_abs(val))/2.0
        def smooth_neg(val):
            return (val - smooth_abs(val))/2.0
        def smooth_abs(val):
            return (val**2.0 + 1e-8)**0.5

        int_scale = Constant(1)
        inv_x_N = 1.0/x_N
        filter_val = 1.0-exp(1e1*(model.X*x_N/model.x_N_ - 1.0))
        filter = smooth_pos(filter_val)
        g_phi_d = grad(phi_d)[0]
        pos_grads_on = smooth_pos(g_phi_d)/smooth_abs(g_phi_d)
        neg_grads_on = smooth_neg(g_phi_d)/smooth_abs(g_phi_d)
        # -1e4 for dimensionalised
        pos_f = pos_grads_on*(exp(1e0*g_phi_d) - 1.0)
        neg_f = neg_grads_on*(exp(1e0*g_phi_d) - 1.0)/(x_N-model.x_N_)**2.0
        # int = (1.0 - filter)*(pos_f - neg_f)*int_scale
        int = (1.0 - filter)*(pos_f - neg_f)*int_scale
        int_scale.assign(1e0/abs(assemble(int*dx)))

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
        reg_scale = Constant(0.0)#-1.0e-10)#-5e-4)
        reg_power = Constant(2.0)
        int_reg = (inner(grad(phi),grad(phi))**reg_power*reg_scale #  + 
                   # inner(grad(h),grad(h))**reg_power*reg_scale
                   )

        scaling = Constant(1e-0) 
        J = Functional(scaling*int*dx*dt[FINISH_TIME])# + scaling*int_reg*dx*dt[START_TIME])

        # print int_scale((0,0))
        # q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(model.w[0], model.map_dict) 
        # sw_io.write_array_to_file('deposit_data.json', phi_d, 'w')
        # sw_io.write_array_to_file('runout_data.json', x_N, 'w')
        # sys.exit()

    else:

        print "unknown job_id?"
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

        q_, h_, phi_, phi_d_, x_N_, u_N_ = sw_io.map_to_arrays(model.w[0], model.map_dict) 

        import matplotlib.pyplot as plt

        dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=False)
        dJdh = compute_gradient(J, InitialConditionParameter(h_ic), forget=False)
        # dJdq_a = compute_gradient(J, ScalarParameter(q_a), forget=False)
        # dJdq_pa = compute_gradient(J, ScalarParameter(q_a), forget=False)
        # dJdq_pb = compute_gradient(J, ScalarParameter(q_a), forget=False)

        sw_io.write_array_to_file('dJdphi.json',dJdphi.vector().array(),'w')
        sw_io.write_array_to_file('dJdh.json',dJdh.vector().array(),'w')

        import IPython
        IPython.embed()

        sys.exit()

    # clear old data
    sw_io.clear_file('phi_ic_adj{}.json'.format(job))
    sw_io.clear_file('h_ic_adj{}.json'.format(job))
    sw_io.clear_file('phi_d_adj{}.json'.format(job))
    sw_io.clear_file('q_ic_adj{}.json'.format(job))
    j_log = []

    parameters["adjoint"]["stop_annotating"] = True

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
            sw_io.write_array_to_file('phi_ic_adj{}_latest.json'.format(job),phi_ic,'w')
            sw_io.write_array_to_file('phi_ic_adj{}.json'.format(job),phi_ic,'a')

            try:
                h_ic = value[1].vector().array()
                sw_io.write_array_to_file('h_ic_adj{}_latest.json'.format(job),h_ic,'w')
                sw_io.write_array_to_file('h_ic_adj{}.json'.format(job),h_ic,'a')
            except:
                pass

            try:
                q_a_ = value[2]((0,0)); q_pa_ = value[3]((0,0)); q_pb_ = value[4]((0,0))
                sw_io.write_q_vals_to_file('q_ic_adj{}_latest.json'.format(job),q_a_,q_pa_,q_pb_,'w')
                sw_io.write_q_vals_to_file('q_ic_adj{}.json'.format(job),q_a_,q_pa_,q_pb_,'a')
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
            sw_io.write_array_to_file('j_log{}.json'.format(job), j_log, 'w')

            (fwd_var, output) = adjointer.get_forward_solution(adjointer.equation_count - 1)
            var = adjointer.get_variable_value(fwd_var)
            q, h, phi, phi_d, x_N, u_N = sw_io.map_to_arrays(var.data, model.map_dict)
            sw_io.write_array_to_file('phi_d_adj{}_latest.json'.format(job),phi_d,'w')
            sw_io.write_array_to_file('phi_d_adj{}.json'.format(job),phi_d,'a')

            # from IPython import embed; embed()  
            
            plotter.update_plot(phi_ic, phi_d, j)

            print "* * * J = {}".format(j)

            tic()

            return func_value                

    #######################################
    #### END OF REDUCED FUNCTIONAL HACK
    #######################################

        
    tic()

    if job == 2 or job == 5:

        reduced_functional = MyReducedFunctional(J, 
                                                 [InitialConditionParameter(phi_ic),
                                                  # InitialConditionParameter(h_ic),
                                                  # ScalarParameter(q_a), 
                                                  # ScalarParameter(q_pa), 
                                                  # ScalarParameter(q_pb)
                                                  ])
        bounds = [[0.5], 
                  [1.5]]
        
        # # set int_1 scale to 0.0 initially 
        # # creates instabilities until a rough value has been obtained.
        # int_1_scale.assign(0.0)

        for i in range(15):
            reg_scale.assign(reg_scale_base*2**(0-4*i))

            adj_html("forward.html", "forward")
            adj_html("adjoint.html", "adjoint")
            # from IPython import embed; embed()
            
            # SLSQP L-BFGS-B Newton-CG
            m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                             options = {'maxiter': 5,
                                        'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                             bounds = bounds) 

            # # rescale integral scaling
            # int_0_scale.assign(1e-5/assemble(int_0))
            # int_1_scale.assign(1e-7/assemble(int_1))

    elif job == 3:

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
        bounds = [[0.5# , 0.1, 0.0, 0.2, 1.0
                   ], 
                  [1.5# , 0.5, 1.0, 0.99, 5.0
                   ]]

        m_opt = maximize(reduced_functional, 
                     method = "L-BFGS-B", 
                     options = {'disp': True, 'gtol': 1e-20}, 
                     bounds = bounds) 
