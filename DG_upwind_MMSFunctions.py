def u0_s():

    return ' 0.6*sin(x[1]) + cos(x[0]) + 2.5 '

def u1_s():

    return ' x[1]*sin(x[0]) '

def c_s():

    return ' 0.6*sin(0.7*x[1]) + 0.4*cos(0.8*x[0]*x[1]/pi) + 0.9*cos(0.6*x[0]) + 0.9 '

def c_fs():

    return ' (-0.32*x[0]*sin(0.8*x[0]*x[1]/pi)/pi + 0.42*cos(0.7*x[1]))*x[1]*sin(x[0]) + (0.6*sin(x[1]) + cos(x[0]) + 2.5)*(-0.32*x[1]*sin(0.8*x[0]*x[1]/pi)/pi - 0.54*sin(0.6*x[0])) '

