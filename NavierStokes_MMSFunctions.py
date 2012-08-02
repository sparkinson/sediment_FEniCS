def u_s():

    return ' 0.6*sin(x[1]) + cos(x[0]) + 2.5 '

def v_s():

    return ' x[1]*sin(x[0]) '

def p_s():

    return ' sin(x[0]*x[1]/pi) + sin(x[0]) + cos(x[1]) - 1.0 '

def u_fs():

    return ' 0.6*x[1]*sin(x[0])*cos(x[1]) - (0.6*sin(x[1]) + cos(x[0]) + 2.5)*sin(x[0]) + x[1]*cos(x[0]*x[1]/pi)/pi + 0.6*sin(x[1]) + 2.0*cos(x[0]) '

def v_fs():

    return ' (0.6*sin(x[1]) + cos(x[0]) + 2.5)*x[1]*cos(x[0]) + x[1]*pow(sin(x[0]), 2) + x[1]*sin(x[0]) + x[0]*cos(x[0]*x[1]/pi)/pi - sin(x[1]) '

