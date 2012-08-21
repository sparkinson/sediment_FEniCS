# -sin(x)*cos(y) + sin(y)*cos(x)

def u_s():

    return ' sin(x[0])*cos(x[1]) '

def v_s():

    return ' -sin(x[1])*cos(x[0]) '

def p_s():

    return ' sin(x[0]*x[1]/pi) + sin(x[0]) + cos(x[1]) - 1.0 '

def c_s():

    return ' 3.7*sin(1.3*x[0]*x[1]/pi) - 1.8*sin(1.7*x[0]) - 1.3*cos(2.1*x[1]) + 5.2 '

def u_fs():

    return ' sin(x[0])*pow(sin(x[1]), 2)*cos(x[0]) + sin(x[0])*cos(x[0])*pow(cos(x[1]), 2) + 2.0*sin(x[0])*cos(x[1]) + x[1]*cos(x[0]*x[1]/pi)/pi + cos(x[0]) '

def v_fs():

    return ' pow(sin(x[0]), 2)*sin(x[1])*cos(x[1]) + sin(x[1])*pow(cos(x[0]), 2)*cos(x[1]) - 2.0*sin(x[1])*cos(x[0]) + x[0]*cos(x[0]*x[1]/pi)/pi - sin(x[1]) '

def c_fs():

    return ' (4.81*x[1]*cos(1.3*x[0]*x[1]/pi)/pi - 3.06*cos(1.7*x[0]))*sin(x[0])*cos(x[1]) - (4.81*x[0]*cos(1.3*x[0]*x[1]/pi)/pi + 2.73*sin(2.1*x[1]))*sin(x[1])*cos(x[0]) + 0.6253*pow(x[0], 2)*sin(1.3*x[0]*x[1]/pi)/pow(pi, 2) + 0.6253*pow(x[1], 2)*sin(1.3*x[0]*x[1]/pi)/pow(pi, 2) - 0.5202*sin(1.7*x[0]) - 0.5733*cos(2.1*x[1]) '

