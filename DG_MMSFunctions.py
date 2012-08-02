def u0_s():

    return ' 0.6*sin(x[1]) + cos(x[0]) + 2.5 '

def u1_s():

    return ' x[1]*sin(x[0]) '

def c_s():

    return ' 3.7*sin(1.3*x[0]*x[1]/pi) - 1.8*sin(1.7*x[0]) - 1.3*cos(2.1*x[1]) + 5.2 '

def c_f():

    return ' (4.81*x[0]*cos(1.3*x[0]*x[1]/pi)/pi + 2.73*sin(2.1*x[1]))*x[1]*sin(x[0]) + (0.6*sin(x[1]) + cos(x[0]) + 2.5)*(4.81*x[1]*cos(1.3*x[0]*x[1]/pi)/pi - 3.06*cos(1.7*x[0])) + 6.253*pow(x[0], 2)*sin(1.3*x[0]*x[1]/pi)/pow(pi, 2) + 6.253*pow(x[1], 2)*sin(1.3*x[0]*x[1]/pi)/pow(pi, 2) - 5.202*sin(1.7*x[0]) - 5.733*cos(2.1*x[1]) '

