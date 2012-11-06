def u_s():

    return ' cos(x[0]) + 2.5 '

def h_s():

    return ' sin(x[0]) + 1.2 '

def q_s():

    return ' (cos(x[0]) + 2.5)*(sin(x[0]) + 1.2) '

def Sh_s():

    return ' (cos(x[0]) + 2.5)*cos(x[0]) - (sin(x[0]) + 1.2)*sin(x[0]) '

def Sq_s():

    return ' pow((cos(x[0]) + 2.5), 2)*cos(x[0]) - (2*cos(x[0]) + 5.0)*(sin(x[0]) + 1.2)*sin(x[0]) + (0.5*sin(x[0]) + 0.6)*cos(x[0]) + 0.5*(sin(x[0]) + 1.2)*cos(x[0]) '

