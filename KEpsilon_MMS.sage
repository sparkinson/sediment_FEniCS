y = var('y')

def function(phi_0, phi_x, phi_y, phi_xy, 
             f_sin_x, f_cos_x, f_sin_y, f_cos_y, f_sin_xy, f_cos_xy, 
             alpha_x, alpha_y, alpha_xy):
    
    f_0 = phi_0 
    f_x = phi_x*(f_sin_x*sin(alpha_x*x) + f_cos_x*cos(alpha_x*x)) 
    f_y = phi_y*(f_sin_y*sin(alpha_y*y) + f_cos_y*cos(alpha_y*y)) 
    f_xy = phi_xy*(f_sin_xy*sin(alpha_xy*x*y/pi) + f_cos_xy*cos(alpha_xy*x*y/pi)) 
    f = f_0 + f_x + f_y + f_xy
    return f

# u = function(2.5, 1.0, 0.6, 0.0, 
#              0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
#              1.5, 1.5, 0.6)
p = function(-1.0, 1.0, 1.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
             1.0, 1.0, 1.0)
# rho = function(5.2, -1.8, -1.3, 3.7, 
#                1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
#                1.7, 2.1, 1.3)
ke = function(0.9, 0.9, 0.6, 0.4, 
             0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
             0.6, 0.7, 0.8)
eps = function(8.2, -3.8, 4.3, 1.7, 
             1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
             0.7, 0.8, 0.6)
u = 2.5 + cos(x) + 0.6*sin(y)
v = integral(-diff(u,x),y)  # divergence free
nu_T = ke^2/eps
nu = 1.0

tau_xx = 2*diff(u,x)            *nu
tau_xy = diff(u,y) + diff(v,x)  *nu
tau_yy = 2*diff(v,y)            *nu
tau_yx = diff(u,y) + diff(v,x)  *nu

tau_xx_R = 2*nu_T*diff(u,x) 
tau_xy_R = nu_T*(diff(u,y) + diff(v,x))
tau_yy_R = 2*nu_T*diff(v,y) 
tau_yx_R = nu_T*(diff(u,y) + diff(v,x))

p = p + (2/3)*ke

Su = u*diff(u,x) + v*diff(u,y) - diff(tau_xx, x) - diff(tau_xy, y) - diff(tau_xx_R, x) - diff(tau_xy_R, y) + diff(p,x)  
Sv = u*diff(v,x) + v*diff(v,y) - diff(tau_yx, x) - diff(tau_yy, y) - diff(tau_yx_R, x) - diff(tau_yy_R, y) + diff(p,y)  

P = nu_T*(2*(diff(u,x)^2 + diff(v,y)^2 + diff(u,y)*diff(v,x)) + diff(u,y)^2 + diff(v,x)^2) - (2./3.)*ke*(diff(u,x) + diff(v,y)) 

pr = 1
ab = 1

Ske = (u*diff(ke,x) + v*diff(ke,y) 
       - diff(nu_T*diff(ke, x), x) -  diff(nu_T*diff(ke, y), y)
       - pr*P 
       + ab*eps
       ) 
Seps = (u*diff(eps,x) + v*diff(eps,y) 
        - nu_T*(diff(eps, x, x) + diff(eps, y, y)) - diff(nu_T, x)*diff(eps, x) -  diff(nu_T, y)*diff(eps, y) 
        - pr*(eps/ke)*P 
        + ab*(eps^2/ke)
        )
  
print "def u0_s():"
print "    return '", str(u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u1_s():"
print "    return '", str(v.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def p_s():"
print "    return '", str(p.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def ke_s():"
print "    return '", str(ke.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def eps_s():"
print "    return '", str(eps.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u0_fs():"
print "    return '", str(Su.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u1_fs():"
print "    return '", str(Sv.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def ke_fs():"
print "    return '", str(Ske.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def eps_fs():"
print "    return '", str(Seps.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def P0_s():"
print "    return '", str(P.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def nu_T():"
print "    return '", str(nu_T.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def sym():"
print "    return '", str((diff(u,y) - diff(v,x)).simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
