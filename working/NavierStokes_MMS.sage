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

u = function(2.5, 1.0, 0.6, 0.0, 
             0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
             1.0, 1.0, 1.0)
p = function(-1.0, 1.0, 1.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
             1.0, 1.0, 1.0)
v = integral(-diff(u,x),y)  # divergence free
nu = 1.0

tau_xx = 2*diff(u,x)            *nu
tau_xy = diff(u,y) + diff(v,x)  *nu
tau_yy = 2*diff(v,y)            *nu
tau_yx = diff(u,y) + diff(v,x)  *nu

Su = u*diff(u,x) + v*diff(u,y) - diff(tau_xx, x) - diff(tau_xy, y) + diff(p,x)  
Sv = u*diff(v,x) + v*diff(v,y) - diff(tau_yx, x) - diff(tau_yy, y) + diff(p,y)  
  
print "def u_s():"
print "    return '", str(u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def v_s():"
print "    return '", str(v.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def p_s():"
print "    return '", str(p.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u_fs():"
print "    return '", str(Su.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def v_fs():"
print "    return '", str(Sv.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
