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

p = function(-1.0, 1.0, 1.0, 1.0,
             1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
             1.0, 1.0, 1.0)
c = function(0.9, 0.9, 0.6, 0.4, 
             0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
             0.6, 0.7, 0.8)
u = 0.0#2.5 + cos(x) + 0.6*sin(y)
v = integral(-diff(u,x),y)  # divergence free
kappa = 1.0

Sc = (u*diff(c,x) + v*diff(c,y) 
       - kappa*(diff(c, x, x) + diff(c, y, y))
       ) 
  
print "def u0_s():"
print "    return '", str(u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u1_s():"
print "    return '", str(v.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def c_s():"
print "    return '", str(c.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def c_fs():"
print "    return '", str(Sc.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
