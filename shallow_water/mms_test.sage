def function(phi_0, phi_x, f_sin_x, f_cos_x, alpha_x, x = x):
    f_0 = phi_0 
    f_x = phi_x*(f_sin_x*sin(alpha_x*x) + f_cos_x*cos(alpha_x*x)) 
    f = f_0 + f_x
    return f

u = function(10.0, 1.0, 1.0, 0.0, 2.3)

s_u = x*diff(u,x) - diff(u,x,x)
# s_u = - diff(u,x,x)
s_u = x*diff(u,x)

print "def u():"
print "    return '", str(u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_u():"
print "    return '", str(s_u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
