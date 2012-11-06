def function(phi_0, phi_x, f_sin_x, f_cos_x, alpha_x):
    
    f_0 = phi_0 
    f_x = phi_x*(f_sin_x*sin(alpha_x*x) + f_cos_x*cos(alpha_x*x)) 
    f = f_0 + f_x
    return f

u = function(2.5, 1.0, 0.0, 1.0, 1.0)
h = function(1.2, 1.0, 1.0, 0.0, 1.0)

q = u*h

Sh = diff(q,x)
Sq = diff(q**2/h + 0.5*h*h, x)

print "def u_s():"
print "    return '", str(u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def h_s():"
print "    return '", str(h.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def q_s():"
print "    return '", str(q.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def Sh_s():"
print "    return '", str(Sh.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def Sq_s():"
print "    return '", str(Sq.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
