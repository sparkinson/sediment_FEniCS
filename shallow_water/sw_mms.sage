def function(phi_0, phi_x, f_sin_x, f_cos_x, alpha_x, x = x):
    f_0 = phi_0 
    f_x = phi_x*(f_sin_x*sin(alpha_x*x) + f_cos_x*cos(alpha_x*x)) 
    f = f_0 + f_x
    return f

Fr = 1.0
h = function(1.0, 1.0, 1.0, 0.0, 2.3)
h_n = function(1.0, 1.0, 1.0, 0.0, 2.3, pi)
phi = function(0.5, 0.5, 0.0, 1.0, 1.4)
phi_n = function(0.5, 0.5, 0.0, 1.0, 1.4, pi)
q = function(Fr*phi_n**0.5*h_n, 1.2, 1.0, 0.0, 1.0)

u_N = Fr*phi_n**0.5

s_h = x*u_N*diff(h,x) - diff(q,x)
s_q = x*u_N*diff(q,x) - diff(q**2/h + 0.5*(phi*h), x)
s_phi = x*u_N*diff(phi, x) - diff(q*phi/h, x)

print "def h():"
print "    return '", str(h.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def phi():"
print "    return '", str(phi.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def q():"
print "    return '", str(q.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u_N():"
print "    return '", str(u_N.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_h():"
print "    return '", str(s_h.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_q():"
print "    return '", str(s_q.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_phi():"
print "    return '", str(s_phi.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_u_N():"
print "    return '", str(s_u_N.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"