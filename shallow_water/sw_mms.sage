def function(phi_0, phi_x, f_sin_x, f_cos_x, alpha_x, x = x):
    f_0 = phi_0 
    f_x = phi_x*(f_sin_x*sin(alpha_x*x) + f_cos_x*cos(alpha_x*x)) 
    f = f_0 + f_x
    return f

h = function(10.0, 1.0, 1.0, 0.0, 2.3)
h_n = function(10.0, 1.0, 1.0, 0.0, 2.3, pi)
phi = function(1.0, 0.5, 0.0, 1.0, 1.4)
phi_n = function(1.0, 0.5, 0.0, 1.0, 1.4, pi)
u_N = phi_n**0.5
q = function(u_N*h_n, 1.2, 1.0, 0.0, 1.0)
print str(diff(q/h,x)).replace('^','**')
q_n = function(u_N*h_n, 1.0, 1.0, 0.0, 1.0, pi)
grad_u = diff(q/h,x)
c_d = function(2.0, 1.2, 1.0, 0.0, 1.0)

s_h = 1./pi*(x*u_N*diff(h,x) - diff(q,x))
alpha = (2*q/h + (phi*h)**0.5)*h # u must always be positive and b*delta_x must be 1.0
s_q = 1./pi*(x*u_N*diff(q,x) - diff(q**2/h + 0.5*(phi*h), x) + diff(alpha * diff(q/h, x), x))
s_phi = 1./pi*(x*u_N*diff(phi, x) - diff(q*phi/h, x)) - phi/h
s_c_d = 1./pi*x*u_N*diff(c_d, x) + phi/h

print "def h():"
print "    return '", str(h.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def phi():"
print "    return '", str(phi.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def q():"
print "    return '", str(q.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def grad_u():"
print "    return '", str(grad_u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def c_d():"
print "    return '", str(c_d.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def u_N():"
print "    return '", str(u_N.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_h():"
print "    return '", str(s_h.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_q():"
print "    return '", str(s_q.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_phi():"
print "    return '", str(s_phi.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def s_c_d():"
print "    return '", str(s_c_d.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
