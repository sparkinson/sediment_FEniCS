# This file was *autogenerated* from the file sw_mms.sage.
from sage.all_cmdline import *   # import sage library
_sage_const_2p3 = RealNumber('2.3'); _sage_const_2 = Integer(2); _sage_const_2p0 = RealNumber('2.0'); _sage_const_1p0 = RealNumber('1.0'); _sage_const_1p2 = RealNumber('1.2'); _sage_const_1p4 = RealNumber('1.4'); _sage_const_0p0 = RealNumber('0.0'); _sage_const_1p = RealNumber('1.'); _sage_const_0p5 = RealNumber('0.5'); _sage_const_10p0 = RealNumber('10.0')
def function(phi_0, phi_x, f_sin_x, f_cos_x, alpha_x, x = x):
    f_0 = phi_0 
    f_x = phi_x*(f_sin_x*sin(alpha_x*x) + f_cos_x*cos(alpha_x*x)) 
    f = f_0 + f_x
    return f

h = function(_sage_const_10p0 , _sage_const_1p0 , _sage_const_1p0 , _sage_const_0p0 , _sage_const_2p3 )
h_n = function(_sage_const_10p0 , _sage_const_1p0 , _sage_const_1p0 , _sage_const_0p0 , _sage_const_2p3 , pi)
phi = function(_sage_const_1p0 , _sage_const_0p5 , _sage_const_0p0 , _sage_const_1p0 , _sage_const_1p4 )
phi_n = function(_sage_const_1p0 , _sage_const_0p5 , _sage_const_0p0 , _sage_const_1p0 , _sage_const_1p4 , pi)
u_N = phi_n**_sage_const_0p5 
q = function(u_N*h_n, _sage_const_1p2 , _sage_const_1p0 , _sage_const_0p0 , _sage_const_1p0 )
print str(diff(q/h,x)).replace('^','**')
q_n = function(u_N*h_n, _sage_const_1p0 , _sage_const_1p0 , _sage_const_0p0 , _sage_const_1p0 , pi)
grad_u = diff(q/h,x)
c_d = function(_sage_const_2p0 , _sage_const_1p2 , _sage_const_1p0 , _sage_const_0p0 , _sage_const_1p0 )

s_h = _sage_const_1p /pi*(x*u_N*diff(h,x) - diff(q,x))
alpha = (_sage_const_2 *q/h + (phi*h)**_sage_const_0p5 )*h # u must always be positive and b*delta_x must be 1.0
s_q = _sage_const_1p /pi*(x*u_N*diff(q,x) - diff(q**_sage_const_2 /h + _sage_const_0p5 *(phi*h), x) + diff(alpha * diff(q/h, x), x))
s_phi = _sage_const_1p /pi*(x*u_N*diff(phi, x) - diff(q*phi/h, x)) - phi/h
s_c_d = _sage_const_1p /pi*x*u_N*diff(c_d, x) + phi/h

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
