y = var('y')
u = sin(x)*cos(y)
v = -cos(x)*sin(y)
p = cos(x)*cos(y)
c = -cos(x)*cos(y)

Su = u*diff(u,x) + v*diff(u,y) + diff(p,x) - (diff(u, x, x) + diff(u, y, y))
Sy = u*diff(v,x) + v*diff(v,y) + diff(p,y) - (diff(v, x, x) + diff(v, y, y))

print str(Su).replace('*', '').replace('sin', '\\text{sin}').replace('cos', '\\text{cos}')
print str(Sy).replace('*', '').replace('sin', '\\text{sin}').replace('cos', '\\text{cos}')
print str(Su).replace('x', 'x[0]').replace('y', 'x[1]').replace('^', '**')
print str(Sy).replace('x', 'x[0]').replace('y', 'x[1]').replace('^', '**')

Sc = u*diff(c,x) + v*diff(c,y) #- (diff(c, x, x) + diff(c, y, y))
print str(Sc).replace('x', 'x[0]').replace('y', 'x[1]').replace('^', '**')