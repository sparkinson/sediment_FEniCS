from dolfin import *
from dolfin_adjoint import *
import libadjoint
import hashlib
import numpy as np
import scipy.sparse as sp

N = 10
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace([V,V])

def main(ic, annotate=True):
    f = Function(W)

    test = TestFunction(W)
    trial = TrialFunction(W)
    L = 0; a = 0
    for i in range(len(ic)):
        a += inner(test[i], trial[i])*dx
        L += inner(test[i], ic[i])*dx
    solve(a == L, f)

    slope_limit(f, annotate=annotate)
    
    return f

def reverse(a):
    b = a.copy()
    for i in range(len(a)):
        j = len(a) - 1 - i
        b[j] = a[i]
    return b

def slope_limit(f, annotate=True):

    arr = f.vector().array()

    # properties
    ele_dof = 2 
    n_ele = len(mesh.cells())
    
    for i_eq in range(2):

        # create storage arrays for max, min and mean values
        u_i_max = np.ones([n_ele + 1]) * -1e200
        u_i_min = np.ones([n_ele + 1]) * 1e200
        u_c = np.empty([n_ele])
    
        # for each vertex in the mesh store the mean values
        for b in range(len(mesh.cells())):
            indices = W.sub(i_eq).dofmap().cell_dofs(b)

            u_i = np.array([arr[index] for index in indices])
            u_c[b] = u_i.mean()

            if (u_c[b] > u_i_max[b]):
                u_i_max[b] = u_c[b]
            u_i_max[b+1] = u_c[b]
            if (u_c[b] < u_i_min[b]):
                u_i_min[b] = u_c[b]
            u_i_min[b+1] = u_c[b]

        # apply slope limit
        for b in range(len(mesh.cells())):

            # calculate alpha
            alpha = 1.0
            for d in range(ele_dof):
                index = W.sub(i_eq).dofmap().cell_dofs(b)[d]

                limit = True
                if arr[index] > u_c[b]:
                    u_c_i = u_i_max[b+d]
                elif arr[index] < u_c[b]:
                    u_c_i = u_i_min[b+d]
                else:
                    limit = False

                if limit:
                    if u_c_i != u_c[b]:
                        if ((arr[index] - u_c[b]) > (u_c_i - u_c[b]) and
                            (u_c_i - u_c[b])/(arr[index] - u_c[b]) < alpha):
                            alpha = (u_c_i - u_c[b])/(arr[index] - u_c[b])
                    else:
                        alpha = 0

            # apply slope limiting
            indices = W.sub(i_eq).dofmap().cell_dofs(b)
            u_i = np.array([arr[index] for index in indices])
            slope = u_i - u_c[b]
            for d in range(ele_dof): 
                arr[indices[d]] = u_c[b] + alpha*slope[d]

    f.vector()[:] = arr

    if annotate:
        annotate_slope_limit(f)

def annotate_slope_limit(f):
    # First annotate the equation
    adj_var = adjglobals.adj_variables[f]
    rhs = SlopeRHS(f)

    adj_var_next = adjglobals.adj_variables.next(f)

    W = f.function_space()
    identity_block = solving.get_identity(W)

    eq = libadjoint.Equation(adj_var_next, blocks=[identity_block], targets=[adj_var_next], rhs=rhs)
    cs = adjglobals.adjointer.register_equation(eq)

    # Record the result
    adjglobals.adjointer.record_variable(adjglobals.adj_variables[f], libadjoint.MemoryStorage(adjlinalg.Vector(f)))

class SlopeRHS(libadjoint.RHS):
    def __init__(self, f):
        self.adj_var = adjglobals.adj_variables[f]
        self.f = f

    def dependencies(self):
        return [self.adj_var]

    def coefficients(self):
        return [self.f]

    def __str__(self):
        return "SlopeRHS" + hashlib.md5(str(self.f)).hexdigest()

    def __call__(self, dependencies, values):

        d = Function(values[0].data)
        slope_limit(d, annotate=False)

        return adjlinalg.Vector(d)

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):

        f = Function(values[0].data)
        arr = f.vector().array()
        c = contraction_vector.data
        c_arr = c.vector().array()

        out = arr.copy()

        # properties
        ele_dof = 2 
        n_ele = len(mesh.cells())

        # gradient matrix
        n_dof = ele_dof * len(mesh.cells())
        G = sp.lil_matrix((len(c_arr), len(c_arr)))

        for i_eq in range(2):

            # create storage arrays for max, min and mean values
            u_i_max = np.ones([n_ele + 1]) * -1e200
            u_i_min = np.ones([n_ele + 1]) * 1e200
            u_c = np.empty([n_ele])

            # for each vertex in the mesh store the mean values
            for b in range(len(mesh.cells())):
                indices = W.sub(i_eq).dofmap().cell_dofs(b)

                u_i = np.array([arr[index] for index in indices])
                u_c[b] = u_i.mean()

                if (u_c[b] > u_i_max[b]):
                    u_i_max[b] = u_c[b]
                u_i_max[b+1] = u_c[b]
                if (u_c[b] < u_i_min[b]):
                    u_i_min[b] = u_c[b]
                u_i_min[b+1] = u_c[b]

            # apply slope limit
            for b in range(len(mesh.cells())):

                # obtain cell data
                indices = W.sub(i_eq).dofmap().cell_dofs(b)
                c_u = np.array([c_arr[i] for i in indices])
                c_v = np.array([c_arr[i] for i in indices])

                # calculate alpha 
                alpha = 1
                alpha_i = -1
                for d in range(ele_dof):
                    index = W.sub(i_eq).dofmap().cell_dofs(b)[d]

                    limit = True
                    if arr[index] > u_c[b]:
                        u_c_i = u_i_max[b+d]
                    elif arr[index] < u_c[b]:
                        u_c_i = u_i_min[b+d]
                    else:
                        limit = False

                    if limit:
                        print b, d, u_c_i, u_c[b]
                        if u_c_i != u_c[b]: 
                            if ((arr[index] - u_c[b]) > (u_c_i - u_c[b]) and
                                (u_c_i - u_c[b])/(arr[index] - u_c[b]) < alpha):
                                if d == 0:
                                    print b, b-1
                                    indices_u = W.sub(i_eq).dofmap().cell_dofs(b)
                                    indices_v = W.sub(i_eq).dofmap().cell_dofs(b-1)
                                else:
                                    indices_u = reverse(W.sub(i_eq).dofmap().cell_dofs(b))
                                    indices_v = W.sub(i_eq).dofmap().cell_dofs(b+1)
                                u = np.array([arr[i] for i in indices_u])
                                v = np.array([arr[i] for i in indices_v])
                                c_u = np.array([c_arr[i] for i in indices_u])
                                c_v = np.array([c_arr[i] for i in indices_v])

                                alpha = (u_c_i - u_c[b])/(arr[index] - u_c[b])
                                alpha_i = d

                                f_ = v.sum() - u.sum()
                                g_ = u[0] - u[1]
                                d_alpha_ui = -(g_+f_)/g_**2.0 
                                d_alpha_uj = -(g_-f_)/g_**2.0 
                                d_alpha_v  = 1/g_
                        else:
                            alpha = 0

                # default
                indices = W.sub(i_eq).dofmap().cell_dofs(b)
                if alpha_i < 0:
                    d_alpha_ui = 0
                    d_alpha_uj = 0
                    d_alpha_v  = 0
                    alpha_i = 0
                    c_u = np.array([c_arr[i] for i in indices])
                    c_v = np.array([c_arr[i] for i in indices])
                    u = np.array([arr[i] for i in indices])
                    indices_u = W.sub(i_eq).dofmap().cell_dofs(b)
                    try:
                        indices_v = W.sub(i_eq).dofmap().cell_dofs(b-1)
                    except:
                        indices_v =W.sub(i_eq).dofmap().cell_dofs(b+1)

                # apply slope limiting
                for d in range(ele_dof):
                    if d == alpha_i:
                        G[indices[d], indices_u[0]] = 0.5*(1 + alpha + d_alpha_ui*(u[0]-u[1]))
                        G[indices[d], indices_u[1]] = 0.5*(1 - alpha + d_alpha_uj*(u[0]-u[1]))
                        G[indices[d], indices_v[0]] = 0.5*(d_alpha_v*u[0] - d_alpha_v*u[1])
                        G[indices[d], indices_v[1]] = 0.5*(d_alpha_v*u[0] - d_alpha_v*u[1])
                    else:
                        G[indices[d], indices_u[1]] = 0.5*(1 + alpha + d_alpha_uj*(u[1]-u[0]))
                        G[indices[d], indices_u[0]] = 0.5*(1 - alpha + d_alpha_ui*(u[1]-u[0]))
                        G[indices[d], indices_v[0]] = 0.5*(d_alpha_v*u[1] - d_alpha_v*u[0])
                        G[indices[d], indices_v[1]] = 0.5*(d_alpha_v*u[1] - d_alpha_v*u[0])

        if hermitian:
            G = G.transpose()
        f.vector()[:] = G.dot(c_arr)

        return adjlinalg.Vector(f)         

# Create sloped ic
ic_0 = Function(V, name="ic_0")
ic_1 = Function(V, name="ic_1")
ic = [ic_0, ic_1]

for i in range(len(ic)):
    arr = ic[i].vector().array()
    for b in range(0,len(mesh.cells())):
        # obtain cell data
        indices = V.dofmap().cell_dofs(b)
        u_i = np.array([arr[index] for index in indices])
        arr[indices[0]] = np.sin(b/float(N))
        arr[indices[1]] = np.sin((b+1)/float(N))
    for b in range(0,len(mesh.cells()),2):
        # obtain cell data
        indices = V.dofmap().cell_dofs(b)
        u_c = np.array([arr[index] for index in indices]).mean()
        arr[indices[0]] = u_c + 3.0*(arr[indices[0]]-u_c)
        arr[indices[1]] = u_c + 3.0*(arr[indices[1]]-u_c)
    ic[i].vector()[:] = arr

# Annotate the forward model
f = main(ic)
print f.vector().array()
adj_html("forward.html", "forward")
parameters["adjoint"]["stop_annotating"] = True

# Replay test
success = replay_dolfin(tol=0.0, stop=True)
assert success

# Taylor test
J = Functional(inner(f, f)*dx)
m = InitialConditionParameter(ic_0)
dj = compute_gradient(J, m, forget=False)

def Jhat(ic_0):
    ic = [ic_0, ic_1]
    f = main(ic, annotate=False)
    return assemble(inner(f, f)*dx)

minconv = taylor_test(Jhat, m, Jhat(ic_0), dj, perturbation_direction=interpolate(Expression("sin(x[0])"), V))
assert minconv > 1.99


