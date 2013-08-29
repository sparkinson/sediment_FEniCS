from dolfin import *
from dolfin_adjoint import *
import libadjoint
import hashlib
import numpy as np

N = 10
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, "DG", 1)

def main(ic, annotate=True):
    f = project(ic, V, annotate=annotate, name="f")
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

    # create storage arrays for max, min and mean values
    ele_dof = 2 
    n_dof = ele_dof * len(mesh.cells())

    u_i_max = np.ones([len(mesh.cells()) + 1]) * -1e200
    u_c = np.empty([len(mesh.cells())])
    
    # for each vertex in the mesh store the mean values
    for b in range(len(mesh.cells())):
        indices = V.dofmap().cell_dofs(b)

        u_i = np.array([arr[index] for index in indices])
        u_c[b] = u_i.mean()

        if (u_c[b] > u_i_max[b]):
            u_i_max[b] = u_c[b]
        u_i_max[b+1] = u_c[b]
        
    # apply slope limit
    for b in range(len(mesh.cells())):

        # obtain cell data
        indices = V.dofmap().cell_dofs(b)
        u_i = np.array([arr[index] for index in indices])

        # calculate alpha
        alpha = 1.0
        for c in range(ele_dof):
            if u_i[c] > u_c[b]: 
                if u_i_max[b+c] != u_c[b]: 
                    alpha = u_i[c] - u_c[b]# (u_i_max[b+c] - u_c[b])/(u_i[c] - u_c[b])
                else:
                    alpha = 0
                arr[indices[c]] = alpha

        # # apply slope limiting
        # slope = u_i - u_c[b]
        # for c in range(ele_dof):
        #     arr[indices[c]] = u_c[b] + alpha*slope[c]

    f.vector()[:] = arr

    if annotate:
        annotate_slope_limit(f)

def annotate_slope_limit(f):
    # First annotate the equation
    adj_var = adjglobals.adj_variables[f]
    rhs = SlopeRHS(f)

    adj_var_next = adjglobals.adj_variables.next(f)

    V = f.function_space()
    identity_block = solving.get_identity(V)

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

          # create storage arrays for max, min and mean values
          ele_dof = 2 
          n_dof = ele_dof * len(mesh.cells())

          u_i_max = np.ones([len(mesh.cells()) + 1]) * -1e200
          u_c = np.empty([len(mesh.cells())])

          # for each vertex in the mesh store the mean values
          for b in range(len(mesh.cells())):
              indices = V.dofmap().cell_dofs(b)

              u_i = np.array([arr[index] for index in indices])
              u_c[b] = u_i.mean()

              if (u_c[b] > u_i_max[b]):
                  u_i_max[b] = u_c[b]
              u_i_max[b+1] = u_c[b]

          # apply slope limit
          for b in range(len(mesh.cells())):

              # calculate alpha for first index in element
              for d in range(ele_dof):
                  index = V.dofmap().cell_dofs(b)[d]
                  if arr[index] > u_c[b]:
                      val = 0
                      if u_i_max[b+d] != u_c[b]: 
                          if d == 0:
                              indices_u = V.dofmap().cell_dofs(b)
                              indices_v = V.dofmap().cell_dofs(b-1)
                          else:
                              indices_u = reverse(V.dofmap().cell_dofs(b))
                              indices_v = V.dofmap().cell_dofs(b+1)
                          u_i = np.array([arr[index] for index in indices_u])
                          v_i = np.array([arr[index] for index in indices_v])
                          c_ui = np.array([c_arr[index] for index in indices_u])
                          c_vi = np.array([c_arr[index] for index in indices_v])

                          # B = (u_i[0] - u_i[1])/2.
                          # A = v_i.mean() - u_i.mean()
                          # val += 1/(2.*B) * c_vi[0].sum()
                          # val += -(B+A)/(2*B**2.0) * c_ui[0]
                          # val += -(B-A)/(2*B**2.0) * c_ui[1]
                          
                          val += (c_ui[0] - c_ui[1])/2.0
                  else:
                      val = c_arr[index]
                  out[index] = val

              # # apply slope limiting
              # slope = u_i - u_c[b]
              # for c in range(ele_dof):
              #     arr[indices[c]] = u_c[b] + alpha*slope[c]


          f.vector()[:] = out

          return adjlinalg.Vector(f)                                                  

# Annotate the forward model 
ic = project(Expression("sin(x[0])"), V, name="ic")
f = main(ic)
print f.vector().array()
adj_html("forward.html", "forward")
parameters["adjoint"]["stop_annotating"] = True

# Replay test
success = replay_dolfin(tol=0.0, stop=True)
assert success

# Taylor test
J = Functional(inner(f, f)*dx)
m = InitialConditionParameter(ic)
dj = compute_gradient_tlm(J, m, forget=False)

def Jhat(ic):
    f = main(ic, annotate=False)
    return assemble(inner(f, f)*dx)

minconv = taylor_test(Jhat, m, Jhat(ic), dj, seed=1e-9, perturbation_direction=interpolate(Expression("sin(x[0])"), V))
assert minconv > 1.99


