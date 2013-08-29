from dolfin import *
from dolfin_adjoint import *
import libadjoint
import hashlib

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

def main(ic, annotate=True):
    f = project(ic, V, annotate=annotate, name="f")
    slope_limit(f, annotate=annotate)
    return f

def slope_limit(f, annotate=True):
    f.vector()[:] = f.vector().array()*2
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

          d = Function(values[0].data)
          d.vector()[:] = 2*contraction_vector.data.vector().array()
          #plot(contraction_vector.data, interactive=True, title="conctract")
          #plot(d, interactive=True, title="d")
          return adjlinalg.Vector(d)

                                                                                   

# Annotate the forward model 
ic = project(Constant(1), V, name="ic")
f = main(ic)
adj_html("forward.html", "forward")
parameters["adjoint"]["stop_annotating"] = True

# Replay tset
success = replay_dolfin(tol=0.0, stop=True)
assert success

# Taylor test
J = Functional(inner(f, f)*dx)
m = InitialConditionParameter(ic)
dj = compute_gradient(J, m, forget=False)

def Jhat(ic):
    f = main(ic, annotate=False)
    return assemble(inner(f, f)*dx)

minconv = taylor_test(Jhat, m, Jhat(ic), dj)
assert minconv > 1.99


