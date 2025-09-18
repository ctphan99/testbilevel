import numpy as np
from cvxopt import matrix, solvers


class Problem:
    def __init__(self, 
                x_dim, y_dim, cons_dim,
                Pu, Qu, Ru, cu, du, eu,
                Pl, Ql, Rl, cl, dl, el,
                A, B, b):
        
        # Dimensions
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.cons_dim = cons_dim

        # UL
        self.Pu = Pu
        self.Qu = Qu
        self.Ru = Ru
        self.cu = cu
        self.du = du
        self.eu = eu

        # LL
        self.Pl = Pl
        self.Ql = Ql
        self.Rl = Rl
        self.cl = cl
        self.dl = dl
        self.el = el

        ## LL Constraints
        self.A = A
        self.B = B
        self.b = b

    # Objectives
    def f(self, x,y):
        ff = 0.5*x.T@self.Pu@x + x.T@self.Qu@y + 0.5*y.T@self.Ru@y + self.cu.T@x + self.du.T@y + self.eu
        return ff

    def g(self,x,y):
        gg = 0.5*x.T@self.Pl@x + x.T@self.Ql@y + 0.5*y.T@self.Rl@y + self.cl.T@x + self.dl.T@y + self.el
        return gg

    # Gradients
    def gradx_f(self, x,y): return self.Pu@x + self.Qu@y + self.cu
    def grady_f(self, x,y): return self.Qu.T@x + self.Ru@y + self.du
    def gradx_g(self, x,y): return self.Pl@x + self.Ql@y + self.cl
    def grady_g(self, x,y): return self.Ql.T@x + self.Rl@y + self.dl
    def grady_g_pert(self, x,y): return self.Ql.T@x + self.Rl@y + self.dl

    # Hessians
    def hessxx_f(self, x,y): return self.Pu
    def hessxy_f(self, x,y): return self.Qu.T
    def hessyx_f(self, x,y): return self.Qu
    def hessyy_f(self, x,y): return self.Ru
    def hessxx_g(self, x,y): return self.Pl
    def hessxy_g(self, x,y): return self.Ql.T
    def hessyx_g(self, x,y): return self.Ql
    def hessyy_g(self, x,y): return self.Rl

    ## LL solution
    # Compute solution of LL problem
    def solve_ll(self, x):
        # Unconstrained
        if self.A is None or self.b is None:
            y = - np.linalg.inv(self.Rl)@(self.Ql.T@x + self.dl)
        # Constrained
        else:
            y = self.solve_constrained(x)
        return y
    
    # Find min of the constrained problem
    def solve_constrained(self, x): 

        solvers.options['feastol'] = 1e-12
        # Supress messages
        solvers.options['show_progress'] = False

        # Setup problem
        m = x.shape[0]
        Ac = matrix(self.A.astype(float), tc='d')
        bc = matrix((self.b-self.B@x).astype(float), tc='d')
        P = matrix(self.Rl.astype(float), tc='d')
        q = matrix((self.Ql.T@x + self.dl).astype(float), tc='d')

        sol = solvers.qp(P, q, Ac, bc)
        argmin = np.array(sol['x'])

        return argmin

    # Project on a polyhedron 
    def projy(self, x):
        if self.A is None or self.b is None:
            return x
        else:
            solvers.options['feastol'] = 1e-12
            # Supress messages
            solvers.options['show_progress'] = False

        # Setup problem
        m = x.shape[0]
        Ac = matrix(self.A.astype(float), tc='d')
        bc = matrix((self.b-self.B@x).astype(float), tc='d')
        P = matrix(np.eye(m).astype(float), tc='d')
        q = matrix((-x).astype(float), tc='d')

        # Solve problem
        sol = solvers.qp(P, q, Ac, bc)
        argmin = np.array(sol['x'])

        return argmin


def cvx_prob():
    # Dimensions
    x_dim, y_dim = 50, 50
    cons_dim = 10

    # UL 
    Pu = 2*np.eye(x_dim) 
    Qu = 0.1*np.random.rand(x_dim, y_dim)
    Ru = 2*np.eye(y_dim)
    cu = np.ones((x_dim, 1))
    du = np.ones((y_dim, 1))
    eu = 0.0

    # LL
    Pl = 2*np.eye(x_dim)
    Ql = np.random.rand(x_dim, y_dim)
    Rl = 2*np.eye(y_dim)
    cl = np.zeros((x_dim, 1))
    dl = np.zeros((y_dim, 1))
    el = 0.0

    # LL constraints
    A = np.random.rand(cons_dim, x_dim) # A x + B y - b \leq 0
    B = np.random.rand(cons_dim, y_dim) # A x + B y - b \leq 0
    b = np.random.rand(cons_dim, 1)

    return Problem(x_dim, y_dim, cons_dim, Pu, Qu, Ru, cu, du, eu, Pl, Ql, Rl, cl, dl, el, A, B, b)
    