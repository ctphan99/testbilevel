import numpy as np
import time, copy


class Algorithm:
    def __init__(self, problem):
        self.prob = problem


    # Compute the matrix that corresponds to the active set 
    def active(self, x, y):
        eps = 1e-3
        active_set = []
        if self.prob.A is None:
            return None
        else:
            for i in range(self.prob.A.shape[0]):
                if -eps < (self.prob.A[i, :]@y + self.prob.B[i, :]@x - self.prob.b[i]) <=0:
                    active_set.append(i)
            
        if len(active_set) == 0:
            Aact = None
            Bact = None
        else:
            Aact = self.prob.A[active_set,:]
            Bact = self.prob.B[active_set,:]

        return Aact, Bact


    # Compute gradient of lambda_star
    def grad_lambdastar(self, x, y, Aact, Bact):
        hessyy_g_inv = np.linalg.inv(self.prob.hessyy_g(x,y))
        g = - np.linalg.inv(Aact@hessyy_g_inv@Aact.T) @ (Aact@hessyy_g_inv@self.prob.hessxy_g(x,y)-Bact)
        return g 


    # Compute gradient of y star
    def grad_ystar(self, x, y, Aact, Bact):
        if Aact is None:
            g = -np.linalg.inv(self.prob.hessyy_g(x,y)) @ self.prob.hessxy_g(x,y)
        else:
            g = np.linalg.inv(self.prob.hessyy_g(x,y)) @ (-self.prob.hessxy_g(x,y)-Aact.T@self.grad_lambdastar(x, y, Aact, Bact)) 
        return g


    # Compute implicit gradient
    def grad_F(self, x, y, Aact, Bact):
        g = self.prob.gradx_f(x,y) + self.grad_ystar(x,y, Aact, Bact).T@self.prob.grady_f(x,y)
        return g


    def g_pert(self, x, y, q):
        pert = q.T@y
        gg = self.g(x, y) + pert
        return gg
    

    def grady_g_pert(self, x, y, q):
        return self.prob.grady_g(x, y) + q
    

    def eval(self, x, y):
        self.x_iter.append(x)
        self.y_iter.append(y)
        ystar = self.prob.solve_ll(x)
        self.gradF.append(np.linalg.norm(self.grad_F(x, ystar, *self.active(x, ystar))))
        self.loss.append(float(self.prob.f(x, ystar)))
        if self.start_time == 0:
            self.iter_time.append(0)
        else:
            self.iter_time.append(time.time() - self.start_time)


    def reset_eval(self):
        self.x_iter = []
        self.y_iter = []
        self.gradF = []
        self.loss = []
        self.iter_time = []
        self.start_time = 0


class DsBlo(Algorithm):
    def __init__(self, problem, out_iter, gamma1, gamma2, beta):
        super().__init__(problem)

        # Parameters
        self.out_iter = out_iter
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.beta = beta
        
        # Eval
        self.x_iter = []
        self.y_iter = []
        self.gradF = []
        self.loss = []
        self.iter_time = []
        self.start_time = 0


    def run(self, x, y):

        # 0th iteration
        self.eval(x, y)
        self.start_time = time.time()

        # Sample q1 ∼ Q
        q = np.expand_dims(np.random.randn(self.prob.y_dim), axis=1)
        self.q = 1e-6*(q/np.linalg.norm(q))
        
        # Find an approximate solution yhat_{q1}(x1) of (2.1b) s.t. Assumption 4 is satisfied
        yhat = self.prob.projy(np.expand_dims(np.random.rand(self.prob.y_dim), axis=1))
        yhat = self.prob.solve_ll(x)

        # Compute m1 = g1 = nablahat F_{q1}(x1)
        Aact, Bact = self.active(x, yhat)
        m = self.grad_F(x, yhat, Aact, Bact)

        for i in range(self.out_iter):

            # Update x_{t+1} = x_t − η_t*m_t
            eta = 1/(self.gamma1*np.linalg.norm(m) + self.gamma2)
            x_prev = copy.deepcopy(x)
            x = x - eta*m

            # Sample xbar_{t+1} ∼ U[xt, xt+1]
            xbar = np.zeros_like(x)
            for j in range(x.shape[0]):
                xbar[j] = np.random.uniform(x_prev[j], x[j], 1)

            # Sample qt+1 ∼ Q independently from qt
            q = np.expand_dims(np.random.randn(self.prob.y_dim), axis=1)
            self.q = 1e-6*(q/np.linalg.norm(q))

            # Find an approximate solution yhat_{q_{t+1}}(xbar_{t+1}) of (2.1b) s.t. Assumption 4 is satisfied
            yhat = self.prob.projy(np.expand_dims(np.random.rand(self.prob.y_dim), axis=1))
            yhat = self.prob.solve_ll(xbar)

            # Compute g
            Aact, Bact = self.active(x, yhat)
            g = self.grad_F(xbar, yhat, Aact, Bact) 

            # Update
            m = self.beta*m + (1-self.beta)*g

            self.eval(x, y)


class Ssigd(Algorithm):

    def __init__(self, problem, out_iter, out_step):
        super().__init__(problem)

        # Parameters
        self.out_iter = out_iter
        self.out_step = out_step
        
        q = np.expand_dims(np.random.randn(self.prob.y_dim), axis=1)
        self.q = 1e-3*(q/np.linalg.norm(q))
        
        # Eval
        self.x_iter = []
        self.y_iter = []
        self.gradF = []
        self.loss = []
        self.iter_time = []
        self.start_time = 0


    def run(self, x, y):

        # 0th iteration
        self.eval(x, y)
        self.start_time = time.time()

        for i in range(self.out_iter):

            # steps in LL 
            y = self.prob.projy(np.expand_dims(np.random.rand(self.prob.y_dim), axis=1))
            y = self.prob.solve_ll(x)

            # step in the UL (one GD step)
            Aact, Bact = self.active(x, y)
            x = x-self.out_step*self.grad_F(x, y, Aact, Bact)

            self.eval(x, y)