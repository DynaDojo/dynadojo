import sys
import numpy as np
from scipy.stats import ortho_group
from scipy.integrate import solve_ivp
import tensorflow as tf

from ..abstractions import AbstractSystem
from abc import abstractmethod


class SimpleSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 # singular values are non-neg by convention; >0 since we don't want a nontrivial null space
                 embedder_sv_range=(0.1, 1),
                 controller_sv_range=(0.1, 1),
                 IND_range=(0, 10),
                 OOD_range=(-10, 0),
                 noise_scale=0.01,
                 t_range=(0, 1),
                 ):
        super().__init__(latent_dim, embed_dim)

        self._t_range = t_range

        self.IND_range = IND_range
        self.OOD_range = OOD_range

        self._noise_scale = noise_scale
        self._rng = np.random.default_rng()

        self._embedder_sv_range = embedder_sv_range
        self._controller_sv_range = controller_sv_range
        self._embedder = None
        self._controller = None
        self._update_embedder_and_controller()

    @property
    def embedder(self):
        return self._embedder

    @property
    def controller(self):
        return self._controller

    def _update_embedder_and_controller(self):
        self._embedder = self._sv_to_matrix(self.latent_dim, self.embed_dim, self._embedder_sv_range)
        self._controller = self._sv_to_matrix(self.latent_dim, self.embed_dim, self._controller_sv_range)

    @AbstractSystem.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self._update_embedder_and_controller()

    @AbstractSystem.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self._update_embedder_and_controller()

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        """Uniformly samples embedded-dimensional points from an inside or outside distribution"""
        init_cond_range = self.IND_range if in_dist else self.OOD_range
        return self._rng.uniform(*init_cond_range, (n, self.embed_dim))

    def calc_error(self, x, y) -> float:
        """Returns NSE"""
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        """Calculates the L2 norm / dimension of every vector in the control"""
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self.embed_dim

    def calc_dynamics(self, t, x):
        raise NotImplementedError

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        init_conds = init_conds @ np.linalg.pinv(self.embedder)
        time = np.linspace(self._t_range[0], self._t_range[1], num=timesteps)

        def dynamics(t, x, u):
            i = np.argmin(np.abs(t - time))
            dx = self.calc_dynamics(t, x) + self.controller @ u[i]
            if noisy:
                dx += self._rng.normal(scale=self._noise_scale, size=self.latent_dim)
            return dx

        for x0, u in zip(init_conds, control):
            sol = solve_ivp(dynamics, t_span=[self._t_range[0], self._t_range[1]], y0=x0, t_eval=time, dense_output=True, args=(u,))
            data.append(sol.y)

        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.embedder
        return data

    def _sv_to_matrix(self, m, n, sv_range):
        U = ortho_group.rvs(m)
        sigma = np.eye(m, n) * self._rng.uniform(*sv_range, size=n)
        V = ortho_group.rvs(n)
        N = U @ sigma @ V
        return N

class OpinionSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale,
                 IND_range, 
                 OOD_range,
                 seed=None):

        super().__init__(latent_dim, embed_dim)

        assert embed_dim == latent_dim
        self._rng = np.random.default_rng(seed)

        self.noise_scale = noise_scale
        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def create_model(self, x0):
        return

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            if in_dist:
                x0.append({node: np.random.uniform(
                    self.IND_range[0], self.IND_range[1]) for node in range(self.latent_dim)})
            else:
                x0.append({node: np.random.uniform(
                    self.OOD_range[0], self.OOD_range[1]) for node in range(self.latent_dim)})
        return x0 

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        if noisy:
            noise = np.random.normal(
                0, self.noise_scale, (self.latent_dim))
        else:
            noise = np.zeros((self.latent_dim))

        def dynamics(x0):
            self.create_model(x0)

            iterations = self.model.iteration_bunch(timesteps)
            dX = []
            for iteration in iterations:
                step = []
                for idx in range(self.latent_dim):
                    if (idx in iteration["status"]):
                        step.append(iteration["status"][idx])
                    else:
                        step.append(dX[-1][idx])
                dX.append(step + noise)
            return dX

        if control:
            for x0, u in zip(init_conds, control):
                sol = dynamics(x0)
                data.append(sol)

        else:
            for x0 in init_conds:
                sol = dynamics(x0)
                data.append(sol)

        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

class EpidemicSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale,
                 IND_range, 
                 OOD_range,
                 group_status,
                 seed=None):

        super().__init__(latent_dim, embed_dim)

        if not group_status:
            assert embed_dim == latent_dim

        self._rng = np.random.default_rng(seed)

        self.noise_scale = noise_scale
        self.IND_range = IND_range
        self.OOD_range = OOD_range
        self.group_status = group_status

    def create_model(self, x0):
        return

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            if in_dist:
                x0.append({node: int(np.random.uniform(
                    self.IND_range[0], self.IND_range[1])) for node in range(self.latent_dim)})
            else:
                x0.append({node: int(np.random.uniform(
                    self.OOD_range[0], self.OOD_range[1])) for node in range(self.latent_dim)})

        return x0 

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        if noisy:
            noise = np.random.normal(
                0, self.noise_scale, (self.latent_dim))
        else:
            noise = np.zeros((self.latent_dim))

        def dynamics(x0):
            self.create_model(x0)

            iterations = self.model.iteration_bunch(timesteps)
            dX = []
            for iteration in iterations:
                if(self.group_status):
                    step = [val for _, val in iteration['node_count'].items()]
                    dX.append(step)
                else:
                    step = []
                    for idx in range(self.latent_dim):
                        if (idx in iteration["status"]):
                            step.append(iteration["status"][idx])
                        else:
                            step.append(dX[-1][idx])
                    dX.append([int(x) for x in (step + noise)])
            return dX

        if control:
            for x0, u in zip(init_conds, control):
                sol = dynamics(x0)
                data.append(sol)

        else:
            for x0 in init_conds:
                sol = dynamics(x0)
                data.append(sol)

        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

'''
Forward-Backward Neural Network Systems, adapted from Maziar Raissi https://github.com/maziarraissi/FBSNNs
'''
class FBSNNSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale,
                 IND_range, 
                 OOD_range,
                 layers,
                 T,
                 seed=None):

        super().__init__(latent_dim, embed_dim)

        assert embed_dim == 1
        
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()

        self.T = T # terminal time
        self.latent_dim = latent_dim # number of dimensions
        
        
        # layers
        if layers:
            self.layers = layers
        else:
            self.layers = [self.latent_dim+1] + 4*[256] + [1] # (latent_dim+1) --> 1
        
       
        
        self.IND_range = IND_range
        self.OOD_range = OOD_range
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
    
    def _initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self._xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def _xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev), dtype=tf.float32)
    
    def _neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def _net_u(self, t, X): # N x 1, N x latent_dim
        u = self._neural_net(tf.concat([t,X], 1), self.weights, self.biases) # N x 1
        Du = tf.gradients(u, X)[0] # N x latent_dim
        
        return u, Du

    def _Dg_tf(self, X): # N x latent_dim
        return tf.gradients(self._g_tf(X), X)[0] # N x latent_dim
        
    def _loss_function(self, t, W, X0, noisy): # N x (timesteps) x 1, N x (timesteps) x latent_dim, 1 x latent_dim
        loss = 0
        X_list = []
        Y_list = []
        
        t0 = t[:,0,:]
        W0 = W[:,0,:]
        Y0, Z0 = self._net_u(t0,X0) # N x 1, N x latent_dim
        
        X_list.append(X0)
        Y_list.append(Y0)

        if noisy:
            noise = self._rng.normal(
                    0, self.noise_scale, (X0.shape[0], X0.shape[1]))
        else:
            noise = 0
        
        for n in range(0,self.timesteps):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            X1 = X0 + self._mu_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.squeeze(tf.matmul(self._sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1])
            Y1_tilde = Y0 + self._phi_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self._sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True)
            Y1, Z1 = self._net_u(t1,X1)
            
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))
            
            t0 = t1
            W0 = W1
            X0 = X1 + noise
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)
            
        loss += tf.reduce_sum(tf.square(Y1 - self._g_tf(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self._Dg_tf(X1)))

        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        
        return loss, X, Y, Y[0,0,0]

    def _fetch_minibatch(self):
        T = self.T
        
        Dt = np.zeros((self.N,self.timesteps+1,1)) # N x (timesteps) x 1
        DW = np.zeros((self.N,self.timesteps+1,self.latent_dim)) # N x (timesteps) x latent_dim
        
        dt = T/self.timesteps
        
        Dt[:,1:,:] = dt
        np.random.seed(self.DW_seed)
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(self.N,self.timesteps,self.latent_dim))
        
        t = np.cumsum(Dt,axis=1) # N x timesteps x 1
        W = np.cumsum(DW,axis=1) # N x timesteps x latent_dim
        
        return t, W
                
    
    def _predict(self, Xi_star, t_star, W_star):
        
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}
        
        X_star = self.sess.run(self.X_pred, tf_dict)
        Y_star = self.sess.run(self.Y_pred, tf_dict)
        
        return X_star, Y_star
        

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        # initialize NN
        self.weights, self.biases = self._initialize_NN(self.layers)
        self.DW_seed = np.random.randint(0,100)

        X0 = []
        if in_dist:
            X0 = self._rng.uniform(self.IND_range[0], self.IND_range[1], (n, self.latent_dim))
        else:
            X0 = self._rng.uniform(self.OOD_range[0], self.OOD_range[1], (n, self.latent_dim))
        return X0
        

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        self.N = len(init_conds) 
        self.timesteps = timesteps

        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        # tf placeholders and graph (training)
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N, self.timesteps+1, 1]) # N x (timesteps) x 1
        self.W_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N, self.timesteps+1, self.latent_dim]) # N x (timesteps) x latent_dim
        self.Xi_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N, self.latent_dim]) # 1 x latent_dim

        self.loss, self.X_pred, self.Y_pred, self.Y0_pred = self._loss_function(self.t_tf, self.W_tf, self.Xi_tf, noisy)
    
        # initialize session and variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        t_test, W_test = self._fetch_minibatch()
    
        X_pred, _ = self._predict(init_conds, t_test, W_test)

        if control is not None:
            control = control.reshape(-1,control.shape[2])
            sol = np.reshape(self._solve(np.reshape(t_test[0:self.N,:,:],[-1,1]), np.reshape(X_pred[0:self.N,:,:],[-1,self.latent_dim]), self.T, control), [self.N,-1,1])
        else:
            U = np.zeros([self.N*(self.timesteps+1), 1])
            sol = np.reshape(self._solve(np.reshape(t_test[0:self.N,:,:],[-1,1]), np.reshape(X_pred[0:self.N,:,:],[-1,self.latent_dim]), self.T, U), [self.N,-1,1])
        
        return sol

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

    @abstractmethod
    def _solve(t, X, T, U): # (N+1) x 1, (N+1) x latent_dim
        pass

    @abstractmethod
    def _phi_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        pass # N x1
    
    @abstractmethod
    def _g_tf(self, X): # N x latent_dim
        pass # N x 1
    
    @abstractmethod
    def _mu_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return np.zeros([self.N,self.latent_dim]) # N x latent_dim
    
    @abstractmethod
    def _sigma_tf(self, t, X, Y): # N x 1, N x latent_dim, N x 1
        return tf.linalg.diag(tf.ones([self.N,self.latent_dim])) # N x latent_dim x latent_dim