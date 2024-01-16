from abc import abstractmethod

import numpy as np
import tensorflow as tf

from ...abstractions import AbstractSystem

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

        super().__init__(latent_dim, embed_dim, seed)

        #assert embed_dim == 1
        
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()

        self.T = T  # terminal time
        self.latent_dim = latent_dim  # number of dimensions
        
        
        if layers:
            self.layers = layers
        else:
            self.layers = [self.latent_dim+1] + 4*[256] + [1]  # (latent_dim+1) --> 1
        
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
        
        for n in range(0,self.timesteps-1):
            t1 = t[:,n,:]
            W1 = W[:,n,:]
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
        
        Dt = np.zeros((self.N,self.timesteps,1)) # N x (timesteps) x 1
        DW = np.zeros((self.N,self.timesteps,self.latent_dim)) # N x (timesteps) x latent_dim
        
        dt = T/self.timesteps
        
        Dt[:,1:,:] = dt
        np.random.seed(self.DW_seed)
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(self.N,self.timesteps-1,self.latent_dim))
        
        t = np.cumsum(Dt,axis=1) # N x timesteps x 1
        W = np.cumsum(DW,axis=1) # N x timesteps x latent_dim
        
        return t, W
                
    
    def _predict(self, Xi_star, t_star, W_star):
        
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}
        X_star = self.sess.run(self.X_pred, tf_dict)
        Y_star = self.sess.run(self.Y_pred, tf_dict)
        
        return X_star, Y_star
        
    # Expands initial conditions of X_old: (n, self.embed_dim) —– where self.embed_dim is always 1 —– 
    # to X_new: (n, self.latent_dim) where each new initial condition if solved exactly corresponds back to X_old
    def _expand_init_conds(self, x0):
        expanded = []
        for x in x0:
            expanded.append(self._unsolve_target(x, self.T, self.latent_dim))
        return expanded


    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        self.weights, self.biases = self._initialize_NN(self.layers)
        self.DW_seed = np.random.randint(0,100)

        X0 = []
        if in_dist:
            X0 = self._rng.uniform(self.IND_range[0], self.IND_range[1], (n, self.embed_dim))
        else:
            X0 = self._rng.uniform(self.OOD_range[0], self.OOD_range[1], (n, self.embed_dim))
        X0 = np.array(X0)
        return X0
        

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        self.N = len(init_conds) 
        self.timesteps = timesteps
        init_conds = self._expand_init_conds(init_conds)

        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        # tf placeholders and graph (training)
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N, self.timesteps, 1]) # N x (timesteps) x 1
        self.W_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N, self.timesteps, self.latent_dim]) # N x (timesteps) x latent_dim
        self.Xi_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N, self.latent_dim]) # 1 x latent_dim

        self.loss, self.X_pred, self.Y_pred, self.Y0_pred = self._loss_function(self.t_tf, self.W_tf, self.Xi_tf, noisy)
    
        # initialize session and variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        t_test, W_test = self._fetch_minibatch()
    
        X_pred, _ = self._predict(init_conds, t_test, W_test)

        if control is not None:
            control = control.reshape(-1,control.shape[2])
            sol = self._solve(np.reshape(t_test[0:self.N,:,:],[-1,1]), np.reshape(X_pred[0:self.N,:,:],[-1,self.latent_dim]), self.T, control)
            sol = np.reshape(sol, [self.N,-1,1])
        else:
            U = np.zeros([self.N*(self.timesteps), 1])
            sol = self._solve(np.reshape(t_test[0:self.N,:,:],[-1,1]), np.reshape(X_pred[0:self.N,:,:],[-1,self.latent_dim]), self.T, U)
            sol = np.reshape(sol, [self.N,-1,1])
        
        return sol

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

    @abstractmethod
    def _solve(t, X, T, U): # (N) x 1, (N) x latent_dim
        pass

    @abstractmethod
    def _unsolve_target(self, target, T, new_dim):
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