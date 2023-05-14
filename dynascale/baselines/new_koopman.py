import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.losses import mean_squared_error
from functools import partial


from ..abstractions import Model


class KoopmanLayer(layers.Layer):
    def __init__(self, dim, timesteps):
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.kernel = None

    def build(self, _):
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.dim, self.dim],
            regularizer="l2",
            trainable=True
        )

    def call(self, inputs, **kwargs):
        matrix_powers = [self.kernel]
        for _ in range(self.timesteps - 1):
            matrix_powers.append(matrix_powers[-1] @ self.kernel)
        matrix_powers = tf.convert_to_tensor(matrix_powers)
        trajs = tf.einsum('ijk,lk->lij', matrix_powers, inputs[:, 0])
        return trajs


class Koopman(Model):
    def __init__(self, latent_dim, embed_dim, timesteps, h: int = 30, alpha1: int = 0.01, alpha2: int = 0.02, **kwargs):
        super().__init__(latent_dim, embed_dim, timesteps, **kwargs)

        dense = partial(layers.Dense,
                        activation="relu",
                        kernel_initializer=tf.keras.initializers.Constant(1 / np.sqrt(h)),
                        kernel_regularizer="l2"
                        )

        self.encoder = tf.keras.Sequential([
            dense(h),
            dense(h),
            dense(h),
            layers.Dense(latent_dim, activation="linear", kernel_regularizer="l2")
        ])

        self.decoder = tf.keras.Sequential([
            dense(h),
            dense(h),
            dense(h),
            layers.Dense(embed_dim, activation="linear", kernel_regularizer="l2")
        ])

        x = tf.keras.Input(shape=(None, embed_dim), name="autoencoder_input")
        self.autoencoder = tf.keras.Model(x, self.decoder(self.encoder(x)))
        self.autoencoder.compile(loss="mse", optimizer="Adam")

        x0 = tf.keras.Input(shape=(1, embed_dim,))
        y = tf.keras.Input(shape=(timesteps, embed_dim))
        self.K = KoopmanLayer(self.latent_dim, timesteps)

        encoded = self.encoder(x0)
        advanced = self.K(encoded)
        x_pred = self.decoder(advanced)

        self.model = tf.keras.Model([x0, y], x_pred)

        x_recon = self.decoder(self.encoder(x0))

        # add losses
        L_recon = tf.reduce_sum(mean_squared_error(x0[:, 0], x_recon[:, 0]))
        L_pred = tf.reduce_sum(mean_squared_error(x_pred, y)) / timesteps
        L_lin = tf.reduce_sum(mean_squared_error(self.encoder(y), advanced)) / self.timesteps
        L_inf = tf.norm(x0[:, 0] - x_recon[:, 0], ord=np.inf) + tf.norm(y[:, 1] - x_pred[:, 1], ord=np.inf)
        L = alpha1 * (L_recon + L_pred) + L_lin + alpha2 * L_inf
        self.model.add_loss(L)

        # add metrics
        self.model.add_metric(L_recon, name='reconstruction_loss')
        self.model.add_metric(L_pred, name='state_prediction_loss')
        self.model.add_metric(L_lin, name='linear_dynamics_loss')
        self.model.add_metric(L_inf, name='infinity_norm')

        self.model.compile(loss=None, optimizer="Adam")

    def fit(self,
            x: np.ndarray,
            autoencoder_epochs: int = 100,
            model_epochs: int = 100,
            *args,
            **kwargs):

        self.autoencoder.fit(x, x, epochs=autoencoder_epochs, batch_size=10)
        self.model.fit([x[:, :1], x], x, epochs=model_epochs, batch_size=10)

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        return self.model.predict([x0[:, None], np.zeros((len(x0), timesteps, self.embed_dim))])
