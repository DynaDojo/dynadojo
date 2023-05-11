import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from functools import partial


from ..abstractions import Model



class Koopman(Model):
    def __init__(self, latent_dim, embed_dim, timesteps, **kwargs):
        super().__init__(latent_dim, embed_dim, timesteps, **kwargs)

        dense = partial(layers.Dense, activation="relu", kernel_regularizer="l2")

        self.encoder = dense(latent_dim)
        self.decoder = dense(embed_dim)

        x0 = keras.Input(shape=(self.embed_dim))
        encoded_x0 = self.encoder(x0)
        decoded_x0 = self.decoder(encoded_x0)
        self.autoencoder = keras.Model(x0, decoded_x0)
        self.autoencoder.compile(loss="mse", optimizer="Adam")


    def fit(self, x: np.ndarray, *args, **kwargs):
        x = x.reshape((-1, self.embed_dim))
        self.autoencoder.fit(x=x, y=x, epochs=1000, batch_size=self.timesteps)
        # x = x.reshape((-1, self.embed_dim))
        # self.model.fit(x=x[:-1], y=x[1:])

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        self.model.predict(x0)