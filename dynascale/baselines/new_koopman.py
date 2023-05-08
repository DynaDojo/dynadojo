import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from ..abstractions import Model



class Koopman(Model):
    def __init__(self, latent_dim, embed_dim, timesteps, **kwargs):
        super().__init__(latent_dim, embed_dim, timesteps, **kwargs)

        # encoder
        encoder_input = keras.Input(shape=(self.timesteps, self.embed_dim))
        x = layers.Dense(10, activation="relu", kernel_regularizer="l2")(encoder_input)
        encoder_output = layers.Dense(self.latent_dim, activation="relu", kernel_regularizer="l2")(x)

        # decoder
        x = layers.Dense(10, activation="relu", kernel_regularizer="l2")(encoder_output)
        decoder_output = layers.Dense(self.embed_dim, activation="relu", kernel_regularizer="l2")(x)

        # autoencoder
        self.autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
        self.autoencoder.compile(
            loss="mse",
            optimizer="Adam"
        )

        # koopman


    def fit(self, x: np.ndarray, *args, **kwargs):
        self.autoencoder.fit(x=x, y=x, batch_size=100, epochs=1000)
