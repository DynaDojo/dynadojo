from ..abstractions import Model

import numpy as np
import tensorflow as tf

class Koopman(Model):
    def __init__(self, latent_dim, embed_dim, timesteps, h: int = 80,
                 alpha1: float = 0.01, alpha2: float = 0.01,
                 **kwargs):
        super().__init__(latent_dim, embed_dim, timesteps, **kwargs)

        encoder_input = tf.keras.layers.Input(shape=(None, embed_dim))
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer="l2")(encoder_input)
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer="l2")(x)
        encoder_output = tf.keras.layers.Dense(latent_dim, activation="linear", kernel_regularizer="l2")(x)
        self.encoder = tf.keras.Model(encoder_input, encoder_output)

        decoder_input = tf.keras.layers.Input(shape=(None, latent_dim))
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer="l2")(decoder_input)
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer="l2")(x)
        decoder_output = tf.keras.layers.Dense(embed_dim, activation="linear", kernel_regularizer="l2")(x)
        self.decoder = tf.keras.Model(decoder_input, decoder_output)

        autoencoder_input = tf.keras.layers.Input(shape=(None, embed_dim))
        autoencoder_output = self.decoder(self.encoder(autoencoder_input))
        self.autoencoder = tf.keras.Model(autoencoder_input, autoencoder_output)
        self.autoencoder.compile(loss="mse", optimizer="adam")

        koopman_input = tf.keras.layers.Input(shape=(1, latent_dim,))
        K = tf.keras.layers.Dense(latent_dim, activation="linear")
        koopman_output = [koopman_input]
        for _ in range(timesteps - 1):
            koopman_output.append(K(koopman_output[-1]))
        koopman_output = tf.concat(koopman_output, axis=1)
        self.koopman = tf.keras.Model(koopman_input, koopman_output)


        x0 = tf.keras.layers.Input(shape=(1, embed_dim))
        x_true = tf.keras.layers.Input(shape=(timesteps, embed_dim))

        encoded = self.encoder(x0)
        advanced = self.koopman(encoded)
        decoded = self.decoder(advanced)

        self.model = tf.keras.Model([x0, x_true], decoded)

        mse = tf.keras.losses.MeanSquaredError()

        L_recon = mse(x0, decoded)
        L_pred = tf.reduce_sum(tf.keras.losses.mean_squared_error(x_true, decoded)) / timesteps
        L_lin = tf.reduce_sum(tf.keras.losses.mean_squared_error(self.encoder(x_true), advanced)) / timesteps
        L_inf = tf.norm(x0 - decoded, ord=np.inf) + tf.norm(x_true[:, 1] - decoded[:, 1], ord=np.inf)  # TODO: change
        L = alpha1 * (L_recon + L_pred) + L_lin + alpha2 * L_inf
        self.model.add_loss(L)

        # add metrics
        self.model.add_metric(L_recon, name='reconstruction_loss')
        self.model.add_metric(L_pred, name='state_prediction_loss')
        self.model.add_metric(L_lin, name='linear_dynamics_loss')
        self.model.add_metric(L_inf, name='infinity_norm')

        self.model.compile(loss=None, optimizer="adam")


    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        return self.model.predict([x0[:, None], np.zeros((len(x0), timesteps, self.embed_dim))])


    def fit(self, x: np.ndarray, autoencoder_epochs: int = 100, model_epochs: int = 100, *args, **kwargs):
        self.autoencoder.fit(x, x, epochs=autoencoder_epochs)
        self.model.fit([x[:, :1], x], x, epochs=model_epochs)