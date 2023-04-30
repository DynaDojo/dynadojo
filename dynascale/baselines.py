import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.losses import mean_squared_error

from .abstractions import Model as MyModel


class NaiveLinearRegression(MyModel):
    def __init__(self, latent_dim, embed_dim, timesteps):
        super().__init__(latent_dim, embed_dim, timesteps)
        self.exp_At = np.zeros((embed_dim, embed_dim))


    def fit(self, x: np.ndarray, **kwargs):
        head = x[:, :-1].reshape(self.embed_dim, -1)
        tail = x[:, 1:].reshape(self.embed_dim, -1)
        self.exp_At = tail @ np.linalg.pinv(head)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs):
        preds = [x0]
        for _ in range(timesteps - 1):  # TODO: add general extensions for timesteps that are longer or shorter
            xi = preds[-1] @ self.exp_At  # TODO: check if multiplication should be other way
            preds.append(xi)
        preds = np.array(preds)
        preds = np.transpose(preds, axes=(1, 0, 2))
        return preds


class _KoopmanLayer(Layer):
    def __init__(self, dim, timesteps):
        super(_KoopmanLayer, self).__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.kernel = None

    def build(self, _):
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.dim, self.dim],
            regularizer="l2"
        )

    def call(self, inputs, **kwargs):
        matrix_exponentials = tf.convert_to_tensor([tf.linalg.expm(self.kernel * t) for t in range(self.timesteps)])
        trajs = tf.einsum('ijk,lk->lij', matrix_exponentials, inputs[:, 0, :])
        return trajs

class Koopman(MyModel):
    def __init__(self, latent_dim, embed_dim, timesteps, encoder_hidden_widths=(80, 80),
                 decoder_hidden_widths=(80, 80), alpha1=0.01, alpha2=0.01):
        super().__init__(latent_dim, embed_dim, timesteps)

        self._encoder = self._build_coder("encoder", (None, self.embed_dim), encoder_hidden_widths,
                                          self.latent_dim)
        self._decoder = self._build_coder("decoder", (None, self.latent_dim), decoder_hidden_widths,
                                          self.embed_dim)
        self.autoencoder = self._build_autoencoder()
        self._koopman = self._build_koopman(self.latent_dim)
        self.model = self._build_model(alpha1, alpha2)
        self.autoencoder.compile(optimizer="Adam", loss="mse")
        self.model.compile(optimizer="Adam", loss=None)

    @staticmethod
    def _build_coder(name: str, input_dim, hidden_widths, output_dim):
        inp = Input(shape=input_dim, name=f"{name}_input")
        x = inp
        for i, w in enumerate(hidden_widths):
            x = Dense(w, activation="relu", kernel_regularizer="l2", name=f"{name}_hidden{i}")(x)
        out = Dense(output_dim, activation="linear", kernel_regularizer="l2", name=f"{name}_output")(x)
        return Model(inp, out, name=name)

    def _build_autoencoder(self):
        autoencoder_input = Input(shape=(None, self.embed_dim), name="autoencoder_input")
        autoencoder_output = self._decoder(self._encoder(autoencoder_input))
        return Model(autoencoder_input, autoencoder_output, name="autoencoder")

    def _build_koopman(self, dim):
        koopman_input = Input(shape=(1, dim), name="koopman_input")
        koopman_output = _KoopmanLayer(dim, self.timesteps)(koopman_input)
        return Model(koopman_input, koopman_output, name="koopman")

    def _build_model(self, alpha1, alpha2):
        """
        encoder -> koopman -> decoder
        """
        x_true = Input(shape=(None, self.embed_dim), name="x_true")
        x0 = Input(shape=(1, self.embed_dim), name="model_input")

        encoded = self._encoder(x0)
        advanced = self._koopman(encoded)
        decoded = self._decoder(advanced)
        model = Model([x_true, x0], decoded, name="model")

        # custom loss function
        mse = tf.keras.losses.MeanSquaredError()
        x0_recon = self._decoder(encoded)
        L_recon = mse(x0, x0_recon)
        L_pred = tf.reduce_sum(mean_squared_error(x_true, decoded)) / self.timesteps  # TODO: make variable amount of timestep prediction
        L_lin = tf.reduce_sum(mean_squared_error(self._encoder(x_true), advanced)) / self.timesteps
        L_inf = tf.norm(x0 - x0_recon, ord=np.inf) + tf.norm(x_true[:, 1, :] - decoded[:, 1, :], ord=np.inf)
        L = alpha1 * (L_recon + L_pred) + L_lin + alpha2 * L_inf
        model.add_loss(L)

        # add metrics
        model.add_metric(L_recon, name='reconstruction_loss')
        model.add_metric(L_pred, name='state_prediction_loss')
        model.add_metric(L_lin, name='linear_dynamics_loss')
        model.add_metric(L_inf, name='infinity_norm')

        return model

    def _fit_autoencoder(self, x: np.ndarray, epochs, batch_size, verbose="auto"):
        x = tf.convert_to_tensor(x)
        self.autoencoder.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def _fit_model(self, x: np.ndarray, epochs, batch_size, verbose="auto"):
        x = tf.convert_to_tensor(x)
        x0 = x[:, :1, :]
        self.model.fit([x, x0], x, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def fit(self, x: np.ndarray, autoencoder_epochs: int = 10, model_epochs: int = 10, batch_size: int = 100, verbose="auto", **kwargs):
        self._fit_autoencoder(x, autoencoder_epochs, batch_size, verbose)
        self._fit_model(x, model_epochs, batch_size, verbose)

    def predict(self, x0: np.ndarray, timesteps: int, verbose="auto", **kwargs) -> np.ndarray:
        assert x0.ndim == 2
        num_examples, dim = x0.shape
        x0 = tf.convert_to_tensor(np.expand_dims(x0, axis=1))
        dummy = np.ones((num_examples, self.timesteps, dim))
        return self.model.predict([dummy, x0], verbose=verbose)