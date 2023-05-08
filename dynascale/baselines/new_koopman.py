import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from ..abstractions import Model



class Koopman(Model):
    def __init__(self, latent_dim, embed_dim, timesteps, **kwargs):
        super().__init__(latent_dim, embed_dim, timesteps, **kwargs)

        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation=tf.nn.relu, name='encoder_hidden'),
                tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.sigmoid, name='code')
            ],
            name='encoder'
        )

        # Decodes from low-dimensional code to output, handling any reshaping as necessary
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation=tf.nn.relu, name='decoder_hidden'),
                tf.keras.layers.Dense(self.embed_dim, activation=tf.keras.activations.linear, name='reconstructed'),
            ],
            name='decoder'
        )

        # Reads out the linear dynamics, `K` from the DeepKoopman paper
        linear_dynamics = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.latent_dim, activation=tf.keras.activations.linear, name='linear_dynamics_t1')
            ],
            name='linear_dynamics'
        )

        # Share encoder/decoder for time-points.
        # state_input has size (sequence_length, state_dim) to include t=k, t=k+1,...
        state_input = tf.keras.Input(shape=(timesteps, self.embed_dim), name='state')

        # Unstack state into separate time-points
        # Note that axis=0 is the batch dimension
        state_sequence = tf.unstack(state_input, axis=1)

        # Create the autoencoder graph, which only matters for t=k+0-th time-point
        code_sequence = []
        for state in state_sequence:
            code = encoder(state)
            code_sequence.append(code)
        reconstructed_state_t0 = decoder(code_sequence[0])

        # Feed-forward code through linear dynamics
        # # Option 1: ignore t=k+0, which is trivially correct
        # predicted_code_sequence = []
        predicted_code = code_sequence[0]
        # Option 2: include t=k+0, which makes the graph a little confusing but might make losses more relevant
        predicted_code_sequence = [predicted_code]
        for time_offset in range(1, len(code_sequence)):
            predicted_code = linear_dynamics(predicted_code)
            predicted_code_sequence.append(predicted_code)

        # Predict/reconstruct future state through the decoder
        predicted_state_sequence = [
            decoder(predicted_code) for predicted_code in predicted_code_sequence
        ]

        # Restack predictions across time
        codes = tf.stack(code_sequence, axis=1, name='codes')
        predicted_codes = tf.stack(predicted_code_sequence, axis=1, name='stack_predicted_codes')
        predicted_states = tf.stack(predicted_state_sequence, axis=1, name='stack_predicted_states')

        self.model = tf.keras.Model(
            inputs={'state_input': state_input},
            outputs={'reconstructed_state_t0': reconstructed_state_t0,  # for autoencoder loss
                     'predicted_codes': predicted_codes,  # for linear dynamics
                     'predicted_states': predicted_states,  # for state prediction loss and autoencoder loss
                     },
            name='DeepKoopman'
        )

        RECON_LOSS_WEIGHT = 0.01
        linear_dynamics_loss = tf.math.reduce_mean(tf.math.squared_difference(codes, predicted_codes))
        self.model.add_loss(linear_dynamics_loss)
        # DeepKoopman combines reconstruction (L_recon) and prediction loss (L_pred)
        reconstruction_prediction_loss = tf.math.reduce_mean(tf.math.squared_difference(state_input, predicted_states))
        self.model.add_loss(RECON_LOSS_WEIGHT * reconstruction_prediction_loss)
        # Note: this example does not yet include L_inf loss and l_2 regularization

        # Add metrics
        self.model.add_metric(linear_dynamics_loss, name='linear_dynamics_loss', aggregation='mean')
        self.model.add_metric(reconstruction_prediction_loss, name='reconstruction_prediction_loss', aggregation='mean')

        self.model.compile(optimizer='adam')

    def fit(self, x: np.ndarray, *args, **kwargs):
        self.model.fit(x=x, y=x, batch_size=100, epochs=1000)

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        self.model.predict(x0)