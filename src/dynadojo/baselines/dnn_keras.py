"""
Deep Neural Network (DNN)
===========================
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..abstractions import AbstractAlgorithm


class DNN(AbstractAlgorithm):
    """Deep Neural Network (DNN). Contains 5 hidden layers with 30 neurons each."""

    def __init__(
            self,
            embed_dim,
            timesteps,
            max_control_cost=0,
            activation='relu',
            seed=None,
            **kwargs):
        """
        Initialize the class.

        Parameters
        -------------
        embed_dim : int
            The embedded dimension of the system. Recommended to keep embed dimension small (e.g., <10).
        timesteps : int
            The timesteps of the training trajectories. Must be greater than 2.
        activation : str, optional
            The activation function used in the hidden layers. See ``tensorflow`` documentation for more details on
            acceptable activations. Defaults to ``relu``.
        max_control_cost : float, optional
            Ignores control, so defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments
        """
        super().__init__(embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        if seed:
            keras.utils.set_random_seed(812)
            # tf.config.experimental.enable_op_determinism()
        kreg = "l2"
        self.model = tf.keras.Sequential([
            keras.Input(shape=(None, embed_dim)),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(embed_dim, kernel_regularizer=kreg)
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def fit(self, x: np.ndarray, epochs=2000, verbose=0, **kwargs):
        head = x[:, :-1, :]
        tail = x[:, 1:, :]
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(head, tail, validation_split=0.2, epochs=epochs, callbacks=[callback], verbose=verbose)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        preds = [x0]
        for _ in range(timesteps - 1):
            preds.append(self.model.predict(preds[-1], verbose=0))
        preds = np.array(preds)
        if np.ndim(preds) > 3:
            preds = preds.squeeze()
        preds = preds.transpose(1, 0, 2)
        return preds
