import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..abstractions import AbstractModel


class DNN(AbstractModel):

    def __init__(
        self, 
        embed_dim, 
        timesteps, 
        max_control_cost, 
        activation='relu', 
        seed=None, 
        **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        if seed:
            keras.utils.set_random_seed(812)
            # tf.config.experimental.enable_op_determinism()

        self.model = tf.keras.Sequential([
            keras.Input(shape=(None, embed_dim)),
            keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            keras.layers.Dense(10, activation="linear"),
            keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            keras.layers.Dense(embed_dim, kernel_regularizer="l2")
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def fit(self, x: np.ndarray, epochs=20, verbose=0, **kwargs):
        head = x[:, :-1, :]
        tail = x[:, 1:, :]
        self.model.fit(head, tail, epochs=epochs, verbose=verbose)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        preds = [x0]
        for _ in range(timesteps - 1):
            preds.append(self.model.predict(preds[-1], verbose=0))
        preds = np.array(preds).squeeze().transpose(1, 0, 2)
        return preds

