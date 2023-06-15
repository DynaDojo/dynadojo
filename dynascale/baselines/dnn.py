import numpy as np
import tensorflow as tf

from ..abstractions import AbstractModel


class DNN(AbstractModel):

    def __init__(self, embed_dim, timesteps, max_control_cost, activation=None, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(None, embed_dim)),
            tf.keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            tf.keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            tf.keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            tf.keras.layers.Dense(10, activation="linear"),
            tf.keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            tf.keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            tf.keras.layers.Dense(30, activation=activation, kernel_regularizer="l2"),
            tf.keras.layers.Dense(embed_dim, kernel_regularizer="l2")
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

