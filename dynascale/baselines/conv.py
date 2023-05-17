import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ..abstractions import Model



class Conv(Model):

    def __init__(self, embed_dim, timesteps, max_control_cost, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(None, embed_dim)),
            tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=10),
            tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=10),
            tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=10),
            tf.keras.layers.Dense(10, activation="linear"),
            tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=10),
            tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=10),
            tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=10),
            tf.keras.layers.Dense(embed_dim, kernel_regularizer="l2")
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def fit(self, x: np.ndarray, epochs=20, *args, **kwargs) -> np.ndarray:
        head = x[:, :-1, :]
        tail = x[:, 1:, :]
        self.model.fit(head, tail, epochs=epochs, verbose=0)

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        preds = [x0]
        for _ in tqdm(range(timesteps - 1), leave=False):
            preds.append(self.model.predict(preds[-1], verbose=0))
        preds = np.array(preds).squeeze().transpose(1, 0, 2)
        return preds

