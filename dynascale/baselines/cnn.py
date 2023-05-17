import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from ..abstractions import Model

class CNN(Model):

    def __init__(self, embed_dim, timesteps, max_control_cost, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        model_input = tf.keras.layers.Input(shape=(None, embed_dim))
        x = tf.keras.layers.Conv1D(embed_dim, 10, activation="relu", padding="same")(model_input)
        x = tf.keras.layers.Conv1D(embed_dim, 10, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv1D(embed_dim, 10, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv1D(embed_dim, 10, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv1D(embed_dim, 10, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv1D(embed_dim, 10, activation="relu", padding="same")(x)
        x = tf.keras.layers.Dense(embed_dim, activation="linear")(x)
        self.model = tf.keras.Model(model_input, x)
        self.model.compile(optimizer="adam", loss="mae")

    def fit(self, x: np.ndarray, epochs=20, verbose=0, *args, **kwargs) -> np.ndarray:
        head = x[:, :-1, :]
        tail = x[:, 1:, :]
        self.model.fit(head, tail, epochs=epochs, verbose=verbose)

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        preds = [x0[:, None, :]]
        for _ in tqdm(range(timesteps - 1), leave=False):
            preds.append(self.model.predict(preds[-1], verbose=0))
        preds = np.array(preds).squeeze().transpose(1, 0, 2)
        preds = np.clip(preds, 0, 1)
        preds = np.round(preds)
        return preds