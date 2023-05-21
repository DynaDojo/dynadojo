import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from ..abstractions import Model

class Basic(Model):

    def __init__(self, embed_dim, timesteps, max_control_cost, **kwargs):
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        model_input = tf.keras.layers.Input(shape=(None, embed_dim))
        x = tf.keras.layers.Dense(embed_dim, activation="tanh")(model_input)
        x = tf.keras.layers.Dense(embed_dim, activation="tanh")(x)
        x = tf.keras.layers.Dense(embed_dim, activation="tanh")(x)
        x = tf.keras.layers.Dense(embed_dim, activation="tanh")(x)
        model_output = tf.clip_by_value(x, 0, 1)
        self.model = tf.keras.Model(model_input, model_output)
        self.model.compile(optimizer="adam", loss="mse")


    def fit(self, x: np.ndarray, epochs=20, verbose=0, **kwargs) -> np.ndarray:
        head = x[:, :-1, :]
        tail = x[:, 1:, :]
        self.model.fit(head.astype(float), tail.astype(float), epochs=epochs, verbose=verbose)

    def _predict(self, x0: np.ndarray, timesteps: int, *args, **kwargs) -> np.ndarray:
        preds = [x0[:, None, :]]
        for _ in tqdm(range(timesteps - 1), leave=False):
            preds.append(self.model.predict(preds[-1], verbose=0))
        preds = np.array(preds).squeeze().transpose(1, 0, 2)
        preds = np.clip(preds, 0, 1)
        preds = np.round(preds)
        return preds