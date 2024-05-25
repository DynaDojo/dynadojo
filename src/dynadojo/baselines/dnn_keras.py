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

"""In progress implementation of the Transformer model in Keras. Doesn't work because of issues with output shape. 

class EncoderTransformer(AbstractAlgorithm):
    def __init__(
            self,
            embed_dim,
            timesteps,
            max_control_cost=0,
            activation='relu',
            seed=None,
            **kwargs):

        # Initialize the class.

        # Parameters
        # -------------
        # embed_dim : int
        #     The embedded dimension of the system. Recommended to keep embed dimension small (e.g., <10).
        # timesteps : int
        #     The timesteps of the training trajectories. Must be greater than 2.
        # activation : str, optional
        #     The activation function used in the hidden layers. See ``tensorflow`` documentation for more details on
        #     acceptable activations. Defaults to ``relu``.
        # max_control_cost : float, optional
        #     Ignores control, so defaults to 0.
        # **kwargs : dict, optional
        #     Additional keyword arguments

        super().__init__(embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        if seed:
            keras.utils.set_random_seed(812)
            # tf.config.experimental.enable_op_determinism()
        
        input_shape = (None, embed_dim)
        self.model = model = self.build_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(
            self,
            input_shape,
            head_size,
            num_heads,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            dropout=0,
            mlp_dropout=0,
    ):  
        kreg = "l2"
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu", kernel_regularizer=kreg)(x)
            x = layers.Dropout(mlp_dropout)(x)

        outputs = layers.Dense(input_shape[-1], kernel_regularizer=kreg)(x)
        return keras.Model(inputs, outputs)

    def fit(self, 
            x: np.ndarray, 
            epochs=2000, 
            batch_size=32, 
            lr=1e-3,
            validation_split=0.2,
            patience=3,
            min_delta=0.0,
            start_early_stop_from_epoch=0,
            verbose=0, 
            **kwargs):
        head = x[:, :-1, :]
        tail = x[:, 1:, :]

        self.model.optimizer.lr = lr
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=patience, 
                                                    min_delta=min_delta, 
                                                    restore_best_weights=True,
                                                    start_from_epoch=start_early_stop_from_epoch)
        self.model.fit(head, tail, batch_size=batch_size, validation_split=validation_split, epochs=epochs, callbacks=[callback], verbose=verbose)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        n, d = x0.shape
        next = x0
        preds = x0.reshape(n, 1, d)
        for _ in range(timesteps - 1):
            next = self.model.predict(next.reshape(n, 1, d), verbose=0)
            preds = np.concatenate([preds, next], axis=1)
        return preds

"""