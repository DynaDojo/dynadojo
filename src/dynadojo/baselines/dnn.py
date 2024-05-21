"""
Deep Neural Network (DNN)
===========================
"""
from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import time

from ..abstractions import AbstractAlgorithm

import logging
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



class NNTorch(AbstractAlgorithm, torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            timesteps,
            max_control_cost=0,
            activation='relu',
            seed=None,
            **kwargs):
        AbstractAlgorithm.__init__(self, embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        torch.nn.Module.__init__(self)

        if seed:
            torch.manual_seed(seed)

        self.model = self.create_model()
        self.dummy_param = torch.empty(0)
        self.criterion = torch.nn.MSELoss()
    
    @abstractmethod
    def create_model(self):
        raise NotImplementedError

    @property
    def device(self):
        return self.dummy_param.device
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
    
    def fit(self, x: np.ndarray, 
            epochs=2000,
            batch_size=32, 
            lr=1e-3,
            validation_split=0.1,
            patience=3, 
            min_delta=0.0,
            start_early_stop_from_epoch=100,
            verbose=0, **kwargs):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)

        if validation_split == 0 or patience == 0:
            if verbose > 0:
                print('Training on the full dataset, no validation, no early stopping')
            train = np.array(x)
            train_size = len(x)
            early_stopper = None
        else:
            if verbose > 0:
                print(f'Training on {1-validation_split} of the data, validating on the rest')
                print(f'Early stopping from epoch {start_early_stop_from_epoch} with patience {patience} and min_delta {min_delta}')
            validation_size = int(x.shape[0] * validation_split)
            train_size = len(x) - validation_size
            train, val = random_split(x, [len(x)-validation_size, validation_size])
            train = np.array(train)
            val = np.array(train)

            #Validation dataset
            x_val = torch.tensor(np.array(val[:][:-1, :]), dtype=torch.float32).to(self.device) 
            y_val = torch.tensor(np.array(val[:][1:, :]), dtype=torch.float32).to(self.device) 

            early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, start_from_epoch=start_early_stop_from_epoch)

        if batch_size > train_size:
            batch_size = train_size
        x = torch.tensor(np.array(train[:][:-1, :]), dtype=torch.float32)
        y = torch.tensor(np.array(train[:][1:, :]), dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=4)

        losses = []
        self.train() 
        training_start_time = time.time()
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for input_seq, target_seq in dataloader:
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                y_hat = self.forward(input_seq)
                loss = self.criterion(y_hat, target_seq)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            losses.append(epoch_loss/len(dataloader))

            if early_stopper is None:
                if verbose > 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, took {time.time() - training_start_time:.2f}s')
                    training_start_time = time.time()
            else:
                self.eval()  # Set model to evaluation mode
                with torch.no_grad():  # Disable gradient calculation for validation
                    # Make predictions on validation set 
                    val_outputs = self(x_val).detach()
                    val_loss = self.criterion(val_outputs, y_val).item()
                
                    if verbose > 0:
                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}, , took {time.time() - training_start_time:.2f}s')
                        training_start_time = time.time()
                if early_stopper.early_stop(epoch, val_loss, self.state_dict()):
                    if verbose > 0:
                        print(f'Early stopping at epoch {epoch+1}')
                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}')
                    break
        
        self.load_state_dict(early_stopper.best_weights)
        return losses

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            trajectory = x0.reshape(x0.shape[0], 1, x0.shape[1])
            next_input = torch.tensor(trajectory, dtype=torch.float32).to(self.device)
            
            # Iteratively predict future values
            for _ in range(timesteps - 1):
                next_input = self.forward(next_input)
                to_add = next_input.cpu().numpy().reshape(next_input.shape[0], 1, next_input.shape[1])
                trajectory = np.concatenate([trajectory, to_add], axis=1)
        return trajectory          

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, start_from_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

    def early_stop(self, epoch, validation_loss, weights):
        if epoch < self.start_from_epoch:
            return False
        
        if weights is None:
            self.best_weights = weights
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_weights = weights
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class DNNTorch(NNTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_model(self):
        model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=0, end_dim=1),
            torch.nn.Linear(self.embed_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.embed_dim)
        )
        return model

    def forward(self, x):
        # The model will flatten the input so each timestep is a separate sample we will unflatten it back
        timesteps = x.shape[0]
        return self.model(x).unflatten(0, (timesteps, -1))