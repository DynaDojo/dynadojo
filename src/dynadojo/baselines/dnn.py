"""
Deep Neural Network (DNN)
===========================
"""
from abc import abstractmethod
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import time

from ..abstractions import AbstractAlgorithm



class TorchBaseClass(AbstractAlgorithm, torch.nn.Module):
    """
    Base class for all PyTorch neural network models.
    """
    def __init__(
            self,
            embed_dim,
            timesteps,
            max_control_cost=0,
            activation='relu',
            seed=None,
            device=None,
            **kwargs):
        AbstractAlgorithm.__init__(self, embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        torch.nn.Module.__init__(self)

        if seed:
            torch.manual_seed(seed)

        self.device = device or "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # self.model = self.create_model()
        self.criterion = torch.nn.MSELoss()
    
    # @abstractmethod
    # def create_model(self):
    #     raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
    
    def fit(self, x: np.ndarray, 
            epochs=5000,
            batch_size=32, 
            lr=1e-2,
            validation_split=0.1,
            patience=15, 
            min_delta=0.0,
            min_epochs=1000,
            verbose=0, **kwargs):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) #, weight_decay=1e-2)

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
            x_val = torch.tensor(np.array(val[:, :-1, :]), dtype=torch.float32).to(self.device) 
            y_val = torch.tensor(np.array(val[:, 1:, :]), dtype=torch.float32).to(self.device) 

            early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, start_from_epoch=start_early_stop_from_epoch)

        if batch_size > train_size:
            batch_size = train_size
        
        x = torch.tensor(np.array(train[:, :-1, :]), dtype=torch.float32)
        y = torch.tensor(np.array(train[:, 1:, :]), dtype=torch.float32)

        dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True) #, num_workers=4)

        losses = []
        self.train() 
        training_start_time = time.time()
        print(f'Dataloader length: {len(dataloader)}')
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
                if verbose > 0 and (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, took {time.time() - training_start_time:.2f}s')
                    training_start_time = time.time()
            else:
                self.eval()  # Set model to evaluation mode
                with torch.no_grad():  # Disable gradient calculation for validation
                    # Make predictions on validation set 
                    val_outputs = self(x_val).detach()
                    val_loss = self.criterion(val_outputs, y_val).item()
                
                    if verbose > 0 and (epoch+1) % 10 == 0:
                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}, , took {time.time() - training_start_time:.2f}s')
                        training_start_time = time.time()
                if early_stopper.early_stop(epoch, val_loss, self.state_dict()):
                    if verbose > 0:
                        print(f'Early stopping at epoch {epoch+1}')
                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}')
                    break
        if early_stopper is not None:
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
                to_add = next_input.cpu().numpy().reshape(next_input.shape[0], 1, next_input.shape[-1])
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
    
class DNN(TorchBaseClass):
    def __init__(self, 
            embed_dim,
            timesteps,
            **kwargs):
        super().__init__(embed_dim, timesteps, **kwargs)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, embed_dim*5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*5, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, self.embed_dim),
        )
        self.model.to(self.device)
    

    def forward(self, x):
        # The model will flatten the input so each timestep is a separate sample we will unflatten it back
        # Note x.shape = (batch_size, timesteps, embed_dim)
        timesteps = x.shape[0]
        x_flat = x.reshape(-1, x.shape[-1])
        return self.model(x_flat).unflatten(0, (timesteps, -1))

# class CNN(TorchBaseClass):
#     def __init__(self, 
#             embed_dim,
#             timesteps,
#             **kwargs):
#         super().__init__(embed_dim, timesteps, **kwargs)
#         self.CNN = torch.nn.Sequential(
#             torch.nn.Conv1d(embed_dim, embed_dim,  kernel_size=7, padding=3),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(embed_dim, embed_dim,  kernel_size=5, padding=2),
#             torch.nn.ReLU(),
#         )
#         self.linear = torch.nn.Sequential(
#             torch.nn.Linear(embed_dim, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, embed_dim),
#         )
#         self.CNN.to(self.device)
#         self.linear.to(self.device)
    

#     def forward(self, x):
#         # The model will flatten the input so each timestep is a separate sample we will unflatten it back
#         # x.shape = (batch_size, timesteps, embed_dim)
#         return self.linear(self.CNN(x.permute(0, 2, 1)).permute(0, 2, 1))
    
# class PositionalEncoding(torch.nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = torch.nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         if d_model%2 != 0:
#             pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
#         else:
#             pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(1)]
#         return self.dropout(x)

# class Transformer(TorchBaseClass):
#     def __init__(self, 
#             embed_dim,
#             timesteps,
#             # Decoder layer hyperparameters
#             model_dim=512, num_heads=16, dim_feedforward=2048, dropout=0.1, activation='relu', 
#             # Decoder Transformer hyperparameters
#             num_layers=6, norm=None, 
#             **kwargs):
        
#         super().__init__(embed_dim, timesteps, **kwargs)
#         self.model_dim = model_dim

#         self.embedding = torch.nn.Linear(embed_dim, model_dim).to(self.device)
#         self.max_timesteps = 5000
#         self.positional_encoder = PositionalEncoding(model_dim, max_len=self.max_timesteps).to(self.device)      
#         # decoder_layer = torch.nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, 
#         #                                            activation=activation, batch_first=True)
#         # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(self.device)
#         # self.fc_out = torch.nn.Linear(model_dim, embed_dim).to(self.device)
#         self.transformer = torch.nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
#                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True).to(self.device)

        
    

#     def forward(self, x):
#         assert x.shape[1] <= self.max_timesteps, f"Sequence length {x.shape[1]} exceeds maximum sequence length {self.max_timesteps}"
        
#         batch_size, seq_len, input_dim = x.size()
#         # Embed input to model_dim
#         x = self.embedding(x)
#         # Scale the embeddings
#         x *= torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
#         # Add positional encoding
#         x = self.positional_encoder(x)
        
#         # Generate mask for the decoder to prevent attending to future positions
#         tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
#         tgt_mask.to(x.device)
#         output = self.transformer(x, x, src_mask=tgt_mask) 
#         # output = self.fc_out(output)
#         return output
    
