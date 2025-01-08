
import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
    
    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)

class Decoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers
    ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)

    
class LSTMVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
        self, input_size, hidden_size, latent_size, 
        kld_weight, num_layers, device=torch.device("cuda")
    ):
        super(LSTMVAE, self).__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.kld_weight = kld_weight

        self.lstm_enc = Encoder(
            input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers
        )
        self.lstm_dec = Decoder(
            input_size=latent_size,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )

        self.fc1 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc2 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0].view(self.num_layers, batch_size, self.hidden_size).to(self.device)

        mean = self.fc1(enc_h[-1])
        logvar = self.fc2(enc_h[-1])
        z = self.reparametrize(mean, logvar)

        h_ = self.fc3(z).unsqueeze(0).repeat(self.num_layers, 1, 1)

        z1 = z.unsqueeze(1).repeat(1, seq_len, 1)
        hidden = (h_.contiguous(), h_.contiguous())
        reconstruct_output, hidden = self.lstm_dec(z1, hidden)

        x_hat = reconstruct_output
        loss = self.loss_function(x_hat, x, mean, logvar)

        return loss, x_hat, z


    def loss_function(self, x_hat, x, mean, log_var):

        reconstruction_loss = F.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mean**2 - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_loss*self.kld_weight
        
        return loss

#Simple LSTM

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, input_size)  # Output should match the input size for reconstruction


    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(F.relu(out))
        out = self.fc(F.relu(out))    
        return out

class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Linear layer instead of embedding as we are dealing with time series data
        self.input_layer = nn.Linear(input_size, d_model)
        
        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                # activation = "relu", #ReLU is the default tho. There is also "gelu"
                batch_first=True 
            ),
            num_layers=num_encoder_layers
        )
        
        # Additional Linear layer and activation function
        self.fc = nn.Linear(d_model, d_model)
        
        # Output layer
        self.decoder = nn.Linear(d_model, input_size)
        
    def forward(self, X):
        # X is of shape (batch, seq_len, input_size)

        # Apply linear layer directly without permuting
        X = self.input_layer(X)  # X now has shape (batch, seq_len, d_model)
        
        # Transformer Encoder forward pass
        X = self.encoder(X)  # Output shape will be (batch, seq_len, d_model)

        # Apply activation function and output layer
        X = self.fc(F.relu(X))
        X = self.decoder(F.relu(X))  # Activation applied to each time step
        
        # Use the last time step of the sequence

        return X


class Ensemble(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, 
        kld_weight, num_layers, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(Ensemble, self).__init__()

        self.lstm_vae = LSTMVAE(input_size = input_size, hidden_size = hidden_size, latent_size = latent_size, 
                                kld_weight = kld_weight, num_layers = num_layers).to(device)
        self.simple_lstm = SimpleLSTM(input_size = input_size, hidden_size=hidden_size, 
                                      num_stacked_layers = num_layers).to(device)
        self.transformer = Transformer(input_size = input_size, d_model = d_model, nhead = nhead, 
                                       num_encoder_layers = num_encoder_layers, dim_feedforward=dim_feedforward).to(device)
        
        # # Replace final layers of each model with Identity to extract features
        #For simple lstm
        self.simple_lstm.fc = nn.Identity()
        self.simple_lstm.fc1 = nn.Identity()

        #For transformer
        self.transformer.fc = nn.Identity()
        self.transformer.decoder = nn.Identity()


        # Fully connected layer after concatenation
        self.fc = nn.Linear(hidden_size+d_model+latent_size, input_size)

    def forward(self, x):

        # Forward pass through each model
        _, x1, z = self.lstm_vae(x.clone())  # Clone to avoid in-place operations

        x2 = self.simple_lstm(x.clone())
        # print("Simple LSTM Shape: ", x2.shape)  
        x2 = x2.view(x2.size(0), -1)
        # print("Simple LSTM Shape after using linear layer: ", x2.shape)

        x3 = self.transformer(x.clone())
        # print("Transformer Shape: ", x3.shape)
        x3 = x3.view(x3.size(0), -1)

        z = z.view(z.size(0), -1)
        # Concatenate the outputs along the feature dimension
        output = torch.cat((x2, x3, z), dim=1)
        # print("Shape after concatenation: ", x.shape)
        
        # Apply the fully connected layer with ReLU activation. Only take last time step
        output = self.fc(output)
        # print("Final shape:", output.shape)

        return output
