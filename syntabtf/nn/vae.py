import torch.nn as nn
import torch
import numpy as np

class Encoder(nn.Module):
    
    def __init__(self, data_dim, hiddens, emb_dim):
        super(Encoder, self).__init__()
        
        seq = []
        dim = data_dim
        for hid_dim in hiddens:
            seq += [nn.Linear(dim, hid_dim),
                    nn.ReLU()]
            dim = hid_dim
            
        self.seq = nn.Sequential(*seq)
        self.fc_mu = nn.Linear(dim, emb_dim)
        self.fc_logvar = nn.Linear(dim, emb_dim)
        
    def forward(self, x):
        
        x = self.seq(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        
        return mu, std, logvar
        

class Decoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, data_dim):
        super(Decoder, self).__init__()
        
        seq = []
        dim = emb_dim
        
        for hid_dim in hidden_dim:
            seq += [nn.Linear(dim, hid_dim),
                    nn.ReLU()]
            dim = hid_dim
        
        seq += [nn.Linear(dim, data_dim)]
        self.seq = nn.Sequential(*seq)
        self.sigmas = nn.Parameter(torch.ones(data_dim) * 0.1)
        
    def forward(self, x):
        return self.seq(x), self.sigmas
        

class TVAE(nn.Module):
    
    def __init__(self, data_dim, encoder_hiddens, decoder_hiddens, emb_dim, encoder=None, decoder=None):
        super(TVAE, self).__init__()
        
        self.data_dim = data_dim
        self.emb_dim = emb_dim
        self.encoder_hiddens = encoder_hiddens
        self.decoder_hiddens = decoder_hiddens
        
        self.encoder = encoder if encoder is not None else Encoder(data_dim, encoder_hiddens, emb_dim)
        self.decoder = decoder if decoder is not None else Decoder(emb_dim, decoder_hiddens, data_dim)
        
    
    def forward(self, x):
        
        # fit to encoder
        mu, std, logvar = self.encoder(x)
        
        # sampling
        eps = torch.randn_like(std)
        embeddings = mu + (std * eps)
        
        # fit to decoder
        x_rec, sigmas = self.decoder(embeddings)
        
        return x_rec, mu, logvar, sigmas
    
    
    def sample(self, n_samples, batch_size):
        """
        Generate synthetic data
        """
        steps = n_samples // batch_size + 1
        
        data = []
        for _ in range(steps):
            mean = torch.zeros(batch_size, self.emb_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std)
            
            gen_data, sigmas = self.decoder(noise)
            
            data.append(gen_data.detach().cpu().numpy())
            
        data = np.concatenate(data, axis=0)
        data = data[:n_samples]
        
        return data, sigmas.detach().cpu().numpy()
        