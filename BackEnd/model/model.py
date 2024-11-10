from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn

class BaseModel:
    
    def GetLocalOutlierFactor(params):
        model_lof = LocalOutlierFactor(
            **params
        )
        
        return model_lof
    
    def GetVQVAE(params):
        model_vqvae = VQVAE(
            params
        )
        return model_vqvae

class VQVAE(nn.Module):
    def __init__(self, params):
        super(VQVAE, self).__init__()
        
        # Unpack parameters from the params dictionary
        input_dim = params.get('input_dim')
        latent_dim = params.get('latent_dim')
        num_embeddings = params.get('num_embeddings')

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        self.quantizer = nn.Embedding(num_embeddings, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        z_e = self.encoder(x)
        z_q = self.quantizer(torch.argmax(z_e, dim=-1))  # Vector Quantization
        return z_q, z_e

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_q, z_e = self.encode(x)
        recon_x = self.decode(z_q)
        return recon_x, z_e
    