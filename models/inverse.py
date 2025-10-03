import torch
import torch.nn as nn
import torch.nn.functional as F


class InversecVAE(nn.Module):
    """
    Conditional Variational Autoencoder
    S-parameters [201×4] -> Pattern [48×32]
    """
    
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # ===== ENCODER: S-params -> latent =====
        # Treat [201, 4] as 1D sequence with 4 channels
        self.encoder = nn.Sequential(
            # [4, 201] -> [32, 100]
            nn.Conv1d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # [32, 100] -> [64, 50]
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # [64, 50] -> [128, 25]
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # [128, 25] -> [256, 12]
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.flatten_size = 256 * 13  # ~3328
        
        # Latent parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # ===== DECODER: latent + condition -> pattern =====
        # Conditioning vector (same as encoder output for symmetry)
        self.condition_encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [64, 1]
        )
        self.condition_size = 64
        
        # Decoder input: latent + condition
        decoder_input_size = latent_dim + self.condition_size
        
        # Project to initial spatial size
        self.decoder_input = nn.Sequential(
            nn.Linear(decoder_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 6 * 4),  # [256, 6, 4]
            nn.ReLU()
        )
        
        # Transpose convolutions to upsample
        self.decoder = nn.Sequential(
            # [256, 6, 4] -> [128, 12, 8]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # [128, 12, 8] -> [64, 24, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # [64, 24, 16] -> [32, 48, 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # [32, 48, 32] -> [1, 48, 32]
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def encode(self, S_params):
        """
        Encode S-parameters to latent distribution
        
        Args:
            S_params: [B, 201, 4]
        Returns:
            mu, logvar: [B, latent_dim]
        """
        # Transpose for Conv1d: [B, 4, 201]
        x = S_params.transpose(1, 2)
        x = self.encoder(x)  # [B, 256, 13]
        x = x.view(x.size(0), -1)  # [B, 3328]
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, S_params):
        """
        Decode latent vector to pattern
        
        Args:
            z: [B, latent_dim]
            S_params: [B, 201, 4] conditioning
        Returns:
            logits: [B, 1, 48, 32]
        """
        # Encode condition
        c = S_params.transpose(1, 2)  # [B, 4, 201]
        c = self.condition_encoder(c)  # [B, 64, 1]
        c = c.squeeze(-1)  # [B, 64]
        
        # Concatenate latent and condition
        x = torch.cat([z, c], dim=1)  # [B, latent_dim + 64]
        
        # Decode to pattern
        x = self.decoder_input(x)  # [B, 256*6*4]
        x = x.view(-1, 256, 6, 4)  # [B, 256, 6, 4]
        logits = self.decoder(x)  # [B, 1, 48, 32]
        
        return logits
    
    def forward(self, S_params):
        """
        Full forward pass through cVAE
        
        Args:
            S_params: [B, 201, 4]
        Returns:
            logits: [B, 1, 48, 32]
            mu, logvar: [B, latent_dim]
        """
        mu, logvar = self.encode(S_params)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, S_params)
        return logits, mu, logvar
