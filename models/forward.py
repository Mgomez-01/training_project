import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    """
    Simple forward model: Pattern [48×32] -> S-parameters [201×4]
    """
    
    def __init__(self):
        super().__init__()
        
        # 2D CNN encoder for pattern
        self.encoder = nn.Sequential(
            # [1, 48, 32] -> [32, 24, 16]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # [32, 24, 16] -> [64, 12, 8]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # [64, 12, 8] -> [128, 6, 4]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # [128, 6, 4] -> [256, 3, 2]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Flatten: 256 * 3 * 2 = 1536
        self.flatten_size = 256 * 3 * 2
        
        # Decoder to S-parameters
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 201 * 4)  # Output: [201, 4]
        )
    
    def forward(self, pattern):
        """
        Args:
            pattern: [B, 1, 48, 32] binary pattern
        Returns:
            S_params: [B, 201, 4] S-parameters
        """
        # Encode pattern
        x = self.encoder(pattern)  # [B, 256, 3, 2]
        x = x.view(x.size(0), -1)  # [B, 1536]
        
        # Decode to S-parameters
        S = self.decoder(x)  # [B, 804]
        S = S.view(-1, 201, 4)  # [B, 201, 4]
        
        return S


class ResBlock2D(nn.Module):
    """2D Residual block with skip connection"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ForwardModelResNet(nn.Module):
    """
    More powerful forward model with residual connections
    Pattern [48×32] -> S-parameters [201×4]
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            # Initial conv: [1, 48, 32] -> [64, 24, 16]
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            # ResBlock + downsample: [64, 24, 16] -> [128, 12, 8]
            nn.Sequential(
                ResBlock2D(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # ResBlock + downsample: [128, 12, 8] -> [256, 6, 4]
            nn.Sequential(
                ResBlock2D(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
        ])
        
        self.flatten_size = 256 * 6 * 4  # After 3 downsamples from 48×32
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 201 * 4)
        )
    
    def forward(self, pattern):
        """
        Args:
            pattern: [B, 1, 48, 32] binary pattern
        Returns:
            S_params: [B, 201, 4] S-parameters
        """
        x = pattern
        for layer in self.encoder:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        S = self.decoder(x)
        S = S.view(-1, 201, 4)
        return S
