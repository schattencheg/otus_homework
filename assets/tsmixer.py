import torch
import torch.nn as nn

class TSMixer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, num_classes):
        super(TSMixer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Temporal mixing layers
        self.temporal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len, seq_len),
                nn.GELU(),
                nn.Linear(seq_len, seq_len),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Feature mixing layers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, input_size),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([seq_len, input_size]) for _ in range(num_layers)
        ])
        
        # Global average pooling and classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        for i in range(self.num_layers):
            # Temporal mixing
            temp = x.transpose(1, 2)  # [batch, features, seq_len]
            temp = self.temporal_layers[i](temp)
            temp = temp.transpose(1, 2)  # [batch, seq_len, features]
            x = x + temp
            
            # Layer normalization
            x = self.layer_norms[i](x)
            
            # Feature mixing
            feat = self.feature_layers[i](x)
            x = x + feat
            
            # Layer normalization
            x = self.layer_norms[i](x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.gap(x).squeeze(-1)  # [batch, features]
        
        # Classification
        x = self.classifier(x)
        
        return x
