import torch
import torch.nn as nn

class PureBehavioralAttentionClassifier(nn.Module):
    """
    Pure Behavioral Engagement Recognition Network (Zero Scene Background Shortcut).
    
    Fuses:
      • Branch 1 (Interaction): 40-dim role-aware track pooling -> 48-dim
      • Branch 2 (Affect):       7 expression probabilities + reliability -> 48-dim
    Total Fused Dimension: 48 + 48 = 96 dimensions per frame.
    """
    def __init__(
        self, 
        dim_inter=40,
        dim_affect=8, 
        branch_dim=48, 
        num_heads=4, 
        num_classes=3, 
        dropout=0.15
    ):
        super(PureBehavioralAttentionClassifier, self).__init__()
        
        self.dim_inter = dim_inter
        self.dim_affect = dim_affect
        
        # Interaction Branch: 40 -> 48
        self.branch_inter = nn.Sequential(
            nn.Linear(dim_inter, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Affect Branch: 8 -> 48
        self.branch_affect = nn.Sequential(
            nn.Linear(dim_affect, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Balanced Fused Embedding: 48 + 48 = 96
        fused_dim = branch_dim * 2
        assert fused_dim % num_heads == 0, f"fused_dim ({fused_dim}) must be divisible by num_heads ({num_heads})"
        
        # Multi-Head Self-Attention across 8 video frames
        self.attn = nn.MultiheadAttention(
            embed_dim=fused_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Intermediate MLP Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch, 8, 48)
        # Slices:
        # [0:40]  -> Track-aware role-aware interaction features
        # [40:48] -> 7 affect probabilities + affect reliability
        
        x_inter = x[:, :, 0:self.dim_inter]
        x_affect = x[:, :, self.dim_inter:self.dim_inter + self.dim_affect]
        
        feat_inter = self.branch_inter(x_inter)    # (batch, 8, 48)
        feat_affect = self.branch_affect(x_affect)  # (batch, 8, 48)
        
        # Concatenate behavioral states: (batch, 8, 96)
        fused = torch.cat([feat_inter, feat_affect], dim=-1)
        
        # Temporal Attention
        attn_out, _ = self.attn(fused, fused, fused)
        out = fused + attn_out  # Residual
        out = self.norm(out)
        out = self.dropout(out)
        
        # Temporal Mean Pooling across 8 frames
        pooled = out.mean(dim=1)  # (batch, 96)
        
        # Logits output
        logits = self.classifier(pooled)  # (batch, 3)
        return logits

if __name__ == "__main__":
    model = PureBehavioralAttentionClassifier()
    dummy_input = torch.randn(2, 8, 48)
    out = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Pure Behavioral Model output shape: {out.shape}")
    print(f"Total trainable parameters: {num_params:,}")
