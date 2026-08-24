import torch
import torch.nn as nn

class MultiBranchTemporalAttentionClassifier(nn.Module):
    """
    Multi-Branch Balanced Feature Fusion Network.
    
    Independently projects Scene (576-dim), Interaction (32-dim), and Affect (8-dim)
    into equal-sized embedding branches (32-dim each) to prevent the high-dimensional 
    scene features from overwhelming behavioral signals.
    """
    def __init__(
        self, 
        dim_scene=576, 
        dim_inter=32, 
        dim_affect=8, 
        branch_dim=32, 
        num_heads=4, 
        num_classes=3, 
        dropout=0.1
    ):
        super(MultiBranchTemporalAttentionClassifier, self).__init__()
        
        self.dim_scene = dim_scene
        self.dim_inter = dim_inter
        self.dim_affect = dim_affect
        
        # Branch A: Scene Projection (576 -> 32)
        self.branch_scene = nn.Sequential(
            nn.Linear(dim_scene, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Branch B: Interaction Projection (32 -> 32)
        self.branch_inter = nn.Sequential(
            nn.Linear(dim_inter, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Branch C: Affect Projection (8 -> 32)
        self.branch_affect = nn.Sequential(
            nn.Linear(dim_affect, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Balanced fused dimension: 32 + 32 + 32 = 96
        fused_dim = branch_dim * 3
        
        # Multi-Head Self-Attention over the sequence of 8 frames
        self.attn = nn.MultiheadAttention(
            embed_dim=fused_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Final classification head
        self.fc = nn.Linear(fused_dim, num_classes)
        
    def forward(self, x):
        # Input shape: (batch, 8, 616)
        # Feature slices:
        # [0:576]   -> MobileNetV3 Scene
        # [576:608] -> YOLOv8 32-dim Interaction
        # [608:616] -> HSEmotion 8-dim Affect
        
        s_end = self.dim_scene
        i_end = s_end + self.dim_inter
        a_end = i_end + self.dim_affect
        
        x_scene = x[:, :, 0:s_end]
        x_inter = x[:, :, s_end:i_end]
        x_affect = x[:, :, i_end:a_end]
        
        feat_scene = self.branch_scene(x_scene)    # (batch, 8, 32)
        feat_inter = self.branch_inter(x_inter)    # (batch, 8, 32)
        feat_affect = self.branch_affect(x_affect)  # (batch, 8, 32)
        
        # Balanced Concatenation (Exactly 33.3% representation per modality)
        fused = torch.cat([feat_scene, feat_inter, feat_affect], dim=-1)  # (batch, 8, 96)
        
        # Temporal Multi-Head Self-Attention
        attn_out, _ = self.attn(fused, fused, fused)
        out = fused + attn_out  # Residual connection
        out = self.norm(out)
        out = self.dropout(out)
        
        # Mean pooling over 8 frames
        pooled = out.mean(dim=1)  # (batch, 96)
        
        # Logits output
        logits = self.fc(pooled)  # (batch, 3)
        return logits

# Alias for compatibility
TemporalAttentionClassifier = MultiBranchTemporalAttentionClassifier

if __name__ == "__main__":
    model = MultiBranchTemporalAttentionClassifier()
    dummy_input = torch.randn(2, 8, 616)
    out = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Multi-Branch Attention Model output shape: {out.shape}")
    print(f"Total trainable parameters: {num_params:,}")
