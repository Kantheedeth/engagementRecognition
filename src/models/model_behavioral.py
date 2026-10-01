import torch
import torch.nn as nn

class PureBehavioralAttentionClassifier(nn.Module):
    """
    Pure Behavioral Engagement Recognition Network (Zero Scene Background Shortcut).
    
    Uses either or both behavioral branches:
      • Interaction: 32 pose, orientation, motion, and reliability features
      • Affect:       7 expression probabilities plus reliability

    Each active branch is projected to ``branch_dim``. With both branches, the
    temporal embedding is their concatenation; single-branch ablations retain
    only the selected projection.
    """
    def __init__(
        self, 
        dim_inter=32, 
        dim_affect=8, 
        branch_dim=48, 
        num_heads=4, 
        num_classes=3, 
        dropout=0.15,
        branch_mode="both",
        interaction_indices=None,
    ):
        super(PureBehavioralAttentionClassifier, self).__init__()
        
        self.dim_inter = dim_inter
        self.dim_affect = dim_affect
        if branch_mode not in ("both", "interaction", "affect"):
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        self.branch_mode = branch_mode

        if interaction_indices is None:
            interaction_indices = tuple(range(dim_inter))
        else:
            interaction_indices = tuple(interaction_indices)
        if not interaction_indices:
            raise ValueError("interaction_indices must contain at least one feature index")
        if any(not isinstance(index, int) for index in interaction_indices):
            raise ValueError("interaction_indices must contain only integers")
        if len(interaction_indices) != len(set(interaction_indices)):
            raise ValueError("interaction_indices must not contain duplicates")
        if any(index < 0 or index >= dim_inter for index in interaction_indices):
            raise ValueError(
                f"interaction_indices must be between 0 and {dim_inter - 1}"
            )
        self.interaction_indices = interaction_indices
        self.register_buffer(
            "_interaction_index_tensor",
            torch.tensor(interaction_indices, dtype=torch.long),
            persistent=False,
        )
        
        # Interaction branch: selected interaction features -> branch_dim. The
        # raw affect boundary remains at dim_inter, so ablations do not require
        # rewriting the stored (8, 40) matrices.
        self.branch_inter = nn.Sequential(
            nn.Linear(len(interaction_indices), branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Affect branch: dim_affect -> branch_dim
        self.branch_affect = nn.Sequential(
            nn.Linear(dim_affect, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Concatenate active branch embeddings for temporal modeling.
        fused_dim = branch_dim * (2 if branch_mode == "both" else 1)
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
        # Input shape: (batch, 8, 40)
        # Slices:
        # [0:32]  -> Interaction features
        # [32:40] -> 7 affect probabilities + affect reliability
        
        x_inter = x.index_select(2, self._interaction_index_tensor)
        x_affect = x[:, :, self.dim_inter:self.dim_inter + self.dim_affect]
        
        if self.branch_mode == "interaction":
            fused = self.branch_inter(x_inter)
        elif self.branch_mode == "affect":
            fused = self.branch_affect(x_affect)
        else:
            feat_inter = self.branch_inter(x_inter)
            feat_affect = self.branch_affect(x_affect)
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
    dummy_input = torch.randn(2, 8, 40)
    out = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Pure Behavioral Model output shape: {out.shape}")
    print(f"Total trainable parameters: {num_params:,}")
