import torch
import torch.nn as nn

class TemporalAttentionClassifier(nn.Module):
    def __init__(self, input_dim=585, hidden_dim=128, num_heads=4, num_classes=3, dropout=0.1):
        super(TemporalAttentionClassifier, self).__init__()
        
        # Linear projection layer to project high-dimensional frame features
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Multi-Head Self-Attention over the 8 time steps (frames)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Classifier head
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Input shape: (batch, seq_len, input_dim) -> e.g., (batch, 8, 585)
        
        # Project to hidden dimension
        out = self.proj(x)
        out = self.norm1(out)
        out = torch.relu(out)
        out = self.dropout1(out)
        
        # Self-attention over the sequence of 8 frames
        # attn_output shape: (batch, seq_len, hidden_dim)
        attn_out, _ = self.attn(out, out, out)
        out = out + attn_out  # residual connection
        out = self.norm2(out)
        out = self.dropout2(out)
        
        # Mean pooling over the 8 frames (timesteps)
        # shape: (batch, hidden_dim)
        pooled = out.mean(dim=1)
        
        # Final classification
        # shape: (batch, num_classes)
        logits = self.fc(pooled)
        return logits

# Example utility to check parameter count
if __name__ == "__main__":
    model = TemporalAttentionClassifier(input_dim=585, hidden_dim=128, num_heads=4, num_classes=3)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Temporal Attention Network created successfully.")
    print(f"Total trainable parameters: {num_params:,}")
