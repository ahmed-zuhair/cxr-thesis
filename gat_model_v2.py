"""
gat_model_v2.py
===============
Improved GAT model based on ablation of v1.

Changes from v1:
    - 3 GAT layers with residual connections (vs 2 layers no residual)
    - Learnable attention pooling (vs global_mean_pool)
    - Lower dropout 0.1 (vs 0.3)
    - Deeper final classifier

Why these changes:
    - v1 underperformed the CNN baseline by ~4% AUC (0.83 → 0.79)
    - Hypothesis: mean pooling was too blunt; some patches contain signal,
      others are background. Attention pooling learns this weighting.
    - Residual connections preserve the DenseNet's information as message
      passing proceeds (prevents info loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax

from models import DenseNet121MultiLabel
from graph_builder import build_spatial_edges, feature_map_to_graph_batch


class DenseNetFeatureExtractor(nn.Module):
    """Frozen DenseNet-121 feature extractor — same as v1."""
    def __init__(self, pretrained_ckpt: str = None):
        super().__init__()
        full_model = DenseNet121MultiLabel(num_classes=14, pretrained=False)
        if pretrained_ckpt is not None:
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            full_model.load_state_dict(ckpt["model_state"])
            print(f"[backbone] loaded weights from {pretrained_ckpt}")
            print(f"[backbone] original val AUC: {ckpt.get('val_mean_auc', 'unknown')}")
        self.features = full_model.backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
            feat = F.relu(feat, inplace=True)
        return feat

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self


class AttentionPool(nn.Module):
    """
    Learnable attention pooling over graph nodes.
    For each graph in the batch, computes attention scores over its nodes
    and returns an attention-weighted sum.

    Also exposes the attention weights for explainability (Week 5).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                return_weights: bool = False):
        """
        Args:
            x     : (total_nodes, in_channels) node features
            batch : (total_nodes,) graph assignment
        Returns:
            pooled: (batch_size, in_channels) graph-level features
            weights (optional): (total_nodes,) attention weights for viz
        """
        scores = self.attn(x).squeeze(-1)          # (total_nodes,)
        weights = pyg_softmax(scores, batch)       # softmax per-graph
        weighted = x * weights.unsqueeze(-1)       # (total_nodes, in_channels)
        # Sum per graph
        num_graphs = int(batch.max().item()) + 1
        pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        pooled.index_add_(0, batch, weighted)
        if return_weights:
            return pooled, weights
        return pooled


class GATHeadV2(nn.Module):
    """
    3-layer GAT with residual connections + attention pooling.
    """
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256,
                 heads: int = 4, num_classes: int = 14, dropout: float = 0.1):
        super().__init__()
        # Project input 1024 → hidden_channels*heads for residual compatibility
        proj_dim = hidden_channels * heads  # e.g. 256*4 = 1024
        self.input_proj = nn.Linear(in_channels, proj_dim)

        # 3 GAT layers, all output proj_dim (enables residuals)
        self.gat1 = GATConv(proj_dim, hidden_channels, heads=heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(proj_dim, hidden_channels, heads=heads,
                            dropout=dropout, concat=True)
        self.gat3 = GATConv(proj_dim, hidden_channels, heads=heads,
                            dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.norm3 = nn.LayerNorm(proj_dim)

        # Attention-weighted pooling
        self.pool = AttentionPool(proj_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_classes),
        )

        self.dropout = dropout

    def forward(self, batch, return_attention: bool = False):
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch

        # Project to compatible dim
        x = self.input_proj(x)

        # GAT block 1 with residual
        h = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x + F.dropout(h, p=self.dropout, training=self.training))

        # GAT block 2 with residual
        h = F.elu(self.gat2(x, edge_index))
        x = self.norm2(x + F.dropout(h, p=self.dropout, training=self.training))

        # GAT block 3 with residual
        h = F.elu(self.gat3(x, edge_index))
        x = self.norm3(x + F.dropout(h, p=self.dropout, training=self.training))

        # Attention-weighted pooling
        if return_attention:
            pooled, attn_weights = self.pool(x, batch_vec, return_weights=True)
        else:
            pooled = self.pool(x, batch_vec)

        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights
        return logits


class GATModelV2(nn.Module):
    """Full pipeline: frozen DenseNet + graph + GAT v2 head."""
    def __init__(self, pretrained_ckpt: str, num_classes: int = 14,
                 knn_k: int = 3, use_knn: bool = True, dropout: float = 0.1):
        super().__init__()
        self.backbone = DenseNetFeatureExtractor(pretrained_ckpt=pretrained_ckpt)
        self.head = GATHeadV2(in_channels=1024, num_classes=num_classes,
                              dropout=dropout)
        self.knn_k = knn_k
        self.use_knn = use_knn
        self.register_buffer("spatial_edges", build_spatial_edges(), persistent=False)

    def forward(self, images: torch.Tensor, return_attention: bool = False):
        feat_map = self.backbone(images)
        feat_map = feat_map.detach().clone().requires_grad_(False)
        graph_batch = feature_map_to_graph_batch(
            feat_map, self.spatial_edges, k=self.knn_k, use_knn=self.use_knn
        )
        return self.head(graph_batch, return_attention=return_attention)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GATModelV2(pretrained_ckpt=None, num_classes=14)
    print(f"trainable params: {count_trainable(model):,}")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"logits shape: {tuple(y.shape)}")
    y2, attn = model(x, return_attention=True)
    print(f"logits w/ attn: {tuple(y2.shape)}, attention weights: {tuple(attn.shape)}")
    print("✓ GATModelV2 works end-to-end")
