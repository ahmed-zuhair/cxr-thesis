"""
gat_model.py
============
Graph Attention Network over CNN features for multi-label CXR classification.

Architecture:
    CXR image (B, 3, 224, 224)
        ↓
    Frozen DenseNet-121 feature extractor → (B, 1024, 7, 7)
        ↓
    Convert to graph: 49 nodes, spatial + feature-kNN edges
        ↓
    GAT layer 1 (8 heads, 128 per head) → (B*49, 1024) output
        ↓
    GAT layer 2 (1 head, 256 output) → (B*49, 256)
        ↓
    Attention-weighted global pooling → (B, 256)
        ↓
    Linear → (B, 14) multi-label logits

Why freeze the backbone?
    - We've already trained a strong DenseNet-121 (val AUC 0.83)
    - Freezing saves ~60% of training compute per epoch
    - Isolates the contribution of graph structure from the contribution of features
      (scientifically cleaner ablation for the thesis)
    - Graph head has only ~2M params vs DenseNet's 7M — fast to train
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from models import DenseNet121MultiLabel
from graph_builder import build_spatial_edges, feature_map_to_graph_batch


class DenseNetFeatureExtractor(nn.Module):
    """
    Extracts the final 7x7 feature map from DenseNet-121.
    Frozen — no gradients, eval mode always.
    """
    def __init__(self, pretrained_ckpt: str = None):
        super().__init__()
        # Load our multi-label wrapper, then extract the backbone's features portion
        full_model = DenseNet121MultiLabel(num_classes=14, pretrained=False)
        if pretrained_ckpt is not None:
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            full_model.load_state_dict(ckpt["model_state"])
            print(f"[backbone] loaded weights from {pretrained_ckpt}")
            print(f"[backbone] original val AUC: {ckpt.get('val_mean_auc', 'unknown')}")

        # DenseNet-121: .features is the conv part, .classifier is the linear head.
        # We need the output BEFORE final avgpool — that's the 7x7 feature map.
        self.features = full_model.backbone.features
        # Freeze
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, 3, 224, 224). Output: (B, 1024, 7, 7)."""
        with torch.no_grad():
            # DenseNet features end with BatchNorm+ReLU — apply them
            feat = self.features(x)
            feat = F.relu(feat, inplace=True)
        return feat

    def train(self, mode: bool = True):
        """Keep backbone in eval mode even if model.train() is called."""
        super().train(mode)
        self.features.eval()
        return self


class GATHead(nn.Module):
    """
    GAT classification head.

    Takes a PyG Batch (49 nodes per image, 1024-dim features).
    Returns (B, num_classes) logits.
    """
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 128,
                 out_channels: int = 256, heads: int = 8,
                 num_classes: int = 14, dropout: float = 0.3):
        super().__init__()
        # Layer 1: multi-head attention, concatenate heads → hidden*heads
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, concat=True)  # out: hidden*heads = 1024
        # Layer 2: single head for final representation
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                            dropout=dropout, concat=False)  # out: 256

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channels, num_classes),
        )

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        # Graph-level pooling (mean over nodes of each image)
        x = global_mean_pool(x, batch_vec)          # (B, out_channels)
        return self.classifier(x)                   # (B, num_classes)


class GATModel(nn.Module):
    """
    Full model: DenseNet feature extractor → graph builder → GAT head.
    """
    def __init__(self, pretrained_ckpt: str, num_classes: int = 14,
                 knn_k: int = 3, use_knn: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = DenseNetFeatureExtractor(pretrained_ckpt=pretrained_ckpt)
        self.head = GATHead(in_channels=1024, num_classes=num_classes, dropout=dropout)
        self.knn_k = knn_k
        self.use_knn = use_knn
        # Precompute spatial edges (same for every image)
        self.register_buffer("spatial_edges", build_spatial_edges(), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, 224, 224) batch of CXR images.
        Output: (B, num_classes) logits.
        """
        feat_map = self.backbone(images)  # (B, 1024, 7, 7), no grad
        # Feature map built with no_grad — clone to allow gradients downstream
        feat_map = feat_map.detach().clone().requires_grad_(False)
        graph_batch = feature_map_to_graph_batch(
            feat_map, self.spatial_edges, k=self.knn_k, use_knn=self.use_knn
        )
        return self.head(graph_batch)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check without checkpoint (random weights)
    model = GATModel(pretrained_ckpt=None, num_classes=14)
    print(f"trainable params (head only): {count_trainable(model):,}")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}  (expect (2, 14))")
    print("✓ GATModel works end-to-end")
