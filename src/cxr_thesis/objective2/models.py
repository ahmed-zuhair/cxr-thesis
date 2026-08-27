"""Dependency-light CNN, attention, transformer, GCN, and GAT classifiers."""

from __future__ import annotations

import math

import torch
from torch import nn

from .data import GraphBatch


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int = 9, output_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ConvBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        average = values.mean(dim=(2, 3))
        maximum = values.amax(dim=(2, 3))
        channel_weights = torch.sigmoid(self.channel(average) + self.channel(maximum))
        values = values * channel_weights[:, :, None, None]
        spatial_input = torch.cat(
            [values.mean(dim=1, keepdim=True), values.amax(dim=1, keepdim=True)], dim=1
        )
        return values * torch.sigmoid(self.spatial(spatial_input))


class ImageCNNClassifier(nn.Module):
    def __init__(
        self, labels: int, *, attention: bool = False, clinical_dim: int = 9
    ) -> None:
        super().__init__()
        channels = [32, 64, 128, 256]
        stages: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            stages.append(ConvBlock(input_channels, output_channels, stride=2))
            if attention:
                stages.append(CBAM(output_channels))
            input_channels = output_channels
        self.encoder = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clinical = ClinicalEncoder(clinical_dim, 32)
        self.classifier = nn.Linear(channels[-1] + 32, labels)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        image_embedding = self.pool(self.encoder(image)).flatten(1)
        return self.classifier(
            torch.cat([image_embedding, self.clinical(clinical)], dim=1)
        )


class PretrainedDenseNet121Classifier(nn.Module):
    """DenseNet-121 image encoder with a small clinical-feature fusion head.

    ``torchvision`` is imported lazily so the dependency-light frozen Objective 2
    baselines remain usable in environments that do not need the enhanced model.
    Pretrained weights are requested only during training. Checkpoint restoration
    constructs the same architecture without downloading weights before loading
    the saved state dictionary.
    """

    def __init__(
        self,
        labels: int,
        *,
        clinical_dim: int = 9,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        try:
            from torchvision.models import DenseNet121_Weights, densenet121
        except ImportError as error:
            raise RuntimeError(
                "torchvision is required for the enhanced DenseNet-121 model"
            ) from error
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        embedding_dim = int(backbone.classifier.in_features)
        backbone.classifier = nn.Identity()
        self.encoder = backbone
        self.clinical = ClinicalEncoder(clinical_dim, 32)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim + 32, labels),
        )

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("DenseNet-121 expects three-channel CXR tensors")
        image_embedding = self.encoder(image)
        return self.classifier(
            torch.cat([image_embedding, self.clinical(clinical)], dim=1)
        )


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        labels: int,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        embedding_dim: int = 192,
        heads: int = 3,
        layers: int = 4,
        clinical_dim: int = 9,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(1, embedding_dim, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(torch.zeros(1, patches + 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.normalisation = nn.LayerNorm(embedding_dim)
        self.clinical = ClinicalEncoder(clinical_dim, 32)
        self.classifier = nn.Linear(embedding_dim + 32, labels)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(image).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1) + self.position
        image_embedding = self.normalisation(self.encoder(tokens)[:, 0])
        return self.classifier(
            torch.cat([image_embedding, self.clinical(clinical)], dim=1)
        )


def _global_mean_pool(
    values: torch.Tensor, batch_index: torch.Tensor, graphs: int
) -> torch.Tensor:
    output = values.new_zeros((graphs, values.shape[1]))
    output.index_add_(0, batch_index, values)
    counts = values.new_zeros((graphs, 1))
    counts.index_add_(0, batch_index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbour_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, target, x[source])
        degree = x.new_zeros((x.shape[0], 1))
        degree.index_add_(0, target, x.new_ones((source.shape[0], 1)))
        aggregated = aggregated / degree.clamp_min(1.0)
        return torch.relu(self.self_linear(x) + self.neighbour_linear(aggregated))


class GATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        if output_dim % heads:
            raise ValueError("GAT output_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.output = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        nodes = x.shape[0]
        query = self.query(x).view(nodes, self.heads, self.head_dim)
        key = self.key(x).view(nodes, self.heads, self.head_dim)
        value = self.value(x).view(nodes, self.heads, self.head_dim)
        scores = (query[target] * key[source]).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.2)
        target_heads = target[:, None].expand(-1, self.heads)
        maximum = scores.new_full((nodes, self.heads), -torch.inf)
        maximum.scatter_reduce_(
            0, target_heads, scores, reduce="amax", include_self=True
        )
        exponent = torch.exp(scores - maximum[target])
        denominator = scores.new_zeros((nodes, self.heads))
        denominator.scatter_add_(0, target_heads, exponent)
        attention = exponent / denominator[target].clamp_min(1e-8)
        # CUDA autocast may keep the scatter-based softmax in float32 while
        # producing float16 value projections. index_add_ requires the source
        # and destination to have the same dtype, so cast only the final
        # normalised attention coefficients back to the value dtype.
        attention = attention.to(dtype=value.dtype)
        messages = value[source] * attention[:, :, None]
        aggregated = messages.new_zeros((nodes, self.heads, self.head_dim))
        aggregated.index_add_(0, target, messages)
        return torch.relu(self.output(aggregated.reshape(nodes, -1)))


class GraphClinicalClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        labels: int,
        *,
        architecture: str,
        hidden_dim: int = 128,
        clinical_dim: int = 9,
    ) -> None:
        super().__init__()
        if architecture not in {"gcn", "gat"}:
            raise ValueError("Graph architecture must be gcn or gat")
        layer_class = GCNLayer if architecture == "gcn" else GATLayer
        self.layers = nn.ModuleList(
            [
                layer_class(node_dim, hidden_dim),
                layer_class(hidden_dim, hidden_dim),
                layer_class(hidden_dim, hidden_dim),
            ]
        )
        self.clinical = ClinicalEncoder(clinical_dim, 32)
        self.classifier = nn.Linear(hidden_dim + 32, labels)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        graph_embedding = _global_mean_pool(
            x, batch.batch_index, int(batch.clinical.shape[0])
        )
        return self.classifier(
            torch.cat([graph_embedding, self.clinical(batch.clinical)], dim=1)
        )


def build_classifier(
    name: str,
    labels: int,
    *,
    image_size: int = 224,
    node_dim: int = 7,
    clinical_dim: int = 9,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """Construct one of the five frozen Objective 2 model families."""
    normalised = name.strip().lower().replace("-", "_")
    if normalised == "cnn":
        return ImageCNNClassifier(labels, attention=False, clinical_dim=clinical_dim)
    if normalised == "attention_cnn":
        return ImageCNNClassifier(labels, attention=True, clinical_dim=clinical_dim)
    if normalised in {"densenet121", "enhanced_cnn", "pretrained_densenet121"}:
        return PretrainedDenseNet121Classifier(
            labels,
            clinical_dim=clinical_dim,
            pretrained=pretrained,
            dropout=dropout,
        )
    if normalised in {"vit", "vision_transformer", "transformer"}:
        return VisionTransformerClassifier(
            labels, image_size=image_size, clinical_dim=clinical_dim
        )
    if normalised in {"gcn", "gnn"}:
        return GraphClinicalClassifier(
            node_dim, labels, architecture="gcn", clinical_dim=clinical_dim
        )
    if normalised == "gat":
        return GraphClinicalClassifier(
            node_dim, labels, architecture="gat", clinical_dim=clinical_dim
        )
    raise ValueError(f"Unknown Objective 2 model: {name}")
