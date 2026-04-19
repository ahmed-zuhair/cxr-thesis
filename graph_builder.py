"""
graph_builder.py
================
Converts a 2D CNN feature map into a graph for GNN processing.

Design:
    1. Take DenseNet-121's final feature map: shape (B, 1024, 7, 7)
    2. Flatten spatial dims: 49 "patches" per image, each with 1024-dim features
    3. Build edges:
       - Spatial adjacency (8-connectivity on 7x7 grid) — fixed, same for all images
       - Feature k-NN (k=3 nearest patches by cosine similarity) — image-specific
    4. Return a PyTorch Geometric Batch object ready for GNN layers

Why this design?
    - Spatial edges capture local anatomical structure
    - Feature edges capture non-local relationships (e.g., bilateral lung opacities)
    - 49 nodes keeps the graph small enough for quantum layer experiments later

Output: torch_geometric.data.Batch with:
    x          : (B*49, 1024) node features
    edge_index : (2, E) directed edges
    batch      : (B*49,) batch assignment vector
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


# 7x7 grid, node indices 0..48 in row-major order
GRID_SIZE = 7
NUM_NODES = GRID_SIZE * GRID_SIZE  # 49


def build_spatial_edges(grid_size: int = GRID_SIZE) -> torch.Tensor:
    """
    Build spatial edges for a grid_size x grid_size grid with 8-connectivity.
    Each node connects to its 8 neighbors (up/down/left/right/diagonals).
    Includes self-loops (helps GAT stability).

    Returns:
        edge_index: (2, E) long tensor of edges. Directed both ways.
    """
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            edges.append((node, node))  # self-loop
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor = ni * grid_size + nj
                        edges.append((node, neighbor))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    return edge_index


def build_knn_edges(features: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    For a single image's features (49, 1024), add edges to each node's top-k
    most similar nodes by cosine similarity.

    Args:
        features: (N, D) node features for one image
        k: number of neighbors per node
    Returns:
        edge_index: (2, N*k) long tensor of directed edges
    """
    # Normalize for cosine similarity
    feats_norm = F.normalize(features, p=2, dim=1)              # (N, D)
    sim = feats_norm @ feats_norm.t()                            # (N, N)
    # Exclude self by setting diagonal to -inf
    sim.fill_diagonal_(float("-inf"))
    # Top-k neighbors per row
    _, topk_idx = sim.topk(k, dim=1)                             # (N, k)
    # Build edges: for each source i, add edge (i, j) for each j in topk_idx[i]
    sources = torch.arange(features.size(0), device=features.device).repeat_interleave(k)
    targets = topk_idx.flatten()
    edge_index = torch.stack([sources, targets], dim=0)          # (2, N*k)
    return edge_index


def feature_map_to_graph_batch(
    feature_map: torch.Tensor,
    spatial_edges: torch.Tensor,
    k: int = 3,
    use_knn: bool = True,
) -> Batch:
    """
    Convert a batch of CNN feature maps into a PyG Batch of graphs.

    Args:
        feature_map: (B, C, H, W) e.g. (B, 1024, 7, 7)
        spatial_edges: (2, E_spatial) precomputed, same for every image
        k: k-NN neighbors per node for feature-based edges
        use_knn: whether to add feature-similarity edges in addition to spatial
    Returns:
        PyG Batch with .x (B*N, C), .edge_index (2, total_edges), .batch (B*N,)
    """
    B, C, H, W = feature_map.shape
    N = H * W  # 49

    # Reshape: (B, C, H, W) -> (B, N, C) — each image becomes N nodes of C features
    node_feats = feature_map.view(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)

    data_list = []
    for i in range(B):
        x = node_feats[i]  # (N, C)
        if use_knn:
            knn_edges = build_knn_edges(x, k=k)
            edge_index = torch.cat([spatial_edges.to(x.device), knn_edges], dim=1)
            # Remove duplicate edges (spatial + knn may overlap)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = spatial_edges.to(x.device)
        data_list.append(Data(x=x, edge_index=edge_index))

    return Batch.from_data_list(data_list)


if __name__ == "__main__":
    # Sanity check
    B, C, H, W = 4, 1024, 7, 7
    feat = torch.randn(B, C, H, W)
    spatial = build_spatial_edges()
    print(f"spatial edges shape: {spatial.shape}  ({spatial.size(1)} edges)")
    print(f"(expect: 49 self-loops + 4*2 corners*3 + 4*5*5 edges*5 + 25*8 interior*8)")

    batch = feature_map_to_graph_batch(feat, spatial, k=3, use_knn=True)
    print(f"\nbatched graph:")
    print(f"  x shape          : {batch.x.shape}     (should be {B*49} x {C})")
    print(f"  edge_index shape : {batch.edge_index.shape}")
    print(f"  batch vector     : {batch.batch.shape}  unique: {batch.batch.unique().tolist()}")
    print(f"  num graphs       : {batch.num_graphs}")
    print("\n✓ graph_builder works end-to-end")
