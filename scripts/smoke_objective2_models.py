#!/usr/bin/env python3
"""Forward/backward smoke test for all five Objective 2 model families."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_thesis.objective2.data import GraphBatch
from cxr_thesis.objective2.models import build_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--labels", type=int, default=12)
    args = parser.parse_args()
    device = torch.device(args.device)
    results = {}
    for name in ("cnn", "attention_cnn", "vit"):
        model = build_classifier(name, args.labels, image_size=args.image_size).to(device)
        image = torch.rand(2, 1, args.image_size, args.image_size, device=device)
        clinical = torch.rand(2, 9, device=device)
        logits = model(image, clinical)
        logits.mean().backward()
        results[name] = {
            "logit_shape": list(logits.shape),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        del model, image, clinical, logits
    graph_batch = GraphBatch(
        x=torch.rand(12, 7, device=device),
        edge_index=torch.tensor(
            [list(range(12)), list(range(12))], dtype=torch.long, device=device
        ),
        batch_index=torch.tensor([0] * 6 + [1] * 6, device=device),
        clinical=torch.rand(2, 9, device=device),
        labels=torch.zeros(2, args.labels, device=device),
    )
    for name in ("gcn", "gat"):
        model = build_classifier(name, args.labels, node_dim=7).to(device)
        logits = model(graph_batch)
        logits.mean().backward()
        results[name] = {
            "logit_shape": list(logits.shape),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    print("OBJECTIVE 2 FIVE-MODEL SMOKE TEST SUCCESSFUL")


if __name__ == "__main__":
    main()
