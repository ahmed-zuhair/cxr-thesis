"""Model-faithful image explanations without optional XAI dependencies."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


def normalise_saliency(values: torch.Tensor) -> torch.Tensor:
    """Normalise each saliency map independently to the closed interval [0, 1]."""

    if values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("Saliency must have shape [batch, 1, height, width]")
    flattened = values.flatten(1)
    minimum = flattened.amin(dim=1)[:, None, None, None]
    maximum = flattened.amax(dim=1)[:, None, None, None]
    return (values - minimum) / (maximum - minimum).clamp_min(1e-8)


class GradCAM:
    """Compute class-specific Grad-CAM maps for an image/clinical classifier."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target layer must return a feature map")
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def __call__(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        label_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image, clinical)
        if isinstance(label_index, int):
            indices = torch.full(
                (len(image),), label_index, dtype=torch.long, device=logits.device
            )
        else:
            indices = label_index.to(device=logits.device, dtype=torch.long)
        if indices.shape != (len(image),):
            raise ValueError("One target label is required per image")
        selected = logits.gather(1, indices[:, None]).sum()
        selected.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        return normalise_saliency(cam.detach()), logits.detach()

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    clinical: torch.Tensor,
    label_index: int | torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
    internal_batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-aggregated absolute integrated gradients and logits."""

    if steps < 2:
        raise ValueError("Integrated gradients requires at least two steps")
    if internal_batch_size <= 0:
        raise ValueError("internal_batch_size must be positive")
    reference = torch.zeros_like(image) if baseline is None else baseline
    if reference.shape != image.shape:
        raise ValueError("Integrated-gradients baseline shape does not match")
    if isinstance(label_index, int):
        indices = torch.full(
            (len(image),), label_index, dtype=torch.long, device=image.device
        )
    else:
        indices = label_index.to(device=image.device, dtype=torch.long)
    if indices.shape != (len(image),):
        raise ValueError("One target label is required per image")
    gradients = []
    logits = None
    alphas = torch.linspace(0.0, 1.0, steps, device=image.device)
    batch = len(image)
    for start in range(0, steps, internal_batch_size):
        chunk = alphas[start : start + internal_batch_size]
        scaled = (
            reference[None]
            + chunk[:, None, None, None, None] * (image - reference)[None]
        ).reshape(-1, *image.shape[1:]).detach().requires_grad_(True)
        repeated_clinical = clinical.repeat(len(chunk), 1)
        repeated_indices = indices.repeat(len(chunk))
        chunk_logits = model(scaled, repeated_clinical)
        selected = chunk_logits.gather(1, repeated_indices[:, None]).sum()
        gradient = torch.autograd.grad(selected, scaled, retain_graph=False)[0]
        gradients.extend(gradient.detach().reshape(len(chunk), batch, *image.shape[1:]))
        if start + len(chunk) == steps:
            logits = chunk_logits[-batch:]
    stacked = torch.stack(gradients)
    average = (stacked[:-1] + stacked[1:]).mean(dim=0) / 2.0
    attribution = ((image - reference) * average).abs().mean(dim=1, keepdim=True)
    if logits is None:
        raise RuntimeError("Integrated gradients produced no logits")
    return normalise_saliency(attribution), logits.detach()
