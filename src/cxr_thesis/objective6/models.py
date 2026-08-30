"""Multimodal DenseNet/Transformer model for clinical report generation."""

from __future__ import annotations

import torch
from torch import nn


class DenseNetTransformerReportGenerator(nn.Module):
    """Generate a report from a CXR and non-diagnostic clinical metadata.

    The image is represented by spatial DenseNet-121 feature tokens. A projected
    age/sex/view vector is appended as a distinct clinical token. The decoder is
    autoregressive and never receives ground-truth disease labels.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        clinical_dim: int = 9,
        label_count: int = 6,
        d_model: int = 256,
        heads: int = 8,
        layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        maximum_length: int = 160,
        pretrained: bool = False,
        freeze_image_encoder: bool = True,
        use_clinical: bool = True,
        use_concept_token: bool = False,
    ) -> None:
        super().__init__()
        if vocabulary_size < 5:
            raise ValueError("vocabulary_size must include special and text tokens")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if maximum_length < 2:
            raise ValueError("maximum_length must be at least two")
        try:
            from torchvision.models import DenseNet121_Weights, densenet121
        except ImportError as error:
            raise RuntimeError("torchvision is required for Objective 6") from error

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        self.image_encoder = backbone.features
        self.image_projection = nn.Linear(1024, d_model)
        self.clinical_projection = nn.Sequential(
            nn.Linear(clinical_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.memory_normalisation = nn.LayerNorm(d_model)
        self.token_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(maximum_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output = nn.Linear(d_model, vocabulary_size)
        self.auxiliary_labels = nn.Linear(d_model * 2, label_count)
        self.concept_projection = (
            nn.Sequential(
                nn.Linear(label_count, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if use_concept_token
            else None
        )
        self.maximum_length = int(maximum_length)
        self.vocabulary_size = int(vocabulary_size)
        self.use_clinical = bool(use_clinical)
        self.use_concept_token = bool(use_concept_token)
        self.set_image_encoder_trainable(not freeze_image_encoder)

    def set_image_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = bool(trainable)

    def set_final_image_block_trainable(self) -> None:
        """Train only DenseNet denseblock4/norm5 while preserving frozen BN state."""

        self.set_image_encoder_trainable(False)
        for name in ("denseblock4", "norm5"):
            module = getattr(self.image_encoder, name)
            for parameter in module.parameters():
                parameter.requires_grad = True

    def load_objective5_encoder(self, checkpoint: dict[str, object]) -> None:
        """Load only the DenseNet feature extractor from an Objective 5 checkpoint."""

        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise TypeError("Checkpoint does not contain model_state")
        prefix = "encoder.features."
        selected = {
            str(key)[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        if not selected:
            raise ValueError("Checkpoint contains no DenseNet feature weights")
        result = self.image_encoder.load_state_dict(selected, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Objective 5 image encoder restoration was incomplete")

    def encode(
        self, image: torch.Tensor, clinical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Objective 6 expects three-channel CXR tensors")
        if clinical.ndim != 2 or clinical.shape[0] != image.shape[0]:
            raise ValueError("Clinical tensor must be [batch, clinical features]")
        feature_map = torch.relu(self.image_encoder(image))
        spatial = feature_map.flatten(2).transpose(1, 2)
        image_tokens = self.image_projection(spatial)
        clinical_token = self.clinical_projection(clinical).unsqueeze(1)
        if not self.use_clinical:
            clinical_token = torch.zeros_like(clinical_token)
        pooled_image = image_tokens.mean(dim=1)
        auxiliary_logits = self.auxiliary_labels(
            torch.cat([pooled_image, clinical_token[:, 0]], dim=1)
        )
        memory_tokens = [image_tokens, clinical_token]
        if self.concept_projection is not None:
            concept_token = self.concept_projection(
                auxiliary_logits.sigmoid()
            ).unsqueeze(1)
            memory_tokens.append(concept_token)
        memory = self.memory_normalisation(torch.cat(memory_tokens, dim=1))
        return memory, auxiliary_logits

    def decode_tokens(
        self, input_ids: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        length = int(input_ids.shape[1])
        if length > self.maximum_length:
            raise ValueError("input sequence exceeds maximum_length")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        target = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=input_ids.eq(0),
        )
        return self.output(decoded)

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory, auxiliary_logits = self.encode(image, clinical)
        return {
            "report_logits": self.decode_tokens(input_ids, memory),
            "clinical_logits": auxiliary_logits,
        }

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        *,
        bos_id: int = 1,
        eos_id: int = 2,
        maximum_length: int | None = None,
    ) -> torch.Tensor:
        """Deterministic greedy decoding used for smoke tests and evaluation."""

        limit = min(maximum_length or self.maximum_length, self.maximum_length)
        memory, _ = self.encode(image, clinical)
        generated = torch.full(
            (image.shape[0], 1), bos_id, dtype=torch.long, device=image.device
        )
        finished = torch.zeros(image.shape[0], dtype=torch.bool, device=image.device)
        for _ in range(limit - 1):
            next_token = self.decode_tokens(generated, memory)[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.zeros_like(next_token), next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        return generated

    @staticmethod
    def _blocked_tokens(sequence: list[int], order: int) -> set[int]:
        if order <= 0 or len(sequence) < order - 1:
            return set()
        prefix = tuple(sequence[-(order - 1) :]) if order > 1 else ()
        blocked = set()
        for index in range(len(sequence) - order + 1):
            gram = tuple(sequence[index : index + order])
            if gram[:-1] == prefix:
                blocked.add(gram[-1])
        return blocked

    @torch.no_grad()
    def generate_beam(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        *,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        maximum_length: int | None = None,
        beam_width: int = 3,
        length_normalization_alpha: float = 0.7,
        no_repeat_ngram_size: int = 4,
    ) -> torch.Tensor:
        """Deterministic batched beam search used by locked v1.1 evaluation."""

        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        limit = min(maximum_length or self.maximum_length, self.maximum_length)
        memory, _ = self.encode(image, clinical)
        batch = int(image.shape[0])
        memory = memory.repeat_interleave(beam_width, dim=0)
        sequences = torch.full(
            (batch, beam_width, 1), bos_id, dtype=torch.long, device=image.device
        )
        scores = torch.full(
            (batch, beam_width), float("-inf"), device=image.device
        )
        scores[:, 0] = 0.0
        finished = torch.zeros(
            (batch, beam_width), dtype=torch.bool, device=image.device
        )
        for _ in range(limit - 1):
            flat = sequences.reshape(batch * beam_width, -1)
            log_probabilities = self.decode_tokens(flat, memory)[:, -1].log_softmax(-1)
            log_probabilities = log_probabilities.reshape(
                batch, beam_width, self.vocabulary_size
            )
            for row in range(batch):
                for beam in range(beam_width):
                    if finished[row, beam]:
                        log_probabilities[row, beam].fill_(float("-inf"))
                        log_probabilities[row, beam, pad_id] = 0.0
                        continue
                    blocked = self._blocked_tokens(
                        sequences[row, beam].tolist(), no_repeat_ngram_size
                    )
                    if blocked:
                        log_probabilities[row, beam, list(blocked)] = float("-inf")
            candidates = scores.unsqueeze(-1) + log_probabilities
            next_scores, flat_indices = candidates.flatten(1).topk(
                beam_width, dim=1
            )
            parent = torch.div(
                flat_indices, self.vocabulary_size, rounding_mode="floor"
            )
            token = flat_indices.remainder(self.vocabulary_size)
            gather = parent.unsqueeze(-1).expand(-1, -1, sequences.shape[-1])
            sequences = torch.cat([sequences.gather(1, gather), token.unsqueeze(-1)], dim=-1)
            finished = finished.gather(1, parent) | token.eq(eos_id)
            scores = next_scores
            if bool(finished.all()):
                break
        lengths = sequences.ne(pad_id).sum(dim=-1).clamp_min(1).float()
        normalizer = ((5.0 + lengths) / 6.0).pow(length_normalization_alpha)
        best = (scores / normalizer).argmax(dim=1)
        return sequences[torch.arange(batch, device=image.device), best]
