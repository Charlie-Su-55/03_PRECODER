"""Parallel UE-FFN-only variant; original model and entry points are unchanged.

FDMA-AS-Lite1024 changes only the two UE Transformer FFNs to 1024 hidden
units. All forward, feedback-channel and BS processing is inherited.
"""

from copy import deepcopy

from torch import nn

from .adaptive_hybrid import AdaptiveHybridPrecoder


def _replace_backbone_ffn(backbone: nn.TransformerEncoder, hidden_dim: int) -> None:
    """Initialize one FFN template per branch, then independently clone it.

    TransformerEncoder clones a single initialized layer template. Preserve
    that organization for the replacement FFNs: identical initial values
    across layers within a branch, but no shared parameter objects/storage.
    Attention, norms, activation, dropout, and layer ordering are untouched.
    """
    if all(layer.linear1.out_features == hidden_dim for layer in backbone.layers):
        return

    first = backbone.layers[0]
    linear1 = nn.Linear(
        first.linear1.in_features,
        hidden_dim,
        bias=first.linear1.bias is not None,
        device=first.linear1.weight.device,
        dtype=first.linear1.weight.dtype,
    )
    linear2 = nn.Linear(
        hidden_dim,
        first.linear2.out_features,
        bias=first.linear2.bias is not None,
        device=first.linear2.weight.device,
        dtype=first.linear2.weight.dtype,
    )
    for layer in backbone.layers:
        for name, template in (("linear1", linear1), ("linear2", linear2)):
            previous = getattr(layer, name)
            replacement = deepcopy(template)
            replacement.train(previous.training)
            replacement.weight.requires_grad_(previous.weight.requires_grad)
            if replacement.bias is not None:
                replacement.bias.requires_grad_(previous.bias.requires_grad)
            setattr(layer, name, replacement)


class AdaptiveHybridPrecoderLite(AdaptiveHybridPrecoder):
    """AdaptiveHybridPrecoder with narrower UE FFNs and an unchanged BS.

    Args:
        cfg: The same configuration object accepted by the original model.
        ue_ffn_dim: Positive FFN hidden width for both UE branches (1024 by
            default). 2048 is a no-replacement compatibility mode for the
            original architecture.

    Construct before creating an optimizer. No parameters are created or
    replaced during forward. Narrow FFNs are initialized afresh, never
    sliced from 2048-wide weights. All state_dict keys are inherited.
    """

    def __init__(self, cfg, ue_ffn_dim: int = 1024):
        if isinstance(ue_ffn_dim, bool) or not isinstance(ue_ffn_dim, int):
            raise TypeError("ue_ffn_dim must be a positive integer.")
        if ue_ffn_dim <= 0:
            raise ValueError("ue_ffn_dim must be a positive integer.")

        # Complete the original initialization first: a matching seed leaves
        # every non-FFN parameter/buffer identical to the original model.
        super().__init__(cfg)
        self.ue_ffn_dim = ue_ffn_dim
        _replace_backbone_ffn(self.encoder.fdma_backbone, ue_ffn_dim)
        _replace_backbone_ffn(self.encoder.aircomp_backbone, ue_ffn_dim)

    @property
    def architecture(self) -> str:
        return f"ueffn{self.ue_ffn_dim}"

    @property
    def experiment_name(self) -> str:
        return f"FDMA-AS-Lite{self.ue_ffn_dim}"

    @property
    def architecture_metadata(self) -> dict:
        """JSON-serializable construction metadata, not a state_dict entry.

        A future Lite-specific training/loading entry point must explicitly
        save/use this metadata together with its complete configuration.
        Existing checkpoint formats and configuration fingerprints are not
        changed by this standalone model.
        """
        return {
            "architecture": self.architecture,
            "ue_ffn_dim": self.ue_ffn_dim,
            "ue_layers": {
                "fdma": len(self.encoder.fdma_backbone.layers),
                "as": len(self.encoder.aircomp_backbone.layers),
            },
            "D_MODEL": self.D,
        }
