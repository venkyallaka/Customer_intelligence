from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class TorchMissingError(RuntimeError):
    pass


def build_ft_transformer(input_dim: int, output_dim: int = 1):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise TorchMissingError(
            "PyTorch is required for the FT-Transformer path. "
            "Install torch after enabling Windows long paths or use --model sklearn."
        ) from exc

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            d_token = 64
            self.feature_tokens = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
            self.value_projection = nn.Linear(1, d_token)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=8,
                dim_feedforward=192,
                dropout=0.12,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, 64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            values = self.value_projection(x.unsqueeze(-1))
            tokens = values + self.feature_tokens.unsqueeze(0)
            cls = self.cls.expand(x.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(encoded[:, 0, :]).squeeze(-1)

    return FTTransformer()
