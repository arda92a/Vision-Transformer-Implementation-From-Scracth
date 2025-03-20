from torch import nn

class MultipLayerPerceptron(nn.Module):

    def __init__(self,
                 embedded_dim : int=768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedded_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedded_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedded_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):

        layer_norm_x = self.layer_norm(x)

        mlp_output = self.mlp(layer_norm_x)

        return mlp_output