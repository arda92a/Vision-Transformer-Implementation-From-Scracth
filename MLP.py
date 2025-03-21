from torch import nn

class MultipLayerPerceptron(nn.Module):
    """
    The MultipLayerPerceptron class implements a simple multi-layer perceptron (MLP) that is used in transformer models
    to apply a series of transformations to the input. It consists of two linear layers with a GELU activation and dropout
    for regularization. The output of the MLP is also normalized using layer normalization.

    Args:
    - embedding_dim (int): The size of the input/output embedding (e.g., 768).
    - mlp_size (int): The size of the hidden layer in the MLP (e.g., 3072).
    - dropout (float): Dropout rate used in the layers to prevent overfitting.
    """

    def __init__(self,
                 embedding_dim : int=768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):

        layer_norm_x = self.layer_norm(x)

        mlp_output = self.mlp(layer_norm_x)

        return mlp_output