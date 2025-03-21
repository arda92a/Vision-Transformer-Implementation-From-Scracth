from torch import nn
from Patcher import PatchEmbedding
from TransformerEncoder import TransformerEncoder
import torch

class ViT(nn.Module):
    """
    The ViT (Vision Transformer) class implements the Vision Transformer model.
    It uses transformer-based architecture for image classification, where the image is first split into patches,
    and each patch is processed as a sequence of tokens by the transformer.

    Args:
    - img_size (int): Size of the input image (e.g., 224).
    - in_channels (int): Number of input channels (e.g., 3 for RGB images).
    - patch_size (int): Size of each patch to split the image into (e.g., 16).
    - embedding_dim (int): The size of the embedding for each patch (e.g., 768).
    - num_transformer_layers (int): Number of transformer encoder layers (e.g., 12).
    - num_heads (int): Number of attention heads in the multi-head attention block (e.g., 12).
    - attn_dropout (float): Dropout rate for the attention mechanism.
    - mlp_size (int): The size of the hidden layer in the multi-layer perceptron (MLP) of the transformer.
    - mlp_dropout (float): Dropout rate for the MLP in the transformer.
    - num_classes (int): The number of output classes for the classification task (e.g., 1000).
    """

    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768,
                 num_transformer_layers:int=12,
                 num_heads: int = 12,
                 attn_dropout: float =0.0,
                 mlp_size: int=3072,
                 mlp_dropout: float = 0.1,
                 num_classes: int = 1000
                 ):
        super().__init__()

        assert img_size % patch_size == 0, f"Image size ({img_size}) must be compatible with patch size ({patch_size})"

        self.patch_number = (img_size**2)//(patch_size**2)

        self.class_embedding = nn.Parameter(torch.randn((1,1,embedding_dim)),
                                            requires_grad=True)
        
        self.positional_embedding = nn.Parameter(torch.randn(1,self.patch_number+1,embedding_dim),
                                                 requires_grad=True)
        
        self.embedding_dropout = nn.Dropout(mlp_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        self.transformer_encoder = nn.Sequential(*[
                TransformerEncoder(embedding_dim=embedding_dim,
                                   num_heads=num_heads,
                                   attn_dropout=attn_dropout,
                                   mlp_size=mlp_size,
                                   mlp_dropout=mlp_dropout)
                for _ in range(num_transformer_layers)
            ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):

        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size,-1,-1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token,x),dim=1)

        x = self.positional_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:,0])

        return x
        


