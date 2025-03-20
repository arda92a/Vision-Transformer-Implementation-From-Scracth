from torch import nn
from MSA import MultiHeadSelfAttentionBlock
from MLP import MultipLayerPerceptron

class TransformerEncoder(nn.Module):

    def __init__(self,
                 embedded_dim: int=768,
                 num_heads: int=12,
                 attn_dropout: float =0.0,
                 mlp_size: int=3072,
                 mlp_dropout: float = 0.1):
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embedded_dim=embedded_dim,
                                                     num_heads=num_heads,
                                                     dropout=attn_dropout)
        
        self.mlp_block = MultipLayerPerceptron(embedded_dim=embedded_dim,
                                               mlp_size=mlp_size,
                                               dropout=mlp_dropout)
        
    def forward(self, x):

        x = self.msa_block(x) + x

        x = self.mlp_block(x) + x

        return x
