from torch import nn
from MSA import MultiHeadSelfAttentionBlock
from MLP import MultipLayerPerceptron

class TransformerEncoder(nn.Module):
    """
    The TransformerEncoder class implements a single encoder layer of a transformer model.
    It consists of a multi-head self-attention block followed by a multi-layer perceptron (MLP).
    The residual connection is applied after each block (i.e., the input is added back to the output).

    Args:
    - embedding_dim (int): The size of the input/output embeddings (e.g., 768).
    - num_heads (int): The number of attention heads in the multi-head attention block (e.g., 12).
    - attn_dropout (float): Dropout rate for the multi-head attention block.
    - mlp_size (int): The size of the hidden layer in the multi-layer perceptron (e.g., 3072).
    - mlp_dropout (float): Dropout rate for the multi-layer perceptron block.
    """

    def __init__(self,
                 embedding_dim: int=768,
                 num_heads: int=12,
                 attn_dropout: float =0.0,
                 mlp_size: int=3072,
                 mlp_dropout: float = 0.1):
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     dropout=attn_dropout)
        
        self.mlp_block = MultipLayerPerceptron(embedding_dim=embedding_dim,
                                               mlp_size=mlp_size,
                                               dropout=mlp_dropout)
        
    def forward(self, x):

        x = self.msa_block(x) + x

        x = self.mlp_block(x) + x

        return x
