from torch import nn

class MultiHeadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int = 768,
                 num_heads: int = 12,
                 dropout: float =0.0,
                 ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn =nn.MultiheadAttention(embed_dim=embedding_dim,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        batch_first=True)
        
    def forward(self, x):
        
        layer_norm_x = self.layer_norm(x)

        msa_output , _ = self.multihead_attn(query = layer_norm_x,
                                            key = layer_norm_x,
                                            value = layer_norm_x,
                                            need_weights = False)

        return msa_output
