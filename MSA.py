from torch import nn

class MultiHeadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embedded_dim:int = 768,
                 num_heads: int = 12,
                 dropout: float =0.0,
                 ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedded_dim)

        self.msa =nn.MultiheadAttention(embed_dim=embedded_dim,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        batch_first=True)
        
    def forward(self, x):
        
        layer_norm_x = self.layer_norm(x)

        msa_output , msa_weights = self.msa(query = x,
                                            key = x,
                                            value = x,
                                            need_weights = False)

        return msa_output
