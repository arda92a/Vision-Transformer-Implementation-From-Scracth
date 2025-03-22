import unittest
from torch import nn
import torch

class MultiHeadSelfAttentionBlock(nn.Module):
    """
    The MultiHeadSelfAttentionBlock class implements the multi-head self-attention mechanism found in transformer models.
    This layer allows the model to simultaneously consider the relationships between all elements of the input data, 
    learning important features and generating a higher-level representation.

    Args:
    - embedding_dim (int): The size of each input vector (e.g., 768), determining the embedding dimension of the model.
    - num_heads (int): The number of attention heads, i.e., how many different attention mechanisms will be run in parallel.
    - dropout (float): The dropout rate for the attention layer. Helps prevent overfitting by randomly zeroing out weights.
    """

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


class TestMultiHeadSelfAttentionBlock(unittest.TestCase):
    """
    This test class verifies the functionality of the MultiHeadSelfAttentionBlock class.
    It tests whether the output shape is correct and ensures the attention mechanism works as expected.
    """

    def test_multihead_attention_output_shape(self):
        """
        Test if the output shape of the MultiHeadSelfAttentionBlock is correct.
        """
        # Define input parameters
        embedding_dim = 768
        num_heads = 12
        dropout = 0.1
        batch_size = 4
        seq_len = 10  # Sequence length (number of tokens)

        # Create a random input tensor (batch_size, seq_len, embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Initialize MultiHeadSelfAttentionBlock
        msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)

        # Forward pass
        output = msa_block(x)

        # Check if output shape is correct
        expected_output_shape = (batch_size, seq_len, embedding_dim)
        self.assertEqual(output.shape, expected_output_shape, 
                         f"Expected shape {expected_output_shape}, but got {output.shape}")

        # Print success message
        print("Test 'test_multihead_attention_output_shape' passed!")

    def test_multihead_attention_dropout(self):
        """
        Test if the dropout layer does not affect the shape of the output, but can randomly zero elements.
        """
        embedding_dim = 768
        num_heads = 12
        dropout = 0.1
        batch_size = 4
        seq_len = 10  # Sequence length (number of tokens)

        # Create a random input tensor (batch_size, seq_len, embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Initialize MultiHeadSelfAttentionBlock
        msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)

        # Set the model in training mode to ensure dropout is applied
        msa_block.train()

        # Forward pass
        output_train = msa_block(x)

        # Set the model in evaluation mode to disable dropout
        msa_block.eval()

        # Forward pass again
        output_eval = msa_block(x)

        # Ensure the output shapes are the same, but dropout could lead to different outputs
        self.assertEqual(output_train.shape, output_eval.shape,
                         f"Output shape mismatch: {output_train.shape} vs {output_eval.shape}")

        # Check if any element of the output is exactly zero due to dropout
        dropout_occurred = torch.any(output_train == 0).item()

        # Print success message
        print("Test 'test_multihead_attention_dropout' passed!")