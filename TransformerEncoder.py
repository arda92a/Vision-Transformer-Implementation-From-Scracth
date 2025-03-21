import unittest
from torch import nn
import torch
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

class TestTransformerEncoder(unittest.TestCase):
    """
    This test class verifies the functionality of the TransformerEncoder class.
    It checks if the output shape is correct and if residual connections work properly.
    """

    def test_transformer_encoder_output_shape(self):
        """
        Test if the output shape of the TransformerEncoder is correct.
        """
        # Define input parameters
        embedding_dim = 768
        num_heads = 12
        attn_dropout = 0.1
        mlp_size = 3072
        mlp_dropout = 0.1
        batch_size = 4
        seq_len = 10  # Number of tokens in the sequence

        # Create a random input tensor (batch_size, seq_len, embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Initialize TransformerEncoder
        encoder = TransformerEncoder(embedding_dim=embedding_dim,
                                     num_heads=num_heads,
                                     attn_dropout=attn_dropout,
                                     mlp_size=mlp_size,
                                     mlp_dropout=mlp_dropout)

        # Forward pass
        output = encoder(x)

        # Check if output shape is correct
        expected_output_shape = (batch_size, seq_len, embedding_dim)
        self.assertEqual(output.shape, expected_output_shape, 
                         f"Expected shape {expected_output_shape}, but got {output.shape}")

        # Print success message
        print("Test 'test_transformer_encoder_output_shape' passed!")

    def test_transformer_encoder_residual_connection(self):
        """
        Test if the residual connection is properly applied.
        """
        embedding_dim = 768
        num_heads = 12
        attn_dropout = 0.1
        mlp_size = 3072
        mlp_dropout = 0.1
        batch_size = 4
        seq_len = 10  # Number of tokens in the sequence

        # Create a random input tensor
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Initialize TransformerEncoder
        encoder = TransformerEncoder(embedding_dim=embedding_dim,
                                     num_heads=num_heads,
                                     attn_dropout=attn_dropout,
                                     mlp_size=mlp_size,
                                     mlp_dropout=mlp_dropout)

        # Forward pass
        output = encoder(x)

        # Check if residual connection was applied by ensuring output is not identical to x
        self.assertFalse(torch.equal(x, output), "Residual connection might not be applied correctly.")

        # Print success message
        print("Test 'test_transformer_encoder_residual_connection' passed!")
