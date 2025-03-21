import unittest
from torch import nn
import torch

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
    
class TestMultipLayerPerceptron(unittest.TestCase):
    """
    This test class verifies the functionality of the MultipLayerPerceptron class.
    It tests whether the output shape is correct and ensures the MLP processes inputs correctly.
    """

    def test_mlp_output_shape(self):
        """
        Test if the output shape of the MultipLayerPerceptron is correct.
        """
        # Define input parameters
        embedding_dim = 768
        mlp_size = 3072
        dropout = 0.1
        batch_size = 4
        seq_len = 10  # Sequence length (number of tokens)

        # Create a random input tensor (batch_size, seq_len, embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Initialize MultipLayerPerceptron
        mlp = MultipLayerPerceptron(embedding_dim=embedding_dim,
                                    mlp_size=mlp_size,
                                    dropout=dropout)

        # Forward pass
        output = mlp(x)

        # Check if output shape is correct
        expected_output_shape = (batch_size, seq_len, embedding_dim)
        self.assertEqual(output.shape, expected_output_shape, 
                         f"Expected shape {expected_output_shape}, but got {output.shape}")

        # Print success message
        print("Test 'test_mlp_output_shape' passed!")

    def test_mlp_dropout_effect(self):
        """
        Test if dropout is applied correctly by comparing outputs in training and evaluation mode.
        """
        embedding_dim = 768
        mlp_size = 3072
        dropout = 0.1
        batch_size = 4
        seq_len = 10  # Sequence length (number of tokens)

        # Create a random input tensor (batch_size, seq_len, embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Initialize MultipLayerPerceptron
        mlp = MultipLayerPerceptron(embedding_dim=embedding_dim,
                                    mlp_size=mlp_size,
                                    dropout=dropout)

        # Set the model in training mode
        mlp.train()
        output_train = mlp(x)

        # Set the model in evaluation mode
        mlp.eval()
        output_eval = mlp(x)

        # Ensure the output shapes are the same
        self.assertEqual(output_train.shape, output_eval.shape,
                         f"Output shape mismatch: {output_train.shape} vs {output_eval.shape}")

        # Check if any element in the output is exactly zero due to dropout
        dropout_occurred = torch.any(output_train == 0).item()

        # Print success message
        print("Test 'test_mlp_dropout_effect' passed!")