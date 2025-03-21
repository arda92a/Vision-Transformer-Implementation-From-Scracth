import unittest
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
        
class TestViT(unittest.TestCase):
    """
    This test class verifies the functionality of the ViT (Vision Transformer) model.
    It performs various checks including output shape validation, embedding correctness,
    transformer encoding behavior, and model training capability.
    """

    def setUp(self):
        """Initialize the ViT model and input data for testing."""
        self.batch_size = 4
        self.img_size = 224
        self.in_channels = 3
        self.patch_size = 16
        self.embedding_dim = 768
        self.num_transformer_layers = 12
        self.num_heads = 12
        self.attn_dropout = 0.1
        self.mlp_size = 3072
        self.mlp_dropout = 0.1
        self.num_classes = 1000

        self.model = ViT(img_size=self.img_size,
                         in_channels=self.in_channels,
                         patch_size=self.patch_size,
                         embedding_dim=self.embedding_dim,
                         num_transformer_layers=self.num_transformer_layers,
                         num_heads=self.num_heads,
                         attn_dropout=self.attn_dropout,
                         mlp_size=self.mlp_size,
                         mlp_dropout=self.mlp_dropout,
                         num_classes=self.num_classes)

        self.input_tensor = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)

    def test_output_shape(self):
        """Test if the output shape of the ViT model is correct."""
        output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, self.num_classes)

        self.assertEqual(output.shape, expected_shape, 
                         f"Expected shape {expected_shape}, but got {output.shape}")
        print("Test 'test_output_shape' passed!")

    def test_class_embedding_addition(self):
        """Ensure the class token is correctly added to the sequence."""
        patch_embeddings = self.model.patch_embedding(self.input_tensor)
        expected_seq_length = (self.img_size // self.patch_size) ** 2 + 1  # Patches + class token

        self.assertEqual(patch_embeddings.shape[1] + 1, expected_seq_length, 
                         "Class token is not properly added to the sequence.")
        print("Test 'test_class_embedding_addition' passed!")

    def test_patch_embedding_output_shape(self):
        """Check if the patch embedding output shape matches expected dimensions."""
        patch_embeddings = self.model.patch_embedding(self.input_tensor)
        expected_patch_shape = (self.batch_size, (self.img_size // self.patch_size) ** 2, self.embedding_dim)

        self.assertEqual(patch_embeddings.shape, expected_patch_shape, 
                         f"Expected patch embedding shape {expected_patch_shape}, but got {patch_embeddings.shape}")
        print("Test 'test_patch_embedding_output_shape' passed!")

    def test_transformer_encoder_output_consistency(self):
        """Ensure transformer encoder produces valid output of expected shape."""
        patch_embeddings = self.model.patch_embedding(self.input_tensor)
        class_token = self.model.class_embedding.expand(self.batch_size, -1, -1)
        transformer_input = torch.cat((class_token, patch_embeddings), dim=1)
        transformer_output = self.model.transformer_encoder(transformer_input)

        expected_transformer_shape = (self.batch_size, transformer_input.shape[1], self.embedding_dim)

        self.assertEqual(transformer_output.shape, expected_transformer_shape, 
                         f"Expected transformer encoder output shape {expected_transformer_shape}, but got {transformer_output.shape}")
        print("Test 'test_transformer_encoder_output_consistency' passed!")

    def test_training_step(self):
        """Verify if the model updates parameters during training."""
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        x = self.input_tensor
        y = torch.randint(0, self.num_classes, (self.batch_size,))

        output_before = model(x).detach().clone()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        output_after = model(x).detach().clone()

        self.assertFalse(torch.equal(output_before, output_after),
                         "Model parameters did not update after training step.")
        print("Test 'test_training_step' passed!")

