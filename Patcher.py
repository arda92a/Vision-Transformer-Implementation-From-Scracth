""""
In this file, the operations from the initial state of the image to the Transformer Encoder layer are implemented. 
The steps of separating the image into patches, flattening the patches and bringing them to the Latent Vector size, and then adding 
Class token and Position Embedding are covered.
"""
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch import nn

def visualize_patcher(img: np.array,
                      patch_size: int):
    """
    This function visualizes how the image is divided into patches. It checks whether the image dimensions are divisible by the patch size, 
    and then divides the image into patches, displaying them in a grid for better understanding of how the patches are created.

    Args:
    - img (np.array): Input image to be visualized.
    - patch_size (int): The size of each patch the image will be divided into.

    """

    height, width , _ = img.shape

    assert height & patch_size == 0 , f"Image Sizes must compatible with Patch size, Heigh: {height} not divisible in Patch Size: {patch_size}"
    assert width & patch_size == 0 , f"Image Sizes must compatible with Patch size, Width: {height} not divisible in Patch Size: {patch_size}"
    print(f"Image Size: [H,W] -> [{height},{width}] is compatible with patch size -> {patch_size}")

    n_rows = width//patch_size
    n_cols = height//patch_size

    fig, axs = plt.subplots(n_cols,n_rows,figsize=(10, 8))

    for w in range(1,n_rows+1,1): # w = 1,2,3 ... n_rows
        for h in range(1,n_cols+1,1): # h = 1,2,3 ... n_cols
            axs[h-1,w-1].imshow(img[(h-1)*patch_size:h*patch_size,(w-1)*patch_size:w*patch_size])

    for ax in axs.flat:
        ax.axis('off')  

    plt.tight_layout(pad=0.2)

class PatchEmbedding(nn.Module):
    """
    PatchEmbedding is a class responsible for converting the image into non-overlapping patches, 
    and embedding them into a latent vector space. The patches are flattened, followed by adding position 
    encoding to represent the spatial relationship of the patches.

    Args:
    - in_channels (int): The number of input channels for the image (typically 3 for RGB images).
    - patch_size (int): The size of each patch that the image will be divided into.
    - embedding_dim (int): The dimension of the embedding space, representing how large the feature vectors will be.
    """

    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 embedding_dim: int):
        
        super().__init__()
        self.patch_size = patch_size

        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embedding_dim,
                                    stride=patch_size,
                                    kernel_size=patch_size,
                                    padding=0)
        
        self.flatten_layer = nn.Flatten(start_dim=2,
                                        end_dim=3)
        
    def forward(self, x):

        assert x.shape[-1] % self.patch_size == 0, f"Image size ({x.shape}) must be compatible with patch size ({self.patch_size})"

        x_patches = self.conv_layer(x)

        x_flattened = self.flatten_layer(x_patches)

        return x_flattened.permute(0,2,1)



