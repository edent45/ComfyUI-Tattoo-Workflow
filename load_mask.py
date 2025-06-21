import os
import numpy as np
import torch
from PIL import Image

class LoadMaskImage:
    """
    Loads a mask image for use in ComfyUI workflows.
    The mask should be a grayscale image where white areas (255) represent
    the areas to keep/process and black areas (0) are ignored/masked out.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 127.0, "min": 0.0, "max": 255.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_mask"
    CATEGORY = "mask"
    
    def load_mask(self, image_path, invert=False, threshold=127.0):
        """
        Load an image from disk and convert it to a mask tensor.
        
        Args:
            image_path: Path to the mask image
            invert: Whether to invert the mask (black becomes white, white becomes black)
            threshold: Threshold value for converting grayscale to binary (0-255)
            
        Returns:
            torch.Tensor: Binary mask tensor
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Mask image not found: {image_path}")
        
        # Load the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Convert to numpy array
        mask_np = np.array(image)
        
        # Apply threshold to convert to binary
        binary_mask = (mask_np > threshold).astype(np.float32)
        
        # Invert if requested
        if invert:
            binary_mask = 1.0 - binary_mask
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(binary_mask)
        
        # Add batch dimension if needed
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return (mask_tensor,)

# This part registers your node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadMaskImage": LoadMaskImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMaskImage": "Load Mask Image"
}

# For debugging
if __name__ == "__main__":
    # Example usage
    loader = LoadMaskImage()
    mask = loader.load_mask("mask.png")
    print(f"Loaded mask with shape: {mask[0].shape}")