import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
import torch
import torch.nn as nn
from skimage.transform import resize
import os

def load_image(file_path):
    """Load medical image from file path."""
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        return load_nifti(file_path)
    elif file_path.endswith('.dcm'):
        return load_dicom(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_nifti(file_path):
    """Load NIfTI image."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def load_dicom(file_path):
    """Load DICOM image."""
    ds = pydicom.dcmread(file_path)
    data = ds.pixel_array
    return data

def preprocess_image(image_data):
    """Preprocess medical image data."""
    # Normalize to [0, 1]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # Resize if necessary (assuming 256x256 is our target size)
    if image_data.shape[0] != 256 or image_data.shape[1] != 256:
        image_data = resize(image_data, (256, 256))
    
    # Add channel dimension
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=0)
    
    # Convert to tensor
    image_tensor = torch.FloatTensor(image_data)
    return image_tensor

def save_image(image_data, output_path):
    """Save processed image."""
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().numpy()
    
    # Remove channel dimension if present
    if len(image_data.shape) == 3 and image_data.shape[0] == 1:
        image_data = image_data.squeeze(0)
    
    # Convert to PIL Image and save
    image = Image.fromarray((image_data * 255).astype(np.uint8))
    image.save(output_path)

class StyleTransferModel(nn.Module):
    """Base class for style transfer models."""
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # TODO: Initialize your GAN model here
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

def load_model(model_path):
    """Load pre-trained model."""
    model = StyleTransferModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model

def process_image(model, image_data, style_type):
    """Process image with the selected style transfer."""
    # Preprocess image
    processed_image = preprocess_image(image_data)
    
    # Add batch dimension
    processed_image = processed_image.unsqueeze(0)
    
    # Process with model
    with torch.no_grad():
        output = model(processed_image)
    
    return output.squeeze(0) 