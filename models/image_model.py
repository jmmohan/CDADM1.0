# image_model.py
import torch
import torchvision.models as models

def get_image_model():
    """
    Load a pre-trained ResNet model for image classification.
    
    Returns:
        model: Pre-trained ResNet model.
    """
    model = models.resnet18(pretrained=True)
    model.eval()
    return model