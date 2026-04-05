import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import os
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook to get gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(output)

        self.model.zero_grad()
        output[0, target_class].backward()

        # Weight the activations by the gradients
        # Use mean over the spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam * 0.0 # avoid division by zero
        
        return cam.detach().cpu().numpy()

def apply_heatmap(image_path, heatmap):
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply color map
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_img, 0.4, 0)
    return superimposed_img

if __name__ == "__main__":
    # Example usage (to be used in dashboard)
    pass
