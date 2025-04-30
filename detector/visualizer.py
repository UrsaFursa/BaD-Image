import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from .utils import load_images
from torch.autograd import Function

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer_module = dict([*model.named_modules()])[target_layer]
        target_layer_module.register_forward_hook(self.forward_hook)
        target_layer_module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        for i in range(len(pooled_grad)):
            activation[i, :, :] *= pooled_grad[i]

        heatmap = activation.mean(dim=0).detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap

def generate_saliency(model, image_path):
    images = load_images(image_path)
    cam = GradCAM(model, 'layer4')
    for i, img in enumerate(images):
        heatmap = cam.generate(img.unsqueeze(0))
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Saliency map {i}")
        plt.show()
