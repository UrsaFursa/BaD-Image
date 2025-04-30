import os
import torch
import torchvision.transforms as transforms
import cv2

def load_images(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])
    images = []
    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        images.append(img)
    return torch.stack(images)

def get_layer_activations(model, images, layer_name):
    layer_output = []
    def hook_fn(module, input, output):
        layer_output.append(output.detach())

    handle = dict([*model.named_modules()])[layer_name].register_forward_hook(hook_fn)
    model(images)
    handle.remove()
    return layer_output[0].mean(dim=[2, 3])
