import torchvision.models as models
import torch.nn as nn

def load_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.eval()
        return model
    else:
        raise ValueError("Model tifak didukung...")
