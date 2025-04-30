import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from .utils import load_images, get_layer_activations

def analyze_backdoor(model, clean_path, suspect_path, layer_name):
    clean_imgs = load_images(clean_path)
    suspect_imgs = load_images(suspect_path)

    clean_act = get_layer_activations(model, clean_imgs, layer_name)
    suspect_act = get_layer_activations(model, suspect_imgs, layer_name)

    pca = PCA(n_components=2)
    combined = torch.cat([clean_act, suspect_act]).numpy()
    proj = pca.fit_transform(combined)

    print("PCA projection of activations (2D):")
    print(proj)
