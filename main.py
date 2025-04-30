from detector.model_loader import load_model
from detector.analyzer import analyze_backdoor
from detector.visualizer import generate_saliency
from config import *

if __name__ == "__main__":
    model = load_model(model_name)
    analyze_backdoor(model, data_path_clean, data_path_suspect, layer_to_analyze)
    generate_saliency(model, data_path_suspect)
