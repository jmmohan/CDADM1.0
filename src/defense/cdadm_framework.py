# cdadm_framework.py
import numpy as np

from detection.adversarial_detection import detect_adversarial
from feature_mapping.pca_feature_mapping import map_features
from defense.adaptive_noise_injection import adaptive_noise_injection

def cdadm_framework(input_data, model, epsilon=0.1, threshold=1.0):
    """
    Cross-Domain Adaptive Defense Mechanism (CDADM).
    
    Args:
        input_data (np.ndarray): Input samples.
        model (object): Trained ML model.
        epsilon (float): Initial noise magnitude.
        threshold (float): Gradient norm threshold.
        
    Returns:
        dict: Results including detection and defended predictions.
    """
    adversarial_flags = detect_adversarial(input_data, model, threshold)
    universal_features = map_features(input_data)
    if np.any(adversarial_flags):
        input_data = adaptive_noise_injection(input_data, epsilon)
    predictions = model.predict(input_data)
    return {
        "adversarial_flags": adversarial_flags,
        "universal_features": universal_features,
        "predictions": predictions
    }
