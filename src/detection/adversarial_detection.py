# adversarial_detection.py
import numpy as np

def detect_adversarial(input_data, model, threshold=1.0):
    """
    Detect adversarial samples by analyzing gradient norms.
    
    
    Args:
        input_data (np.ndarray): Input samples.
        model (object): Trained ML model.
        threshold (float): Gradient norm threshold for adversarial detection.
        
    Returns:
        np.ndarray: Boolean array indicating adversarial samples.
    """
    gradients = np.abs(np.gradient(model.predict(input_data)))
    norms = np.linalg.norm(gradients, axis=-1)
    return norms > threshold